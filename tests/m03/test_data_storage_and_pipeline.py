import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from affect_geometry.classification.protocol import EMOTIONS, digest, prompt, score
from affect_geometry.m03.configuration import Config
from affect_geometry.m03.data import render, select_unique, selections, validate_baseline
from affect_geometry.m03.pipeline import cached, extract, pilot, protocol
from affect_geometry.m03.smoke import prepare_smoke, smoke, tiny_bundle
from affect_geometry.m03.storage import ScoreStore, file_hash, frozen_json


def source_row(config, index, split="test", emotion="joy"):
    identity = f"{config.dataset_id}@{config.dataset_revision}"
    row = dict(
        id=str(index),
        sample_id=digest(f"{identity}:{split}:{index}"),
        source_split=split,
        text=f"Example {index}",
        emotion=emotion,
        label_id=EMOTIONS.index(emotion),
        labels=[EMOTIONS.index(emotion)],
    )
    payload, order = prompt(row)
    return dict(
        row,
        **score("joy", emotion),
        raw_response="joy",
        status="completed",
        text_hash=digest(row["text"]),
        rendered_prompt=payload,
        label_order=order,
        prompt_hash=digest(payload),
    )


def baseline_manifest(config):
    return dict(
        model_id=config.model_id,
        model_revision=config.model_revision,
        dataset_id=config.dataset_id,
        dataset_revision=config.dataset_revision,
        seed=config.seed,
        dtype=config.dtype,
        chat=False,
        max_new_tokens=10,
        protocol="rq2_single_label_v1",
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("correct", False),
        ("prompt_hash", "wrong"),
        ("text_hash", "wrong"),
        ("sample_id", "wrong"),
        ("status", "error"),
        ("label_id", 0),
        ("was_truncated", True),
        ("predicted_emotion", "sadness"),
        ("predicted_emotion", np.nan),
    ],
)
def test_baseline_identity_prompt_parser_and_completeness_checks(field, value):
    config = Config()
    row = source_row(config, 1)
    validate_baseline([row], baseline_manifest(config), config, check_counts=False)
    with pytest.raises(ValueError):
        validate_baseline(
            [dict(row, **{field: value})], baseline_manifest(config), config, check_counts=False
        )


def test_baseline_accepts_missing_prediction_for_invalid_response():
    config = Config()
    row = source_row(config, 1)
    response = "invalid response"
    row.update(score(response, row["emotion"]))
    row.update(raw_response=response, predicted_emotion=np.nan)
    validate_baseline([row], baseline_manifest(config), config, check_counts=False)


def test_baseline_exact_population_and_provenance():
    config = Config()
    row = source_row(config, 1)
    with pytest.raises(ValueError, match="counts"):
        validate_baseline([row], baseline_manifest(config), config)
    with pytest.raises(ValueError, match="manifest mismatch"):
        validate_baseline([row], dict(baseline_manifest(config), dtype="float32"), config)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_baseline([row, row], baseline_manifest(config), config)


def test_training_selection_is_unique_deterministic_and_gold_independent():
    config = Config()
    rows = [source_row(config, i, "train") for i in range(6)]
    rows.append(dict(rows[0], sample_id=digest("duplicate")))
    selected = select_unique(rows, 10, 42)
    assert len(selected) == 6
    ids = [r["sample_id"] for r in selected]
    mutated = [dict(r, emotion="anger", label_id=2, correct=False) for r in reversed(rows)]
    assert [r["sample_id"] for r in select_unique(mutated, 10, 42)] == ids
    test = dict(source_row(config, 99), text_hash=rows[0]["text_hash"])
    parts, groups = selections(rows + [test], config)
    assert all(r["source_split"] == "train" for r in parts["reference"])
    assert groups.loc[groups.text_hash == test["text_hash"], "cross_split_duplicate"].all()
    assert parts["test"] == [test]


def test_alternate_prompts_preserve_text_and_full_label_menu():
    config = Config()
    row = source_row(config, 1)
    assert render(row, "canonical", config) == row["rendered_prompt"]
    alternate = render(row, "alternate_menu", config)
    assert ", ".join(reversed(row["label_order"])) in alternate
    assert row["text"] in alternate
    assert "{" not in render(row, "alternate_prompt", config)


@pytest.mark.parametrize(
    "changes",
    [
        dict(model_revision="main"),
        dict(prefix_batch_size=0),
        dict(bootstrap_replicates=0),
        dict(tuned_indexing="guess"),
        dict(test_history=""),
        dict(output_root="tmp/results"),
    ],
)
def test_invalid_frozen_configuration(changes):
    with pytest.raises(ValueError):
        replace(Config(), **changes)


def test_no_validation_tuning_cli_or_config(tmp_path):
    from affect_geometry.m03.__main__ import parser

    path = tmp_path / "bad.json"
    path.write_text(json.dumps(dict(select_layer_using_validation=True)))
    with pytest.raises(ValueError, match="validation tuning"):
        Config.load(path)
    with pytest.raises(SystemExit):
        parser().parse_args(["extract", "--run-id", "x", "--phase", "validation"])
    with pytest.raises(ValueError, match="run ID"):
        Config().root("../escape")


def test_atomic_context_checked_score_resume_and_full_28_label_rows(tmp_path):
    keys = [("native", 2), ("logit", 0)]
    store = ScoreStore(tmp_path, dict(revision="pinned"), keys)
    values = dict(
        label_scores=np.zeros((2, 28)),
        stop_scores=np.ones((2, 28)),
        token_counts=np.full((2, 28), 2),
    )
    identity = dict(prompt_hash="x", token_paths_hash="p", variant="canonical")
    assert store.get("sample1", identity) is None
    store.put("sample1", identity, values)
    # JSON round trips tuple layer keys safely.
    store = ScoreStore(tmp_path, dict(revision="pinned"), keys)
    np.testing.assert_array_equal(store.get("sample1", identity)["stop_scores"], 1)
    with pytest.raises(ValueError, match="identity differs"):
        store.get("sample1", dict(identity, prompt_hash="different"))
    with pytest.raises(ValueError, match="Frozen configuration differs"):
        ScoreStore(tmp_path, dict(revision="changed"), keys)
    with pytest.raises(ValueError, match="28-label"):
        store.put("sample2", identity, dict(values, label_scores=np.zeros((2, 27))))
    with pytest.raises(ValueError, match="nonfinite"):
        store.put("sample2", identity, dict(values, stop_scores=np.full((2, 28), np.nan)))
    assert not list(tmp_path.glob("*.tmp"))
    frozen_json(tmp_path / "tuple.json", dict(keys=keys))
    frozen_json(tmp_path / "tuple.json", dict(keys=keys))


def test_partial_extraction_resume_no_repeated_backbone_and_strict_calibration_order(
    tmp_path, monkeypatch
):
    from affect_geometry.m03.pipeline import calibrate

    monkeypatch.chdir(tmp_path)
    config, root = prepare_smoke("resume")
    bundle = tiny_bundle(config)
    pilot(config, "resume", bundle=bundle)
    with pytest.raises(ValueError, match="thresholds"):
        extract(config, "resume", phase="test", bundle=bundle)
    extract(config, "resume", phase="reference", bundle=bundle)
    calibrate(config, "resume")
    extract(config, "resume", phase="test", variant="canonical", bundle=bundle)

    def fail(*args, **kwargs):
        raise AssertionError("A complete cache must not rerun the backbone")

    original_forward = bundle[0].model.forward
    monkeypatch.setattr(bundle[0].model, "forward", fail)
    extract(config, "resume", phase="test", variant="canonical", bundle=bundle)
    times = [json.loads(p.read_text()) for p in (root / "runtime").glob("test_canonical*.json")]
    assert any(t["reused"] == 3 and t["backbone_batches"] == 0 for t in times)
    shard = next((root / "scores" / "test" / "canonical").glob("*.npz"))
    retained = {p: file_hash(p) for p in shard.parent.glob("*.npz") if p != shard}
    shard.unlink()
    with pytest.raises(ValueError, match="Missing score shard"):
        cached(root, "test", "canonical", config)
    monkeypatch.setattr(bundle[0].model, "forward", original_forward)
    extract(config, "resume", phase="test", variant="canonical", bundle=bundle)
    assert shard.exists()
    assert all(file_hash(path) == checksum for path, checksum in retained.items())
    times = [json.loads(p.read_text()) for p in (root / "runtime").glob("test_canonical*.json")]
    assert any(t["newly_extracted"] == 1 and t["replayed_for_batch_context"] == 2 for t in times)
    with pytest.raises(ValueError, match="config or implementation changed"):
        protocol(root, replace(config, label_prefix=""))


def test_offline_tiny_gemma_full_workflow_and_exported_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = smoke("end-to-end")
    assert (root / "SYNTHETIC.json").is_file()
    manifest = json.loads((root / "analysis_manifest.json").read_text())
    assert manifest["complete"] and manifest["bootstrap_replicates"] == 32
    checks = json.loads((root / "pilot_checks.json").read_text())
    assert checks["complete"] and checks["gold_labels_used"] is False
    headline = pd.read_csv(root / "tables" / "main_results.csv")
    assert len(headline) == 8 and headline.primary.sum() == 6
    assert set(headline.lens) == {"logit", "tuned", "jacobian", "native", "greedy"}
    metrics = pd.read_csv(root / "tables" / "metrics.csv")
    main = metrics.loc[(metrics.variant == "canonical") & (metrics.score_mode == "total")]
    assert len(main) == 42
    assert set(metrics.score_mode) == {"total", "label_only", "length_normalized"}
    assert set(metrics.variant) == {"canonical", "alternate_menu", "alternate_prompt"}
    assert np.isfinite(metrics.accuracy).all() and np.isfinite(metrics.macro_f1).all()
    for base in ("native", "greedy"):
        np.testing.assert_allclose(
            metrics[f"delta_accuracy_{base}_pp"],
            100 * (metrics[f"repairs_{base}"] - metrics[f"damages_{base}"]) / metrics.n,
        )
    scores = pd.read_parquet(root / "tables" / "complete_label_scores_canonical.parquet")
    assert len(scores) == 3 * 8 * 28
    assert (scores.groupby(["sample_id", "lens", "layer"]).size() == 28).all()
    np.testing.assert_allclose(scores.total_score, scores.label_token_score + scores.stop_score)
    predictions = pd.read_parquet(root / "tables" / "predictions.parquet")
    assert len(predictions) == metrics.n.sum()
    assert predictions.loc[predictions.arm != "greedy", "prediction"].isin(EMOTIONS).all()
    thresholds = json.loads((root / "thresholds.json").read_text())
    assert thresholds["gold_labels_used"] is False and len(thresholds["thresholds"]) == 7
    assert (root / "tables" / "calibration.csv").is_file()
    assert (root / "tables" / "mask_exclusion.csv").is_file()
    sensitivity = pd.read_csv(root / "tables" / "scoring_and_prompt_sensitivity.csv")
    assert set(sensitivity.variant) == {"canonical", "alternate_menu", "alternate_prompt"}
    assert sensitivity.n_changed.between(0, sensitivity.n).all()
    contrast = metrics.loc[metrics.kind == "contrast"]
    assert contrast.delta_accuracy_same_layer_direct_pp.notna().all()
    within = metrics.loc[metrics.kind == "within"]
    assert within.delta_accuracy_upper_direct_pp.notna().all()
    assert (root / "tables" / "trajectories.csv").is_file()
    assert len(list((root / "plots").glob("*.png"))) >= 15
    for name in (
        "direct_curves",
        "contrast_curves",
        "class_metrics",
        "confusion",
        "rank_trajectories",
        "vocabulary_kl",
        "agreement",
        "repairs",
    ):
        assert (root / "plot_data" / f"{name}.csv").is_file()
