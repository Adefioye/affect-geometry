"""End-to-end synthetic orchestration, leakage boundaries, and resume/integrity checks."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from affect_geometry.classification.protocol import EMOTIONS, PROTOCOL, digest, prompt, score
from affect_geometry.data.neuron_transfer import DonorRetriever, build_cohorts, normalize_text
from affect_geometry.experiments.neuron_transfer import NeuronTransferExperiment
from affect_geometry.experiments.neuron_transfer_config import NeuronTransferConfig
from affect_geometry.persistence.correctness_geometry import checksum
from affect_geometry.persistence.neuron_transfer import RecordLedger
from affect_geometry.reporting.neuron_transfer import outcome_metrics, paired_bootstrap


def fake_scores(info, patch=None):
    gold = info["emotion"]
    rival = "joy" if gold == "anger" else "anger"
    base = -0.2 if "error" in info["text"] else 1.0
    active = {0, 1} if gold == "anger" else {2, 3}
    current = 0.1 if "error" in info["text"] else 1.0
    if patch:
        base += (
            2
            * patch["alpha"]
            * sum(
                patch["donor"][(layer, channel)] - current
                for layer, channel in patch["members"]
                if channel in active
            )
        )
    totals = {label: -4.0 for label in EMOTIONS}
    totals[gold], totals[rival] = base, 0.0
    return {
        label: dict(
            total=value,
            label_score=value + 0.1,
            stop_score=-0.1,
            label_tokens=1,
            token_ids=[10 + EMOTIONS.index(label), 2],
        )
        for label, value in totals.items()
    }


class FakeRunner:
    """Deterministic policy fixture, not a substitute for the separate native-model tests."""

    def __init__(self, config, manifest):
        self.config, self.manifest = config, manifest
        self.widths = {0: 8}
        self.scorer = SimpleNamespace(
            metadata=lambda: {v: [10 + i, 2] for i, v in enumerate(EMOTIONS)}
        )

    def encode(self, row, menu=None):
        text, order = prompt(row, self.manifest["seed"], label_order=menu)
        if menu is None:
            assert digest(text) == row["prompt_hash"]
        return dict(
            row, ids=[1, 80, 81], prompt_position=2, prompt_hash=digest(text), label_order=order
        )

    def scores(self, info, labels=None, patch=None):
        result = fake_scores(info, patch)
        return {label: result[label] for label in (labels or EMOTIONS)}, []

    def prompt_logits(self, info, patch=None):
        return np.array([v["total"] for v in fake_scores(info, patch).values()])

    def generate(self, info, gold, patch=None):
        values = fake_scores(info, patch)
        label = max(EMOTIONS, key=lambda c: values[c]["total"])
        return dict(
            **score(label, gold),
            raw_response=label,
            generated_token_ids=[10 + EMOTIONS.index(label), 2],
            generation_finish_reason="eos",
            edit_audit=[],
        )

    def replay(self, row, generated):
        assert generated["raw_response"] == row["raw_response"]
        assert generated["generated_token_ids"] == list(row["generated_token_ids"])

    def attribution(self, info, gold, totals=None):
        active = (0, 1) if gold == "anger" else (2, 3)
        if totals is not None:
            totals[0] = np.array([1.0 if j in active else 0.0 for j in range(8)])
        return [
            dict(
                layer=0,
                channel=j,
                activation=1.0,
                gradient=1.0,
                signed_contribution=1.0,
                importance=1.0,
                rank=i,
            )
            for i, j in enumerate(active)
        ]

    def coordinates(self, info, members):
        active = (0, 1) if info["emotion"] == "anger" else (2, 3)
        amplitude = 0.1 if "error" in info["text"] else 1.0
        return {tuple(v): amplitude if v[1] in active else 0.0 for v in members}

    def patch(self, members, donor, alpha=0.25):
        return dict(members=members, donor=donor, alpha=alpha)

    def finite_difference(self, info, gold, record):
        return dict(gradient=1.0, finite_difference=1.0)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "exp2"
    (root / "dataset").mkdir(parents=True)
    frames = []
    for split, n in (("train", 12), ("validation", 2), ("test", 4)):
        rows = []
        for i in range(n):
            gold = "anger" if i % 2 == 0 else "joy"
            error = split == "test" and i < 2
            text = f"{gold} {'error' if error else 'clear'} event {split} {i} " + (
                "storm knocked everything over"
                if gold == "anger"
                else "friends brought wonderful news"
            )
            row = dict(
                sample_id=f"{split}-{i:02}",
                source_split=split,
                source_row=i,
                text=text,
                text_hash=digest(text),
                emotion=gold,
                status="completed",
            )
            rendered, order = prompt(row, 42)
            info = dict(row)
            label = max(EMOTIONS, key=lambda c: fake_scores(info)[c]["total"])
            row.update(score(label, gold))
            row.update(
                raw_response=label,
                generated_token_ids=[10 + EMOTIONS.index(label), 2],
                rendered_prompt=rendered,
                prompt_hash=digest(rendered),
                label_order=order,
                input_token_count=3,
                was_truncated=False,
                model_revision="a" * 40,
                tokenizer_revision="a" * 40,
                protocol_version=PROTOCOL,
                run_id="exp2-fixture",
            )
            rows.append(row)
        frame = pd.DataFrame(rows)
        frame.to_parquet(root / "dataset" / f"{split}.parquet", index=False)
        frames.append(frame)
    import json

    manifest = dict(
        model_id="allenai/OLMo-2-1124-7B-Instruct",
        model_revision="a" * 40,
        dataset_revision="b" * 40,
        protocol=PROTOCOL,
        chat=False,
        seed=42,
        max_new_tokens=10,
        dtype="float32",
        run_id="exp2-fixture",
        sample_ids_hash=digest("\n".join(pd.concat(frames).sample_id)),
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    config = NeuronTransferConfig(
        str(root),
        "fixture",
        output_root=str(tmp_path / "outputs"),
        dtype="float32",
        allow_partial_source=True,
        discovery_per_emotion=6,
        selection_per_emotion=6,
        smoke_size=2,
        top_k=2,
        minimum_count=1,
        null_replicates=3,
        null_sweeps=3,
        edge_threshold=0.1,
        resolutions=(0.1,),
        community_budget=2,
        minimum_effect_examples=2,
        bootstrap_replicates=30,
    )
    return config, pd.concat(frames, ignore_index=True), manifest


def test_full_pipeline_with_actual_leiden_resume_and_tamper_detection(source):
    pytest.importorskip("leidenalg")
    config, _, _ = source
    experiment = NeuronTransferExperiment(config, FakeRunner)
    root = experiment.run()
    shortlist = experiment.store.read_json("shortlist.json")
    assert 1 <= len(shortlist["communities"]) <= 2
    predictions = pd.read_parquet(root / "predictions.parquet")
    assert set(predictions.arm) == {"baseline", "community", "random", "unclustered", "wrong_label"}
    assert len(predictions) == 4 * (1 + 4 * len(shortlist["communities"]))
    assert set(predictions.sample_id) == {f"test-{i:02}" for i in range(4)}
    assert (root / "path_scores.parquet").exists()
    assert (root / "edit_norms.parquet").exists()
    assert (root / "figures/accuracy_deltas.png").exists()
    budget = experiment.store.read_json("compute_budget.json")
    assert budget["maximum_edited_generation_conditions"] == 16 * len(shortlist["communities"])
    profile = experiment.store.read_json("smoke_profile.json")
    assert {"generation", "attribution", "baseline_28_path_scoring"} <= set(
        profile["component_seconds"]
    )
    diagnostics = pd.read_csv(root / "score_diagnostics.csv")
    assert set(diagnostics.outcome_group) == {"all", "baseline_correct", "baseline_incorrect"}
    ablations = pd.read_csv(root / "heldout_ablation_summary.csv")
    assert len(ablations) and ablations.n.ge(1).all()
    # A community can support one label while harming another; preserve both signs.
    assert ablations.mean_net_margin_effect.min() == -4.0
    assert ablations.mean_net_margin_effect.max() == 4.0
    assert (ablations.mean_net_gold_stop_effect == 0).all()
    means = pd.read_parquet(root / "importance_means.parquet")
    assert len(means) == 8  # Includes non-top-k neurons for the unclustered comparator.
    assert means.loc[means.channel.eq(0), "importance"].item() == 0.5
    before = checksum(root / "predictions.parquet")
    NeuronTransferExperiment(config, FakeRunner).run()
    assert checksum(root / "predictions.parquet") == before
    (root / "controls.json").write_text("{}")
    with pytest.raises(ValueError, match="stale"):
        NeuronTransferExperiment(config, FakeRunner).run("test")


def test_test_outcomes_cannot_change_training_cohorts_or_wrong_label_mapping(source):
    config, rows, _ = source
    first = build_cohorts(rows, config)
    changed = rows.copy()
    mask = changed.source_split.eq("test")
    changed.loc[mask, "raw_response"] = "neutral"
    for i in changed.index[mask]:
        changed.loc[i, "correct"] = changed.loc[i, "emotion"] == "neutral"
    second = build_cohorts(changed, config)
    assert first[0].sample_id.tolist() == second[0].sample_id.tolist()
    assert first[1].sample_id.tolist() == second[1].sample_id.tolist()
    assert first[4] == second[4]
    assert all(v.startswith("train-") for v in first[0].sample_id)


def test_donors_exclude_normalized_duplicates_and_derive_confusions_from_train_errors(source):
    config, rows, _ = source
    rows = rows.reset_index(drop=True)
    train_index = rows[rows.source_split.eq("train")].index[0]
    rows.loc[train_index, "raw_response"] = "fear"
    rows.loc[train_index, "predicted_emotion"] = "fear"
    rows.loc[train_index, "correct"] = False
    discovery, _, _, _, alternative = build_cohorts(rows, config)
    assert alternative["anger"] == "fear"
    retriever = DonorRetriever(discovery)
    row = discovery.iloc[0].to_dict()
    duplicate = dict(row, sample_id="different-id", text="  " + row["text"].upper() + "  ")
    eligible, excluded = retriever.ranked(duplicate, row["emotion"])
    assert row["sample_id"] not in {r["donor"]["sample_id"] for r in eligible}
    assert excluded
    assert normalize_text(row["text"]) == normalize_text(duplicate["text"])


def test_test_gold_labels_do_not_filter_training_duplicate_groups(source):
    config, rows, _ = source
    initial = build_cohorts(rows, config)[0]
    changed = rows.copy()
    train_row = changed[changed.source_split.eq("train")].iloc[0]
    test_index = changed[changed.source_split.eq("test")].index[0]
    changed.loc[test_index, "text"] = train_row.text
    changed.loc[test_index, "text_hash"] = train_row.text_hash
    changed.loc[test_index, "emotion"] = "joy" if train_row.emotion == "anger" else "anger"
    changed.loc[test_index, "raw_response"] = changed.loc[test_index, "emotion"]
    changed.loc[test_index, "correct"] = True
    after = build_cohorts(changed, config)[0]
    assert initial.sample_id.tolist() == after.sample_id.tolist()


def test_atomic_attribution_resume_sums_and_provenance(tmp_path):
    path = tmp_path / "ledger.sqlite"
    with RecordLedger(path, {"run": 1}) as ledger:
        ledger.put_attribution("one", {"rows": []}, {0: np.array([1, 2, 3])})
        ledger.put_attribution("one", {"rows": []}, {0: np.array([100, 100, 100])})
        ledger.put_attribution("two", {"rows": []}, {0: np.array([3, 4, 5])})
        assert ledger.importance_means().importance.tolist() == [2, 3, 4]
    with pytest.raises(ValueError, match="provenance"):
        RecordLedger(path, {"run": 2})


def test_metrics_keep_abstentions_invalid_outputs_and_grouped_pairs():
    base = pd.DataFrame(
        dict(
            sample_id=["a", "b", "c"], correct=[True, False, False], valid_label=[True, True, False]
        )
    )
    after = pd.DataFrame(
        dict(
            sample_id=["c", "a", "b"],
            emotion=["joy"] * 3,
            predicted_emotion=["joy", "anger", "anger"],
            correct=[True, False, False],
            valid_label=[True] * 3,
            donor_id=["donor", "donor", None],
        )
    )
    metrics, _ = outcome_metrics(after, base)
    assert metrics["repaired"] == metrics["damaged"] == 1
    assert metrics["accuracy_delta"] == 0
    assert metrics["format_repairs"] == 1 and metrics["n"] == 3
    assert metrics["donor_coverage"] == pytest.approx(2 / 3)
    values = paired_bootstrap([[1, 1], [-1, -1], [0, 0]], ["duplicate", "duplicate", "other"], 50)
    assert np.array_equal(values[:, 0], values[:, 1])
    assert np.all(values == 0)  # The duplicate pair is always resampled together.


def test_config_rejects_hidden_protocol_changes(source):
    config, _, _ = source
    for changes in (
        {"community_budget": 21},
        {"alpha": 1},
        {"selection_per_emotion": 11},
        {"run_id": "../escape"},
    ):
        with pytest.raises(ValueError):
            replace(config, **changes)


def test_no_discovery_support_stops_without_loading_a_model(source):
    config, _, _ = source
    path = config.classification_dir + "/dataset/train.parquet"
    train = pd.read_parquet(path)
    for i, row in train.iterrows():
        wrong = "joy" if row.emotion == "anger" else "anger"
        for key, value in score(wrong, row.emotion).items():
            train.loc[i, key] = value
        train.loc[i, "raw_response"] = wrong
    train.to_parquet(path, index=False)

    def prohibited_runner(*args):
        raise AssertionError("Empty discovery must not load model weights")

    experiment = NeuronTransferExperiment(config, prohibited_runner)
    root = experiment.run()
    assert experiment.store.read_json("shortlist.json")["communities"] == []
    assert not (root / "checkpoints/test.json").exists()
    assert "Test interventions were not run" in (root / "report.md").read_text()


def test_interrupted_test_resumes_without_duplicating_conditions(source):
    pytest.importorskip("leidenalg")
    config, _, _ = source
    experiment = NeuronTransferExperiment(config, FakeRunner)
    for stage in ("prepare", "smoke", "attribute", "graph", "select", "freeze"):
        experiment.run(stage)
    original = experiment.runner.generate
    calls = [0]

    def interrupt(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 4:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    experiment.runner.generate = interrupt
    with pytest.raises(RuntimeError, match="simulated"):
        experiment.run("test")
    assert not (config.root / "checkpoints/test.json").exists()
    with pytest.raises(ValueError, match="stale"):
        experiment.run("report")
    resumed = NeuronTransferExperiment(config, FakeRunner)
    resumed.run("test")
    resumed.run("report")
    rows = pd.read_parquet(config.root / "predictions.parquet")
    assert not rows.duplicated(["sample_id", "community_id", "arm"]).any()
    assert len(rows) == 4 * (1 + 4 * len(resumed.store.read_json("shortlist.json")["communities"]))
