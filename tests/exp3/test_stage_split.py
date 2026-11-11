import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_scientific_contracts import source_rows

from affect_geometry.activations.exp3_bundle import PreparedActivations, export_cohort_bundle
from affect_geometry.analysis.cohorts import balanced_cohorts
from affect_geometry.classification.protocol import EMOTIONS, MODELS
from affect_geometry.experiments.exp3_stages import (
    analysis_geometry_config,
    load_analysis_config,
    load_extraction_config,
    run_analysis,
)
from affect_geometry.persistence.correctness_geometry import checksum
from affect_geometry.persistence.exp3_hub import download_bundle, publish_stage

PROJECT = Path(__file__).resolve().parents[2]


@pytest.fixture
def prepared(tmp_path):
    rows = source_rows()
    cohorts = balanced_cohorts(rows, EMOTIONS, 42)
    rng = np.random.default_rng(51)
    centers = {label: rng.normal(size=10) for label in cohorts.labels}
    values = np.stack([centers[label] + rng.normal(scale=0.4, size=10) for label in rows.emotion])
    path = tmp_path / "source-layer.npy"
    np.save(path, values)
    root = tmp_path / "bundle"
    bundle = export_cohort_bundle(
        root,
        model="gemma2",
        run_id="fixture",
        cohorts=cohorts,
        source_manifest=dict(model_id=MODELS["gemma2"][0], model_revision="pinned", run_id="exp2"),
        source_audit=dict(full_population=False),
        prompt_samples=rows,
        layer_paths=[path],
        prompt_metadata=dict(hidden_size=10, n_layers=1),
        prompt_checksums={path.name: checksum(path)},
        seed=42,
    )
    return bundle, path


def test_raw_and_projected_samples_all_cohorts_and_roundtrip(prepared):
    bundle, path = prepared
    raw_source = np.load(path)
    source_ids = source_rows().sample_id.tolist()
    projections = {}
    raw_by_id = {}
    for cohort in ("correct", "incorrect", "all"):
        samples = pd.read_csv(bundle.root / f"cohorts/{cohort}/samples.csv")
        raw = np.load(bundle.root / f"cohorts/{cohort}/last_token/layer_000.npy")
        projected = bundle.projected(cohort, 0)
        prep = bundle.preprocessing(cohort, 0)
        positions = [source_ids.index(s) for s in samples.sample_id]
        assert np.array_equal(raw, raw_source[positions])
        expected = raw - prep["grand_mean"]
        expected -= (expected @ prep["components"].T) @ prep["components"]
        assert np.allclose(projected, expected, atol=2e-7)
        assert not np.allclose(np.linalg.norm(projected, axis=1), 1)
        for i, label in enumerate(bundle.cohorts.labels):
            assert np.allclose(
                projected[samples.emotion.eq(label)].mean(0), prep["R"][i], atol=2e-7
            )
        projections[cohort] = dict(zip(samples.sample_id, projected, strict=True))
        raw_by_id[cohort] = dict(zip(samples.sample_id, raw, strict=True))
    for cohort in ("correct", "incorrect"):
        for sample, raw in raw_by_id[cohort].items():
            assert np.array_equal(raw, raw_by_id["all"][sample])
        assert any(
            not np.allclose(value, projections["all"][sample])
            for sample, value in projections[cohort].items()
        )
    restored = PreparedActivations.open(bundle.root, expected_model="gemma2")
    assert restored.cohorts.samples.sample_id.tolist() == bundle.cohorts.samples.sample_id.tolist()
    with pytest.raises(ValueError, match="different model"):
        PreparedActivations.open(bundle.root, expected_model="olmo2")


@pytest.mark.parametrize("skip_permutations", [False, True])
def test_downstream_uses_saved_projection_without_source_or_refitting(
    prepared, tmp_path, monkeypatch, skip_permutations
):
    bundle, source = prepared
    source.unlink()  # Original extraction files and EXP2 records are not downstream dependencies.
    from affect_geometry.experiments import correctness_geometry

    def forbidden(*args, **kwargs):
        raise AssertionError("Downstream must not resample or refit neutral PCA")

    monkeypatch.setattr(correctness_geometry, "balanced_cohorts", forbidden)
    monkeypatch.setattr(correctness_geometry, "preprocess", forbidden)
    if skip_permutations:
        monkeypatch.setattr(correctness_geometry, "permutation_schedule", forbidden)
        monkeypatch.setattr(correctness_geometry, "infer", forbidden)
    config = load_analysis_config(PROJECT / "configs/exp3/gemma2.json")
    config.update(
        smoke=True,
        run_id="analysis-fixture",
        layers=[0],
        permutations=1,
        isomap_neighbors=[3],
        diffusion_neighbors=[3],
        diffusion_scale_ranks=[2],
        diffusion_times=[1],
        neighborhoods=[1],
    )
    config["activation_source"]["local_path"] = str(bundle.root)
    config["nrc_path"] = str(PROJECT / config["nrc_path"])
    config["nrc_provenance"] = str(PROJECT / config["nrc_provenance"])
    output = run_analysis(config, tmp_path, publish=False, skip_permutations=skip_permutations)
    root = Path(output["path"])
    scores = pd.read_csv(root / "metrics/scores.csv")
    assert len(scores) == 18 and scores.status.eq("ok").all()
    assert (root / "plot_data/held_out_predictions.csv").exists()
    assert (root / "plots/cka-all-2d.png").exists()
    if skip_permutations:
        import json

        assert config["run_id"] == "analysis-fixture"
        assert root.name == "analysis-fixture-descriptive"
        assert scores.inference_status.eq("skipped").all()
        assert scores.raw_p.isna().all() and scores.corrected_p.isna().all()
        assert json.loads((root / "manifest.json").read_text())["inference_label"] == (
            "descriptive_permutations_skipped"
        )
        for dim in (1, 2):
            status = json.loads((root / f"inference/{dim}d/status.json").read_text())
            assert status["status"] == "skipped" and status["completed"] == 0
            assert not (root / f"inference/{dim}d/nulls.npz").exists()
        emergence = json.loads((root / "metrics/emergence.json").read_text())
        assert all(row["significance_status"] == "unavailable" for row in emergence)
        assert all(row["stable_significant_intervals"] is None for row in emergence)
        assert all(row["descriptive_peak_layer"] == 0 for row in emergence)
        metrics = pd.read_csv(root / "metrics/metrics.csv")
        assert {"trustworthiness_k1", "continuity_k1", "participation_ratio"} <= set(metrics.metric)
        assert (root / "checkpoints/run.json").exists()
    for cohort in ("correct", "incorrect", "all"):
        with np.load(root / f"preprocessing/{cohort}/layer_000.npz") as prep:
            assert np.array_equal(prep["R"], bundle.preprocessing(cohort, 0)["R"])


class FakeHub:
    token = "fake-test-token"

    def __init__(self):
        self.created, self.uploads = [], []

    def whoami(self):
        return {"name": "researcher"}

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def file_exists(self, **kwargs):
        return False

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)
        return SimpleNamespace(oid="commit123")

    def repo_info(self, **kwargs):
        return SimpleNamespace(sha="immutable456")


def test_publish_complete_bundle_and_analysis_plot_data(prepared):
    bundle, _ = prepared
    hub = FakeHub()
    receipt = publish_stage(
        bundle.root, repo_id="{username}/rq3-gemma2", path_in_repo="activations/fixture", api=hub
    )
    assert receipt["repo_id"] == "researcher/rq3-gemma2" and receipt["revision"] == "commit123"
    assert hub.created[0]["private"] is True
    files = hub.uploads[0]["allow_patterns"]
    assert "checkpoints/run.json" in files
    for cohort in ("correct", "incorrect", "all"):
        assert f"cohorts/{cohort}/last_token/layer_000.npy" in files
        assert f"cohorts/{cohort}/neutral_projected/layer_000.npy" in files
    # The publisher is stage-generic: downstream plot_data and all other covered files ship too.
    from affect_geometry.persistence.correctness_geometry import GeometryStore

    store = GeometryStore(bundle.root.parent / "analysis", dict(model="gemma2"))
    store.csv("plot_data/scores.csv", pd.DataFrame({"score": [0.5]}))
    store.csv("metrics/scores.csv", pd.DataFrame({"score": [0.5]}))
    store.finish()
    publish_stage(
        store.root, repo_id="{username}/rq3-gemma2", path_in_repo="analyses/fixture", api=hub
    )
    assert "plot_data/scores.csv" in hub.uploads[1]["allow_patterns"]
    (bundle.root / "cohorts/all/last_token/layer_000.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="incomplete or modified"):
        publish_stage(
            bundle.root,
            repo_id="{username}/rq3-gemma2",
            path_in_repo="activations/fixture",
            api=hub,
        )
    assert len(hub.uploads) == 2


@pytest.mark.parametrize("preferred_exists", [None, True, False])
def test_download_pins_commit_and_roundtrips_bundle(
    prepared, tmp_path, monkeypatch, preferred_exists
):
    import huggingface_hub

    bundle, _ = prepared
    calls = []
    checks = []
    source = dict(
        repo_id="{username}/rq3-gemma2",
        path_in_repo="activations/exp3-fixture",
        revision="main",
    )
    hub = FakeHub()
    if preferred_exists is not None:
        source["fallback_path_in_repo"] = "activations/rq3-fixture"

        def file_exists(**kwargs):
            checks.append(kwargs)
            return preferred_exists

        monkeypatch.setattr(hub, "file_exists", file_exists)
    expected_prefix = (
        source["fallback_path_in_repo"] if preferred_exists is False else source["path_in_repo"]
    )

    def download(**kwargs):
        calls.append(kwargs)
        assert kwargs["allow_patterns"] == [f"{expected_prefix}/**"]
        shutil.copytree(bundle.root, Path(kwargs["local_dir"]) / expected_prefix)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    root, provenance = download_bundle(source, tmp_path / "download", api=hub)
    restored = PreparedActivations.open(root)
    assert np.array_equal(restored.projected("all", 0), bundle.projected("all", 0))
    assert calls[0]["revision"] == provenance["revision"] == "immutable456"
    assert provenance["path_in_repo"] == expected_prefix
    if preferred_exists is not None:
        assert checks == [
            dict(
                repo_id="researcher/rq3-gemma2",
                repo_type="dataset",
                revision="immutable456",
                filename="activations/exp3-fixture/manifest.json",
            )
        ]


def test_configs_are_split_and_analysis_cannot_change_sampling():
    extraction = load_extraction_config(PROJECT / "configs/exp3/extract_activations.json")
    assert len(extraction["models"]) == 3 and "permutations" not in extraction
    for model in extraction["models"]:
        analysis = load_analysis_config(PROJECT / f"configs/exp3/{model}.json")
        assert "seed" not in analysis and "extraction" not in analysis and "models" not in analysis
        assert analysis["model"] == model
        assert analysis["publish"]["repo_id"] == extraction["models"][model]["publish"]["repo_id"]
        assert analysis_geometry_config(analysis, seed=99)["seed"] == 99


def test_unsupported_projection_keeps_raw_states_and_failure(prepared, tmp_path):
    from affect_geometry.embeddings.geometry import UnsupportedGeometry

    bundle, path = prepared
    np.save(path, np.zeros_like(np.load(path)))
    rows = source_rows()
    exported = export_cohort_bundle(
        tmp_path / "unsupported",
        model="gemma2",
        run_id="unsupported",
        cohorts=balanced_cohorts(rows, EMOTIONS, 42),
        source_manifest=bundle.manifest["source"],
        source_audit=dict(full_population=False),
        prompt_samples=rows,
        layer_paths=[path],
        prompt_metadata=dict(hidden_size=10),
        prompt_checksums={path.name: checksum(path)},
        seed=42,
    )
    for cohort in ("correct", "incorrect", "all"):
        assert (exported.root / f"cohorts/{cohort}/last_token/layer_000.npy").exists()
        assert not (exported.root / f"cohorts/{cohort}/neutral_projected/layer_000.npy").exists()
        with pytest.raises(UnsupportedGeometry, match="variance"):
            exported.projected(cohort, 0)


def test_upload_rejects_different_existing_remote_run(prepared, tmp_path, monkeypatch):
    import huggingface_hub

    bundle, _ = prepared
    remote = tmp_path / "remote-manifest.json"
    remote.write_text('{"different": true}')
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(remote))
    hub = FakeHub()
    hub.file_exists = lambda **kwargs: True
    with pytest.raises(ValueError, match="Remote run manifest differs"):
        publish_stage(
            bundle.root,
            repo_id="{username}/rq3-gemma2",
            path_in_repo="activations/fixture",
            api=hub,
        )
    assert not hub.uploads


def test_full_configs_preserve_unverified_pilot_status():
    for slug in ("gemma2", "apertus", "olmo2"):
        config = load_analysis_config(PROJECT / f"configs/exp3/{slug}.json")
        assert config["layers"] is None and config["permutations"] == 2000
        assert not config["smoke"] and config["settings_frozen"]
        assert not config["pilot_verified"]


def test_validate_remote_bundle_does_not_run_analysis(prepared, tmp_path, monkeypatch):
    from affect_geometry.experiments import correctness_geometry, exp3_stages

    bundle, _ = prepared
    config = load_analysis_config(PROJECT / "configs/exp3/gemma2.json")
    config.update(smoke=True, layers=[0], permutations=1)
    config["activation_source"]["local_path"] = str(tmp_path / "absent")
    config["nrc_path"] = str(PROJECT / config["nrc_path"])
    config["nrc_provenance"] = str(PROJECT / config["nrc_provenance"])
    monkeypatch.setattr(
        exp3_stages,
        "download_bundle",
        lambda *a: (bundle.root, {"revision": "pinned", "repo_id": "researcher/exp3-gemma2"}),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Validation must not run analysis or publish")

    monkeypatch.setattr(correctness_geometry, "run_geometry", forbidden)
    monkeypatch.setattr(exp3_stages, "publish_stage", forbidden)
    result = run_analysis(config, tmp_path, validate_only=True)
    assert result["status"] == "validated" and not result["analysis_started"]
    assert set(result["cohort_counts"]) == {"correct", "incorrect", "all"}
    assert result["hub"]["revision"] == "pinned"
    config["smoke"] = False
    with pytest.raises(ValueError, match="full-population"):
        run_analysis(config, tmp_path, validate_only=True)


def test_standalone_upload_all_outputs_without_activations(tmp_path, monkeypatch):
    from affect_geometry.experiments.exp3_stages import upload_analysis
    from affect_geometry.persistence import exp3_hub
    from affect_geometry.persistence.correctness_geometry import GeometryStore

    config = load_analysis_config(PROJECT / "configs/exp3/gemma2.json")
    root = tmp_path / "outputs/exp3/gemma2" / config["run_id"]
    store = GeometryStore(
        root, dict(model="gemma2", activations={"analysis_configuration": config})
    )
    for directory in ("metrics", "plot_data", "geometry", "validation", "inference"):
        store.csv(f"{directory}/result.csv", pd.DataFrame({"value": [1.0]}))
    (root / "plots").mkdir()
    (root / "plots/figure.png").write_bytes(b"synthetic-figure")
    (root / "reports").mkdir()
    (root / "reports/run.md").write_text("Synthetic test report")
    store.finish()
    hub = FakeHub()
    monkeypatch.setattr(exp3_hub, "hub_api", lambda: hub)
    result = upload_analysis(config, tmp_path)
    assert result["upload"]["repo_id"] == "researcher/rq3-gemma2"
    uploaded = set(hub.uploads[0]["allow_patterns"])
    assert uploaded == {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    assert hub.uploads[0]["path_in_repo"] == f"analyses/{config['run_id']}"
    (root / "extra.csv").write_text("unexpected")
    with pytest.raises(ValueError, match="uncheckpointed"):
        upload_analysis(config, tmp_path)


def test_metric_plot_data_preserves_missing_layers_and_emotion_curves():
    from affect_geometry.reporting.correctness_geometry import metric_curve_table

    rows = []
    for emotion in ("joy", "neutral"):
        for layer in (0, 2):
            rows.append(
                dict(
                    cohort="all",
                    method="shared",
                    dimension=0,
                    setting="original",
                    reference="projected_samples",
                    metric="within_emotion_scatter",
                    emotion=emotion,
                    layer=layer,
                    value=1.0,
                    status="ok",
                    reason="",
                )
            )
    scores = pd.DataFrame({"layer": [0, 1, 2], "depth": [1 / 3, 2 / 3, 1]})
    result = metric_curve_table(pd.DataFrame(rows), scores)
    assert len(result) == 6
    missing = result.loc[result.layer.eq(1)]
    assert set(missing.emotion) == {"joy", "neutral"}
    assert missing.value.isna().all() and missing.status.eq("unsupported").all()
