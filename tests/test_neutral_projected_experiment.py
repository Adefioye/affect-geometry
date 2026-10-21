import hashlib
import io
import json
import urllib.error
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

from affect_geometry.analysis.neutral_projected import (
    NEGATIVE,
    POSITIVE,
    neutral_relative_coordinates,
)
from affect_geometry.analysis.representation_comparison import cosine_rdm
from affect_geometry.analysis.representations import build_pooling_representations
from affect_geometry.config import (
    ActivationConfig,
    AnalysisConfig,
    EmbeddingConfig,
    ExperimentConfig,
    ModelConfig,
    RDMConfig,
    RepresentationConfig,
)
from affect_geometry.data.nrc_vad import prepare_nrc_vad, read_nrc_vad, read_vad_reference
from affect_geometry.distances.rdm import compute_rdm
from affect_geometry.experiments.neutral_projected import (
    compute_projected_rdms,
    run_projected_experiment,
)


def test_reference_reads_official_zip_layout_and_preserves_scores(tmp_path):
    payload = io.BytesIO()
    tsv = "term\tvalence\tarousal\tdominance\njoy\t0.96\t0.648\t0.7\nneutral\t-0.062\t-0.632\t0.1\n"
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("NRC-VAD-Lexicon-v2.1/README.txt", "Read me")
        archive.writestr("NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt", tsv)
        archive.writestr("NRC-VAD-Lexicon-v2.1/OneFilePerDimension/valence.txt", "term\tvalence\n")
    lexicon = tmp_path / "official.zip"
    lexicon.write_bytes(payload.getvalue())
    path = prepare_nrc_vad(["neutral", "joy"], tmp_path / "outputs", local_lexicon=lexicon)
    result = pd.read_csv(path)
    assert result.emotion.tolist() == ["neutral", "joy"]
    assert result.dominance.tolist() == [0.1, 0.7]
    assert result.valence.tolist() == [-0.062, 0.96]
    provenance = json.loads((path.parent / "nrc_vad_provenance.json").read_text())
    assert provenance["sha256"] == hashlib.sha256(lexicon.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="missing exact label"):
        prepare_nrc_vad(["missing"], tmp_path / "outputs", local_lexicon=lexicon)


@pytest.mark.parametrize("row", ["joy\t2\t0\t0", "joy\t0\tinf\t0", "joy\t0\t\t0"])
def test_reference_rejects_invalid_scores(row):
    with pytest.raises(ValueError):
        read_nrc_vad(f"term\tvalence\tarousal\tdominance\n{row}\n".encode())


def test_download_failure_explains_local_fallback(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("url", 406, "Not acceptable", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(RuntimeError, match="--nrc-vad-lexicon"):
        prepare_nrc_vad(["joy"], tmp_path / "outputs")


def test_repository_vad_reference_works_offline_and_matches_va(tmp_path, monkeypatch):
    def forbid_network(*args, **kwargs):
        pytest.fail("A prepared local VAD reference must not access the network")

    monkeypatch.setattr("urllib.request.urlopen", forbid_network)
    reference = Path(__file__).resolve().parents[1] / "data/reference/goemotions_nrc_vad.csv"
    va = pd.read_csv(reference.with_name("goemotions_nrc_va.csv"))
    vad = read_vad_reference(reference.read_bytes())
    pd.testing.assert_frame_equal(vad.drop(columns="dominance"), va)
    assert len(vad) == 28
    provenance = json.loads(reference.with_suffix(".provenance.json").read_text())
    assert provenance["reference_sha256"] == hashlib.sha256(reference.read_bytes()).hexdigest()
    output = prepare_nrc_vad(vad.emotion.tolist(), tmp_path / "outputs", local_reference=reference)
    pd.testing.assert_frame_equal(pd.read_csv(output), vad)
    assert not list(output.parent.glob("*.zip"))
    assert not list(output.parent.glob("*.tsv"))
    snapshot = json.loads((output.parent / "nrc_vad_provenance.json").read_text())
    assert snapshot["upstream_provenance"] == provenance
    with pytest.raises(ValueError, match="missing emotions"):
        prepare_nrc_vad(["absent"], tmp_path / "outputs", local_reference=reference)


def test_prepared_reference_requires_dominance_and_unique_labels():
    with pytest.raises(ValueError, match="dominance"):
        read_vad_reference(b"emotion,source_term,valence,arousal\njoy,joy,0.9,0.5\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_vad_reference(
            b"emotion,source_term,valence,arousal,dominance\n"
            b"joy,joy,0.9,0.5,0.4\njoy,joy,0.9,0.5,0.4\n"
        )


def test_neutral_partition_preserves_distances_and_omits_reference():
    original = np.array([[3, 4], [-1, 4], [-1, -2], [3, -2], [1, 1]], dtype=float)
    frame = pd.DataFrame(
        {
            "pooling": "last_token",
            "layer": 0,
            "distance": "cosine",
            "method": "mds",
            "dimensions": 2,
            "emotion": ["joy", "anger", "sadness", "relief", "neutral"],
            "coordinate_1": original[:, 0],
            "coordinate_2": original[:, 1],
            "valence": [1, -1, -1, 1, 0],
            "arousal": 0,
            "dominance": 0,
        }
    )
    result = neutral_relative_coordinates(frame)
    assert result.quadrant.tolist() == ["I", "II", "III", "IV", "neutral"]
    transformed = result[["coordinate_1", "coordinate_2"]].to_numpy()
    assert np.allclose(transformed[-1], 0)
    assert np.allclose(pdist(original), pdist(transformed))
    assert not {"valence", "arousal", "dominance"}.intersection(result.columns)
    with pytest.raises(ValueError, match="exactly one neutral"):
        neutral_relative_coordinates(frame.iloc[:-1])


def _source(tmp_path):
    root = tmp_path / "source"
    (root / "dataset").mkdir(parents=True)
    (root / "activations" / "last_token").mkdir(parents=True)
    labels = sorted(
        POSITIVE | NEGATIVE | {"confusion", "curiosity", "realization", "surprise", "neutral"}
    )
    rng = np.random.default_rng(3)
    records, activations = [], []
    for index, label in enumerate(labels):
        center = rng.normal(size=8)
        for row in range(8):
            records.append(
                {
                    "sample_id": f"{label}-{row}",
                    "emotion": label,
                    "label_id": index,
                    "split": "train",
                    "text": label,
                }
            )
            activation = center + rng.normal(scale=0.2, size=8)
            if label == "neutral":
                activation[0] += 5 * (row - 3.5)
            activations.append(activation)
    samples = pd.DataFrame(records)
    samples.to_csv(root / "dataset" / "samples.csv", index=False)
    values = np.asarray(activations, dtype=np.float32)
    np.save(root / "activations" / "last_token" / "layer_000.npy", values)
    metadata = {"model": "synthetic", "n_layers": 1, "n_samples": len(samples), "hidden_size": 8}
    (root / "activations" / "metadata.json").write_text(json.dumps(metadata))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    "run_id": "source",
                    "activations": {"poolings": ["last_token"]},
                }
            }
        )
    )
    lexicon = tmp_path / "synthetic-reference.tsv"
    pd.DataFrame(
        {
            "term": labels,
            **{
                axis: rng.uniform(-1, 1, size=len(labels))
                for axis in ["valence", "arousal", "dominance"]
            },
        }
    ).to_csv(lexicon, sep="\t", index=False)
    return root, lexicon, samples, values


@pytest.mark.parametrize("pca_components", [None, 2])
def test_all_distances_use_full_projected_space(tmp_path, monkeypatch, pca_components):
    source, _, samples, values = _source(tmp_path)

    def forbid_pca(*args, **kwargs):
        pytest.fail("No PCA may be fitted after neutral projection")

    monkeypatch.setattr("sklearn.decomposition.PCA.fit", forbid_pca)
    config = ExperimentConfig(
        run_id="full-space",
        output_root=str(tmp_path / "outputs"),
        model=ModelConfig(name="synthetic"),
        activations=ActivationConfig(poolings=["last_token"]),
        representation=RepresentationConfig(transformation="neutral_projected", batch_size=17),
        rdm=RDMConfig(
            pca_components=pca_components,
            max_samples_per_emotion=6,
            crossnobis_folds=2,
            sliced_wasserstein_projections=8,
        ),
    )
    obsolete = config.run_dir / "checkpoints/neutral_projection/layer_000_auxiliary_pca.npz"
    obsolete.parent.mkdir(parents=True)
    np.savez(obsolete, components=np.eye(2, 8))
    compute_projected_rdms(config, source)
    assert not obsolete.exists()
    representation = build_pooling_representations("last_token", values, samples)
    full_projected = representation.neutral_subspace.project_out(
        values.astype(float) - representation.grand_mean
    ).astype(np.float32)
    saved_activations = np.load(config.run_dir / "activations/last_token/layer_000.npy")
    assert np.allclose(saved_activations, full_projected, atol=1e-7)
    metadata = json.loads((config.run_dir / "activations/metadata.json").read_text())
    assert metadata["variant"] == "LT-NeutralProjected"
    assert metadata["complete"] is True
    assert metadata["n_samples"] == len(samples)
    labels = list(representation.centroids.labels)
    selection = pd.read_csv(config.run_dir / "rdms/distribution_sample.csv")
    for name in ["crossnobis", "rbf_mmd", "energy", "sliced_wasserstein"]:
        if name == "crossnobis":
            rows = [np.flatnonzero(samples.emotion.to_numpy() == label) for label in labels]
        else:
            rows = [
                selection.loc[selection.emotion == label, "activation_row"].to_numpy()
                for label in labels
            ]
        expected = compute_rdm(
            name,
            labels,
            [full_projected[group] for group in rows],
            seed=config.seed,
            folds=config.rdm.crossnobis_folds,
            bandwidth_multipliers=config.rdm.mmd_bandwidth_multipliers,
            projections=config.rdm.sliced_wasserstein_projections,
        )
        saved = np.load(config.run_dir / f"rdms/last_token/layer_000/{name}.npz")
        assert np.allclose(saved["raw"], expected.raw, rtol=1e-5, atol=1e-7), name
        assert np.allclose(saved["distance"], expected.distance, rtol=1e-5, atol=1e-7), name
    for name in config.rdm.distances:
        metadata = pd.read_csv(config.run_dir / f"rdms/last_token/layer_000/{name}.csv")
        assert metadata.activation_dimensions.eq(values.shape[1]).all()
        assert metadata.pca_components.eq(0).all()
        assert not metadata.post_projection_pca.any()
    provenance = json.loads((config.run_dir / "representation.json").read_text())
    assert provenance["post_projection_pca"] is False


def test_fresh_run_preserves_raw_source_and_can_recompute_projected_store(tmp_path):
    import shutil

    source, _, samples, values = _source(tmp_path)
    config = ExperimentConfig(
        run_id="fresh",
        output_root=str(tmp_path / "outputs"),
        model=ModelConfig(name="synthetic"),
        activations=ActivationConfig(poolings=["last_token"]),
        representation=RepresentationConfig(transformation="neutral_projected"),
        rdm=RDMConfig(distances=["cosine", "euclidean"]),
    )
    shutil.copytree(source, config.run_dir)
    compute_projected_rdms(config, config.run_dir)
    raw = config.run_dir / "activation_source/activations/last_token/layer_000.npy"
    assert np.array_equal(np.load(raw), values)
    projected_path = config.run_dir / "activations/last_token/layer_000.npy"
    first = np.load(projected_path).copy()
    expected = build_pooling_representations("last_token", values, samples)
    assert np.allclose(
        first,
        expected.neutral_subspace.project_out(values.astype(float) - expected.grand_mean),
        atol=1e-6,
    )
    compute_projected_rdms(config, config.run_dir)
    assert np.array_equal(np.load(projected_path), first)
    assert np.array_equal(np.load(raw), values)


def test_projected_experiment_all_labels_metrics_and_plots(tmp_path, monkeypatch):
    source, lexicon, samples, values = _source(tmp_path)
    reference = lexicon.with_suffix(".csv")
    pd.read_csv(lexicon, sep="\t").rename(columns={"term": "emotion"}).assign(
        source_term=lambda frame: frame.emotion
    ).to_csv(reference, index=False)
    lexicon.unlink()

    def forbid_network(*args, **kwargs):
        pytest.fail("The default reference stage must run offline with a prepared CSV")

    monkeypatch.setattr("urllib.request.urlopen", forbid_network)
    original_manifest = (source / "manifest.json").read_bytes()
    original_array = (source / "activations" / "last_token" / "layer_000.npy").read_bytes()
    config = ExperimentConfig(
        run_id="projected",
        output_root=str(tmp_path / "outputs"),
        log_root=str(tmp_path / "logs"),
        model=ModelConfig(name="synthetic"),
        activations=ActivationConfig(poolings=["last_token"]),
        representation=RepresentationConfig(transformation="neutral_projected", batch_size=17),
        rdm=RDMConfig(
            pca_components=4,
            max_samples_per_emotion=6,
            crossnobis_folds=2,
            sliced_wasserstein_projections=8,
        ),
        embedding=EmbeddingConfig(
            intrinsic_max_dimensions=4,
            isomap_neighbors=3,
            isomap_neighbor_grid=[2, 3],
            quality_neighbors=2,
        ),
        analysis=AnalysisConfig(va_permutations=3, knn_neighbors=3, vad_reference=str(reference)),
    )
    config = replace(
        config,
        analysis=replace(
            config.analysis,
            va_reference=str(config.run_dir / "references" / "goemotions_nrc_vad.csv"),
        ),
    )
    run_projected_experiment(config, source_run=source, dry_run=True)
    assert not config.run_dir.exists()
    gallery = run_projected_experiment(config, source_run=source)
    assert gallery.exists()
    assert (source / "manifest.json").read_bytes() == original_manifest
    assert (source / "activations" / "last_token" / "layer_000.npy").read_bytes() == original_array
    root = config.run_dir
    expected = build_pooling_representations("last_token", values, samples).neutral_projected
    cosine = np.load(root / "rdms" / "last_token" / "layer_000" / "cosine.npz")["distance"]
    euclidean = np.load(root / "rdms" / "last_token" / "layer_000" / "euclidean.npz")["distance"]
    assert np.allclose(cosine, cosine_rdm(expected))
    assert np.allclose(euclidean, squareform(pdist(expected)))
    exploration = pd.read_csv(root / "plot_data" / "exploration_coordinates.csv")
    assert not {"valence", "arousal", "dominance"}.intersection(exploration.columns)
    assert not any(column.startswith("aligned_") for column in exploration)
    assert set(exploration.dimensions) == {1, 2, 3}
    assert set(exploration.method) == {"mds", "isomap"}
    assert exploration.groupby(["distance", "method", "dimensions"]).emotion.nunique().eq(28).all()
    neutral = exploration.loc[exploration.emotion == "neutral"]
    for dimensions in [1, 2, 3]:
        columns = [f"coordinate_{axis}" for axis in range(1, dimensions + 1)]
        assert np.allclose(neutral.loc[neutral.dimensions == dimensions, columns], 0)
    assert len(list((root / "plots" / "exploration").rglob("*.png"))) == 6 * 2 * 3
    for name in config.rdm.distances:
        for method in ["mds", "isomap"]:
            for dimensions in [1, 2, 3]:
                assert (
                    root / "plots/exploration/layer_000" / f"{name}_{method}_{dimensions}d.png"
                ).is_file()
    recovery = pd.read_csv(root / "analyses" / "vad_recovery.csv")
    assert len(recovery) == 6 * 2
    assert recovery.vad_permutation_p.between(0.25, 1).all()
    assert (root / "plots" / "layerwise" / "vad_recovery.png").is_file()
    assert (config.log_dir / "run.log").is_file()
    # Re-render after removing intermediate embedding files: plots depend on CSV exports only.
    for path in (root / "embeddings").rglob("coordinates.csv"):
        path.unlink()
    run_projected_experiment(config, start_at="plot")
