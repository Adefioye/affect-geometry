import json
from pathlib import Path

import numpy as np
import pandas as pd

from affect_geometry.analysis.representation_comparison import (
    GeometryEvaluationConfig,
    RepresentationComparator,
    layerwise_results_frame,
    summarize_layerwise_scores,
)
from affect_geometry.analysis.representations import (
    DEFAULT_REPRESENTATION_VARIANTS,
    build_pooling_representations,
)
from affect_geometry.experiments.representation_comparison import (
    RepresentationComparisonConfig,
    run_representation_comparison,
)
from affect_geometry.persistence.representation_comparison import (
    RepresentationComparisonArtifactStore,
)
from affect_geometry.reporting.representation_comparison import (
    create_representation_comparison_plots,
)


def _synthetic_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    labels = ["anger", "joy", "neutral", "sadness"]
    centers = {
        "anger": np.array([0.8, 0.1, 1.2, -0.3]),
        "joy": np.array([2.1, 1.3, 0.4, 0.2]),
        "neutral": np.array([0.0, 0.2, 0.1, 0.0]),
        "sadness": np.array([-1.7, -0.9, 0.2, 0.5]),
    }
    rows, activations = [], []
    neutral_axis = np.array([-2.0, -1.0, 1.0, 2.0, -1.5, 1.5])
    for label_index, label in enumerate(labels):
        for sample_index in range(6):
            rows.append(
                {
                    "sample_id": f"{label}-{sample_index}",
                    "split": "train",
                    "label_id": label_index,
                    "emotion": label,
                }
            )
            variation = np.zeros(4)
            if label == "neutral":
                variation[0] = neutral_axis[sample_index]
            else:
                variation[1] = (sample_index - 2.5) * 0.02
            activations.append(centers[label] + variation)
    last_token = np.asarray(activations, dtype=np.float64)
    mean = last_token @ np.array(
        [
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.1, 0.1, 0.0],
            [0.0, 0.0, 0.8, 0.2],
            [0.1, 0.0, 0.0, 1.0],
        ]
    )
    reference = pd.DataFrame(
        {
            "emotion": labels,
            "valence": [-0.7, 0.9, 0.0, -0.8],
            "arousal": [0.8, 0.6, -0.5, -0.4],
        }
    )
    return pd.DataFrame(rows), last_token, mean, reference


def test_neutral_projection_removes_leading_neutral_direction() -> None:
    samples, last_token, _, _ = _synthetic_data()
    result = build_pooling_representations(
        "last_token",
        last_token,
        samples,
        neutral_variance_threshold=0.5,
        batch_size=3,
    )

    assert result.neutral_subspace.n_components == 1
    assert result.neutral_subspace.cumulative_explained_variance >= 0.5
    component = result.neutral_subspace.components[0]
    projections = result.neutral_projected @ component
    assert np.allclose(projections, 0.0, atol=1e-10)
    expected_grand_mean = result.centroids.values[
        [label != "neutral" for label in result.centroids.labels]
    ].mean(axis=0)
    assert np.allclose(result.grand_mean, expected_grand_mean)


def test_representation_comparison_analysis_and_artifacts(tmp_path: Path) -> None:
    samples, last_token, mean, reference = _synthetic_data()
    comparator = RepresentationComparator(
        samples,
        reference,
        GeometryEvaluationConfig(batch_size=3, neutral_variance_threshold=0.5),
    )
    comparison = comparator.analyze_layer(0, last_token, mean)

    assert [result.variant for result in comparison.results] == list(
        DEFAULT_REPRESENTATION_VARIANTS
    )
    assert len(comparison.results) == 6
    assert all(result.cosine_rdm.shape == (4, 4) for result in comparison.results)
    assert all(result.mds_coordinates.shape == (4, 2) for result in comparison.results)
    assert all(np.isfinite(result.va_procrustes_r2) for result in comparison.results)

    layerwise = layerwise_results_frame([comparison])
    summary = summarize_layerwise_scores(layerwise)
    assert len(layerwise) == 6
    assert len(summary) == 6
    assert set(summary["max_layer"]) == {0}

    store = RepresentationComparisonArtifactStore.create(tmp_path / "comparison")
    store.write_layer(comparison)
    written_layerwise, written_summary, coordinates = store.finalize()
    paths = create_representation_comparison_plots(
        tmp_path / "comparison" / "plot_data",
        tmp_path / "comparison" / "plots",
    )
    assert len(written_layerwise) == 6
    assert len(written_summary) == 6
    assert written_summary["rank"].tolist() == [1, 2, 3, 4, 5, 6]
    assert len(coordinates) == 24
    assert all(path.exists() for path in paths)
    assert (
        tmp_path / "comparison" / "checkpoints" / "neutral_pca" / "last_token" / "layer_000.npz"
    ).exists()


def _write_activation_run(
    root: Path,
    pooling: str,
    samples: pd.DataFrame,
    activations: np.ndarray,
) -> None:
    (root / "dataset").mkdir(parents=True)
    (root / "activations" / pooling).mkdir(parents=True)
    samples.to_csv(root / "dataset" / "samples.csv", index=False)
    np.save(root / "activations" / pooling / "layer_000.npy", activations)
    (root / "activations" / "metadata.json").write_text(
        json.dumps(
            {
                "n_samples": len(samples),
                "n_layers": 1,
                "hidden_size": activations.shape[1],
                "model": "synthetic",
                "model_revision": "fixed",
            }
        )
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    "run_id": root.name,
                    "model": {"name": "synthetic"},
                    "activations": {"poolings": [pooling]},
                }
            }
        )
    )


def test_experiment_orchestrator_aligns_sources_and_completes(tmp_path: Path) -> None:
    samples, last_token, mean, reference = _synthetic_data()
    last_root, mean_root = tmp_path / "last-run", tmp_path / "mean-run"
    _write_activation_run(last_root, "last_token", samples, last_token)
    reverse = np.arange(len(samples))[::-1]
    _write_activation_run(
        mean_root,
        "mean",
        samples.iloc[reverse].reset_index(drop=True),
        mean[reverse],
    )
    reference_path = tmp_path / "reference.csv"
    reference.to_csv(reference_path, index=False)

    result = run_representation_comparison(
        RepresentationComparisonConfig(
            last_token_run=last_root,
            mean_run=mean_root,
            va_reference=reference_path,
            output_root=tmp_path / "representation-comparison-output",
            batch_size=3,
        )
    )

    assert len(result.layerwise) == 6
    assert len(result.summary) == 6
    assert len(result.coordinates) == 24
    assert all(path.exists() for path in result.plot_paths)
    assert (result.output_root / "analyses" / "provenance.json").exists()
