from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from affect_geometry.config import (
    ActivationConfig,
    AnalysisConfig,
    EmbeddingConfig,
    ExperimentConfig,
    ModelConfig,
    RDMConfig,
)
from affect_geometry.pipeline import analyze, compute_rdms, embed


@pytest.mark.parametrize("pca_components", [4, None])
def test_synthetic_pipeline_writes_reusable_tables(
    tmp_path: Path, pca_components: int | None
) -> None:
    labels = ["joy", "anger", "sadness", "neutral"]
    samples = []
    rng = np.random.default_rng(11)
    activation_rows = []
    for label_index, label in enumerate(labels):
        for sample_index in range(12):
            samples.append(
                {
                    "sample_id": f"{label}-{sample_index}",
                    "source_id": sample_index,
                    "source_row": sample_index,
                    "split": "train",
                    "label_id": label_index,
                    "emotion": label,
                    "text": "synthetic",
                }
            )
            activation_rows.append(rng.normal(label_index, 0.5, 8))
    root = tmp_path / "outputs" / "synthetic" / "smoke"
    (root / "dataset").mkdir(parents=True)
    (root / "activations" / "last_token").mkdir(parents=True)
    pd.DataFrame(samples).to_csv(root / "dataset" / "samples.csv", index=False)
    np.save(root / "activations" / "last_token" / "layer_000.npy", activation_rows)
    reference_path = tmp_path / "va.csv"
    pd.DataFrame(
        {
            "emotion": labels,
            "valence": [0.9, -0.8, -0.7, 0.0],
            "arousal": [0.6, 0.8, -0.5, -0.2],
        }
    ).to_csv(reference_path, index=False)
    config = ExperimentConfig(
        run_id="smoke",
        output_root=str(tmp_path / "outputs"),
        model=ModelConfig(name="synthetic"),
        activations=ActivationConfig(poolings=["last_token"], storage_dtype="float32"),
        rdm=RDMConfig(
            pca_components=pca_components,
            max_samples_per_emotion=8,
            crossnobis_folds=3,
            sliced_wasserstein_projections=16,
        ),
        embedding=EmbeddingConfig(
            dimensions=[1, 2, 3],
            intrinsic_max_dimensions=3,
            isomap_neighbors=2,
            isomap_neighbor_grid=[2],
            quality_neighbors=1,
        ),
        analysis=AnalysisConfig(
            va_reference=str(reference_path), va_permutations=5, knn_neighbors=2
        ),
    )
    compute_rdms(config)
    assert (root / "checkpoints/last_token/layer_000_pca.npz").exists() == (
        pca_components is not None
    )
    embed(config)
    analyze(config)
    assert (root / "rdms" / "last_token" / "layer_000" / "crossnobis.npz").exists()
    assert (root / "plot_data" / "coordinates.csv").exists()
    assert (root / "plot_data" / "layerwise_metrics.csv").exists()
    assert not pd.read_csv(root / "analyses" / "va_recovery.csv").empty
