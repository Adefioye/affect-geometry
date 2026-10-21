from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from affect_geometry.analysis.representation_comparison import (
    LayerComparison,
    summarize_layerwise_scores,
)


@dataclass
class RepresentationComparisonArtifactStore:
    """Persist reproducible per-layer artifacts and consolidated plot-ready tables."""

    root: Path
    _metrics: list[dict[str, int | float | str]] = field(default_factory=list)
    _coordinate_frames: list[pd.DataFrame] = field(default_factory=list)

    @classmethod
    def create(cls, root: str | Path) -> RepresentationComparisonArtifactStore:
        store = cls(Path(root))
        for directory in (
            "analyses",
            "checkpoints/neutral_pca",
            "embeddings",
            "plot_data",
            "plots",
            "rdms",
            "representations",
        ):
            (store.root / directory).mkdir(parents=True, exist_ok=True)
        return store

    def write_layer(self, comparison: LayerComparison) -> None:
        layer = comparison.layer
        for pooling, representations in comparison.representations.items():
            checkpoint = (
                self.root / "checkpoints" / "neutral_pca" / pooling / f"layer_{layer:03d}.npz"
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            subspace = representations.neutral_subspace
            np.savez_compressed(
                checkpoint,
                neutral_mean=subspace.mean,
                components=subspace.components,
                explained_variance=subspace.explained_variance,
                explained_variance_ratio=subspace.explained_variance_ratio,
                variance_threshold=subspace.variance_threshold,
                neutral_samples=subspace.neutral_samples,
                emotion_grand_mean=representations.grand_mean,
            )

        for result in comparison.results:
            variant = result.variant.name
            labels = np.asarray(result.labels, dtype=str)
            representation_path = self.root / "representations" / variant / f"layer_{layer:03d}.npz"
            representation_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                representation_path,
                labels=labels,
                sample_counts=result.sample_counts,
                vectors=result.vectors,
            )

            rdm_path = self.root / "rdms" / variant / f"layer_{layer:03d}_cosine.npz"
            rdm_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                rdm_path,
                labels=labels,
                distance=result.cosine_rdm,
                metric="cosine",
            )

            embedding_path = self.root / "embeddings" / variant / f"layer_{layer:03d}_mds_2d.csv"
            embedding_path.parent.mkdir(parents=True, exist_ok=True)
            result.coordinates_frame().to_csv(embedding_path, index=False)
            eigenvalue_path = (
                self.root / "embeddings" / variant / f"layer_{layer:03d}_eigenvalues.npz"
            )
            np.savez_compressed(eigenvalue_path, eigenvalues=result.mds_eigenvalues)

            self._metrics.append(result.metrics_record())
            self._coordinate_frames.append(result.coordinates_frame())

    def write_provenance(self, value: dict[str, Any]) -> Path:
        path = self.root / "analyses" / "provenance.json"
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        return path

    def finalize(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        layerwise = pd.DataFrame.from_records(self._metrics)
        if layerwise.empty:
            raise ValueError("No layer results were written")
        layerwise = layerwise.sort_values(["variant", "layer"]).reset_index(drop=True)
        coordinates = pd.concat(self._coordinate_frames, ignore_index=True)
        summary = summarize_layerwise_scores(layerwise)

        layerwise.to_csv(self.root / "analyses" / "layerwise_procrustes.csv", index=False)
        summary.to_csv(self.root / "analyses" / "representation_summary.csv", index=False)
        coordinates.to_csv(self.root / "analyses" / "aligned_coordinates.csv", index=False)

        # Essential plots consume only these self-contained CSVs.
        layerwise.to_csv(
            self.root / "plot_data" / "representation_comparison_layerwise.csv",
            index=False,
        )
        summary.to_csv(
            self.root / "plot_data" / "representation_comparison_summary.csv",
            index=False,
        )
        coordinates.to_csv(
            self.root / "plot_data" / "representation_comparison_coordinates.csv",
            index=False,
        )
        return layerwise, summary, coordinates
