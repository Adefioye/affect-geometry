from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from affect_geometry.analysis.representations import (
    CENTERED,
    DEFAULT_REPRESENTATION_VARIANTS,
    NEUTRAL_PROJECTED,
    RAW,
    PoolingRepresentations,
    RepresentationVariant,
    build_pooling_representations,
)
from affect_geometry.analysis.va import align_similarity
from affect_geometry.embeddings.mds import classical_mds


@dataclass(frozen=True)
class GeometryEvaluationConfig:
    neutral_label: str = "neutral"
    neutral_variance_threshold: float = 0.5
    batch_size: int = 2048
    minimum_vector_norm: float = 1e-12
    variants: tuple[RepresentationVariant, ...] = DEFAULT_REPRESENTATION_VARIANTS

    def __post_init__(self) -> None:
        if not 0 < self.neutral_variance_threshold <= 1:
            raise ValueError("neutral_variance_threshold must be in (0, 1]")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.minimum_vector_norm <= 0:
            raise ValueError("minimum_vector_norm must be positive")
        if not self.variants:
            raise ValueError("At least one representation variant is required")
        names = [variant.name for variant in self.variants]
        if len(names) != len(set(names)):
            raise ValueError("Representation variant names must be unique")
        for variant in self.variants:
            if variant.pooling not in {"last_token", "mean"}:
                raise ValueError(f"Unsupported pooling in variant {variant.name!r}")
            if variant.transformation not in {RAW, CENTERED, NEUTRAL_PROJECTED}:
                raise ValueError(f"Unsupported transformation in variant {variant.name!r}")


@dataclass(frozen=True)
class LayerRepresentationResult:
    layer: int
    variant: RepresentationVariant
    labels: tuple[str, ...]
    sample_counts: np.ndarray
    vectors: np.ndarray
    cosine_rdm: np.ndarray
    mds_coordinates: np.ndarray
    mds_eigenvalues: np.ndarray
    aligned_coordinates: np.ndarray
    target_coordinates: np.ndarray
    va_procrustes_r2: float
    neutral_components_fitted: int
    neutral_variance_explained: float

    def metrics_record(self) -> dict[str, int | float | str]:
        removed = (
            self.neutral_components_fitted
            if self.variant.transformation == NEUTRAL_PROJECTED
            else 0
        )
        return {
            "layer": self.layer,
            "variant": self.variant.name,
            "pooling": self.variant.pooling,
            "transformation": self.variant.transformation,
            "distance": "cosine",
            "method": "mds",
            "dimensions": 2,
            "activation_dimensions": self.vectors.shape[1],
            "emotions": len(self.labels),
            "va_procrustes_r2": self.va_procrustes_r2,
            "neutral_components_fitted": self.neutral_components_fitted,
            "neutral_components_removed": removed,
            "neutral_variance_explained": self.neutral_variance_explained,
        }

    def coordinates_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "layer": self.layer,
                "variant": self.variant.name,
                "pooling": self.variant.pooling,
                "transformation": self.variant.transformation,
                "distance": "cosine",
                "method": "mds",
                "dimensions": 2,
                "emotion": self.labels,
                "sample_count": self.sample_counts,
                "coordinate_1": self.mds_coordinates[:, 0],
                "coordinate_2": self.mds_coordinates[:, 1],
                "aligned_valence": self.aligned_coordinates[:, 0],
                "aligned_arousal": self.aligned_coordinates[:, 1],
                "valence": self.target_coordinates[:, 0],
                "arousal": self.target_coordinates[:, 1],
                "va_procrustes_r2": self.va_procrustes_r2,
            }
        )


@dataclass(frozen=True)
class LayerComparison:
    layer: int
    results: tuple[LayerRepresentationResult, ...]
    representations: dict[str, PoolingRepresentations]


def cosine_rdm(
    vectors: np.ndarray,
    *,
    labels: tuple[str, ...] | list[str] | None = None,
    minimum_norm: float = 1e-12,
) -> np.ndarray:
    """Compute a strict cosine-dissimilarity matrix from emotion vectors."""

    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be a matrix, got shape {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("vectors contain non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    invalid = np.flatnonzero(norms <= minimum_norm)
    if len(invalid):
        names = labels or [str(index) for index in range(len(vectors))]
        failed = ", ".join(str(names[index]) for index in invalid)
        raise ValueError(f"Cosine distance is undefined for near-zero vectors: {failed}")
    normalized = vectors / norms[:, None]
    similarities = np.clip(normalized @ normalized.T, -1.0, 1.0)
    distances = 1.0 - similarities
    distances = (distances + distances.T) / 2
    np.fill_diagonal(distances, 0.0)
    return distances


class RepresentationComparator:
    """Evaluate configured pooling/transformation variants using one geometry pipeline."""

    def __init__(
        self,
        samples: pd.DataFrame,
        reference: pd.DataFrame,
        config: GeometryEvaluationConfig | None = None,
    ) -> None:
        self.samples = samples.reset_index(drop=True).copy()
        self.reference = reference.copy()
        self.config = config or GeometryEvaluationConfig()
        if "emotion" not in self.samples:
            raise ValueError("samples must contain an 'emotion' column")
        required_reference = {"emotion", "valence", "arousal"}
        missing = sorted(required_reference - set(self.reference.columns))
        if missing:
            raise ValueError(f"Reference table is missing required columns: {missing}")
        if self.reference["emotion"].duplicated().any():
            raise ValueError("Reference table contains duplicate emotion labels")
        labels = set(self.samples["emotion"].astype(str))
        reference_labels = set(self.reference["emotion"].astype(str))
        missing_reference = sorted(labels - reference_labels)
        if missing_reference:
            raise ValueError(f"Reference table is missing emotions: {missing_reference}")

    def analyze_layer(
        self,
        layer: int,
        last_token_activations: np.ndarray,
        mean_activations: np.ndarray,
    ) -> LayerComparison:
        expected_rows = len(self.samples)
        for pooling, values in (
            ("last_token", last_token_activations),
            ("mean", mean_activations),
        ):
            if values.ndim != 2 or len(values) != expected_rows:
                raise ValueError(
                    f"{pooling} activations must have {expected_rows} rows, got {values.shape}"
                )
        if last_token_activations.shape[1] != mean_activations.shape[1]:
            raise ValueError(
                "Pooling hidden-size mismatch: "
                f"{last_token_activations.shape[1]} != {mean_activations.shape[1]}"
            )

        representations = {
            "last_token": build_pooling_representations(
                "last_token",
                last_token_activations,
                self.samples,
                neutral_label=self.config.neutral_label,
                neutral_variance_threshold=self.config.neutral_variance_threshold,
                batch_size=self.config.batch_size,
            ),
            "mean": build_pooling_representations(
                "mean",
                mean_activations,
                self.samples,
                neutral_label=self.config.neutral_label,
                neutral_variance_threshold=self.config.neutral_variance_threshold,
                batch_size=self.config.batch_size,
            ),
        }

        lookup = self.reference.set_index("emotion")
        results = []
        for variant in self.config.variants:
            pooling_result = representations[variant.pooling]
            labels = pooling_result.centroids.labels
            vectors = pooling_result.values_for(variant.transformation)
            distances = cosine_rdm(
                vectors,
                labels=labels,
                minimum_norm=self.config.minimum_vector_norm,
            )
            mds = classical_mds(distances, 2)
            target = lookup.loc[list(labels), ["valence", "arousal"]].to_numpy(float)
            aligned, r_squared = align_similarity(mds.coordinates, target)
            results.append(
                LayerRepresentationResult(
                    layer=layer,
                    variant=variant,
                    labels=labels,
                    sample_counts=pooling_result.centroids.sample_counts,
                    vectors=vectors,
                    cosine_rdm=distances,
                    mds_coordinates=mds.coordinates,
                    mds_eigenvalues=mds.eigenvalues,
                    aligned_coordinates=aligned,
                    target_coordinates=target,
                    va_procrustes_r2=r_squared,
                    neutral_components_fitted=pooling_result.neutral_subspace.n_components,
                    neutral_variance_explained=(
                        pooling_result.neutral_subspace.cumulative_explained_variance
                    ),
                )
            )
        return LayerComparison(
            layer=layer,
            results=tuple(results),
            representations=representations,
        )


def layerwise_results_frame(comparisons: list[LayerComparison]) -> pd.DataFrame:
    records = [result.metrics_record() for item in comparisons for result in item.results]
    return pd.DataFrame.from_records(records)


def coordinates_frame(comparisons: list[LayerComparison]) -> pd.DataFrame:
    frames = [result.coordinates_frame() for item in comparisons for result in item.results]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_layerwise_scores(layerwise: pd.DataFrame) -> pd.DataFrame:
    required = {"layer", "variant", "pooling", "transformation", "va_procrustes_r2"}
    missing = sorted(required - set(layerwise.columns))
    if missing:
        raise ValueError(f"Layerwise table is missing required columns: {missing}")
    if layerwise.empty:
        return pd.DataFrame(
            columns=[
                "rank",
                "variant",
                "pooling",
                "transformation",
                "layers",
                "mean_va_procrustes_r2",
                "max_va_procrustes_r2",
                "max_layer",
            ]
        )

    rows = []
    grouping = ["variant", "pooling", "transformation"]
    for key, frame in layerwise.groupby(grouping, sort=False):
        ordered = frame.sort_values("layer")
        maximum = ordered.loc[ordered["va_procrustes_r2"].idxmax()]
        rows.append(
            {
                "variant": key[0],
                "pooling": key[1],
                "transformation": key[2],
                "layers": int(ordered["layer"].nunique()),
                "mean_va_procrustes_r2": float(ordered["va_procrustes_r2"].mean()),
                "max_va_procrustes_r2": float(maximum["va_procrustes_r2"]),
                "max_layer": int(maximum["layer"]),
            }
        )
    result = pd.DataFrame.from_records(rows)
    result = result.sort_values("mean_va_procrustes_r2", ascending=False).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1, dtype=np.int64))
    return result
