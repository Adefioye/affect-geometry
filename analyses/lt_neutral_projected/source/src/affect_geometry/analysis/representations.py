from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RAW = "raw"
CENTERED = "centered"
NEUTRAL_PROJECTED = "neutral_projected"


@dataclass(frozen=True)
class RepresentationVariant:
    name: str
    pooling: str
    transformation: str


DEFAULT_REPRESENTATION_VARIANTS = (
    RepresentationVariant("LT-Raw", "last_token", RAW),
    RepresentationVariant("LT-Centered", "last_token", CENTERED),
    RepresentationVariant("LT-NeutralProjected", "last_token", NEUTRAL_PROJECTED),
    RepresentationVariant("Mean-Raw", "mean", RAW),
    RepresentationVariant("Mean-Centered", "mean", CENTERED),
    RepresentationVariant("Mean-NeutralProjected", "mean", NEUTRAL_PROJECTED),
)


@dataclass(frozen=True)
class EmotionCentroids:
    labels: tuple[str, ...]
    values: np.ndarray
    sample_counts: np.ndarray


@dataclass(frozen=True)
class NeutralSubspace:
    """Principal directions learned from within-neutral activation variation."""

    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    variance_threshold: float
    neutral_samples: int

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    @property
    def cumulative_explained_variance(self) -> float:
        return float(self.explained_variance_ratio.sum())

    def project_out(self, values: np.ndarray) -> np.ndarray:
        """Remove this linear subspace without applying the neutral-set mean."""

        values = np.asarray(values, dtype=np.float64)
        if values.shape[-1] != self.components.shape[1]:
            raise ValueError(
                f"Expected vectors with width {self.components.shape[1]}, got {values.shape[-1]}"
            )
        return values - (values @ self.components.T) @ self.components


@dataclass(frozen=True)
class PoolingRepresentations:
    pooling: str
    centroids: EmotionCentroids
    grand_mean: np.ndarray
    neutral_subspace: NeutralSubspace
    raw: np.ndarray
    centered: np.ndarray
    neutral_projected: np.ndarray

    def values_for(self, transformation: str) -> np.ndarray:
        if transformation == RAW:
            return self.raw
        if transformation == CENTERED:
            return self.centered
        if transformation == NEUTRAL_PROJECTED:
            return self.neutral_projected
        raise ValueError(f"Unknown transformation: {transformation!r}")


def _validate_activations(activations: np.ndarray, emotions: np.ndarray) -> None:
    if activations.ndim != 2:
        raise ValueError(f"Activations must be a matrix, got shape {activations.shape}")
    if len(activations) != len(emotions):
        raise ValueError(
            f"Activation/sample length mismatch: {len(activations)} != {len(emotions)}"
        )
    if activations.shape[1] < 1:
        raise ValueError("Activations must contain at least one feature")


def _row_mean(
    activations: np.ndarray,
    rows: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    total = np.zeros(activations.shape[1], dtype=np.float64)
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        total += np.asarray(activations[batch_rows], dtype=np.float64).sum(axis=0)
    return total / len(rows)


def compute_emotion_centroids(
    activations: np.ndarray,
    emotions: pd.Series | np.ndarray,
    *,
    labels: list[str] | tuple[str, ...] | None = None,
    batch_size: int = 2048,
) -> EmotionCentroids:
    """Average sample activations into equally represented emotion nodes."""

    emotion_values = np.asarray(emotions, dtype=str)
    _validate_activations(activations, emotion_values)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    resolved_labels = tuple(labels or sorted(np.unique(emotion_values).tolist()))
    if len(set(resolved_labels)) != len(resolved_labels):
        raise ValueError("Emotion labels must be unique")

    centers, counts = [], []
    for label in resolved_labels:
        rows = np.flatnonzero(emotion_values == label)
        if not len(rows):
            raise ValueError(f"No samples found for emotion {label!r}")
        centers.append(_row_mean(activations, rows, batch_size=batch_size))
        counts.append(len(rows))
    return EmotionCentroids(
        labels=resolved_labels,
        values=np.stack(centers),
        sample_counts=np.asarray(counts, dtype=np.int64),
    )


def emotion_grand_mean(
    centroids: EmotionCentroids,
    *,
    neutral_label: str = "neutral",
) -> np.ndarray:
    """Equal-weight mean of non-neutral emotion centroids."""

    keep = np.asarray([label != neutral_label for label in centroids.labels])
    if keep.all():
        raise ValueError(f"Neutral label {neutral_label!r} is absent from the emotion centroids")
    if not keep.any():
        raise ValueError("At least one non-neutral emotion is required")
    return centroids.values[keep].mean(axis=0)


def fit_neutral_subspace(
    activations: np.ndarray,
    emotions: pd.Series | np.ndarray,
    *,
    neutral_label: str = "neutral",
    variance_threshold: float = 0.5,
    batch_size: int = 2048,
) -> NeutralSubspace:
    """Fit exact covariance PCA to GoEmotions neutral-labeled samples.

    The smallest leading set of components whose cumulative explained variance reaches
    ``variance_threshold`` is retained. Covariance is accumulated in batches so a complete
    neutral activation matrix never has to be materialized in memory.
    """

    emotion_values = np.asarray(emotions, dtype=str)
    _validate_activations(activations, emotion_values)
    if not 0 < variance_threshold <= 1:
        raise ValueError("variance_threshold must be in (0, 1]")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    neutral_rows = np.flatnonzero(emotion_values == neutral_label)
    if len(neutral_rows) < 2:
        raise ValueError(f"Neutral-subspace PCA needs at least two {neutral_label!r} samples")
    mean = _row_mean(activations, neutral_rows, batch_size=batch_size)
    covariance = np.zeros((activations.shape[1], activations.shape[1]), dtype=np.float64)
    for start in range(0, len(neutral_rows), batch_size):
        rows = neutral_rows[start : start + batch_size]
        centered = np.asarray(activations[rows], dtype=np.float64) - mean
        covariance += centered.T @ centered
    covariance /= len(neutral_rows) - 1
    covariance = (covariance + covariance.T) / 2

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    total_variance = float(eigenvalues.sum())
    if not np.isfinite(total_variance) or total_variance <= np.finfo(float).eps:
        raise ValueError("Neutral activations have no usable variance")

    ratios = eigenvalues / total_variance
    count = int(np.searchsorted(np.cumsum(ratios), variance_threshold, side="left") + 1)
    count = min(count, len(neutral_rows) - 1, activations.shape[1])
    return NeutralSubspace(
        mean=mean,
        components=eigenvectors[:, :count].T,
        explained_variance=eigenvalues[:count],
        explained_variance_ratio=ratios[:count],
        variance_threshold=variance_threshold,
        neutral_samples=len(neutral_rows),
    )


def build_pooling_representations(
    pooling: str,
    activations: np.ndarray,
    samples: pd.DataFrame,
    *,
    neutral_label: str = "neutral",
    neutral_variance_threshold: float = 0.5,
    batch_size: int = 2048,
) -> PoolingRepresentations:
    """Construct the raw, centered, and neutral-projected emotion representations."""

    if "emotion" not in samples:
        raise ValueError("samples must contain an 'emotion' column")
    labels = tuple(sorted(samples["emotion"].astype(str).unique().tolist()))
    centroids = compute_emotion_centroids(
        activations,
        samples["emotion"],
        labels=labels,
        batch_size=batch_size,
    )
    grand_mean = emotion_grand_mean(centroids, neutral_label=neutral_label)
    centered = centroids.values - grand_mean
    neutral_subspace = fit_neutral_subspace(
        activations,
        samples["emotion"],
        neutral_label=neutral_label,
        variance_threshold=neutral_variance_threshold,
        batch_size=batch_size,
    )
    projected = neutral_subspace.project_out(centered)
    return PoolingRepresentations(
        pooling=pooling,
        centroids=centroids,
        grand_mean=grand_mean,
        neutral_subspace=neutral_subspace,
        raw=centroids.values,
        centered=centered,
        neutral_projected=projected,
    )
