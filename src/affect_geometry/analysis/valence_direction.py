"""PC1 directions in native activation coordinates, with emotion-level validation."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr


@dataclass(frozen=True)
class ValenceDirection:
    vector: np.ndarray
    center: np.ndarray
    scores: np.ndarray
    counts: np.ndarray
    pearson_r: float
    pearson_p: float
    spearman_r: float
    explained_variance: float
    relative_eigengap: float


def fit_centroid_pc1(activations, sample_labels, emotion_order, valence):
    """Equal-weight, unnormalized centroids, including neutral; orient PC1 by valence."""
    x = np.asarray(activations)
    labels = np.asarray(sample_labels)
    y = np.asarray(valence, dtype=float)
    if x.ndim != 2 or len(x) != len(labels) or len(emotion_order) < 3:
        raise ValueError("Expected sample-by-hidden states and at least three emotions")
    if len(set(emotion_order)) != len(emotion_order) or set(labels) != set(emotion_order):
        raise ValueError("Emotion support must match the unique reference order exactly")
    if y.shape != (len(emotion_order),) or not np.isfinite(y).all() or np.std(y) == 0:
        raise ValueError("Invalid or constant valence reference")
    centroids, counts = [], []
    for emotion in emotion_order:
        rows = x[labels == emotion]
        if not np.isfinite(rows).all():
            raise ValueError("Nonfinite activations")
        centroids.append(rows.mean(axis=0, dtype=np.float64))
        counts.append(len(rows))
    centroids = np.stack(centroids)
    center = centroids.mean(axis=0)
    centered = centroids - center
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    if singular[0] <= 1e-12:
        raise ValueError("Emotion centroids have no variation")
    vector = vt[0]
    scores = centered @ vector
    r, p = pearsonr(scores, y)
    if not np.isfinite(r):
        raise ValueError("Undefined PC1 correlation")
    if r < 0:
        vector, scores, r = -vector, -scores, -r
    return ValenceDirection(
        vector,
        center,
        scores,
        np.asarray(counts),
        float(r),
        float(p),
        float(spearmanr(scores, y).statistic),
        float(singular[0] ** 2 / np.sum(singular**2)),
        float((singular[0] ** 2 - singular[1] ** 2) / singular[0] ** 2),
    )
