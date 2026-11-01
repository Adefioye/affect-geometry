"""Apply the existing neutral projection independently to each cohort."""

import numpy as np

from affect_geometry.analysis.representations import (
    compute_emotion_centroids,
    emotion_grand_mean,
    fit_neutral_subspace,
)
from affect_geometry.embeddings.geometry import UnsupportedGeometry


def preprocess(values, samples, labels, tolerance=1e-10):
    if "neutral" not in labels:
        raise UnsupportedGeometry("neutral_not_supported")
    if not np.isfinite(values).all():
        raise UnsupportedGeometry("nonfinite_activations")
    centroids = compute_emotion_centroids(values, samples.emotion, labels=labels)
    mean = emotion_grand_mean(centroids)
    try:
        neutral = fit_neutral_subspace(values, samples.emotion, variance_threshold=0.5)
    except ValueError as exc:
        raise UnsupportedGeometry(str(exc)) from exc
    r = neutral.project_out(centroids.values - mean)
    norms = np.linalg.norm(r, axis=1)
    if np.any(norms <= tolerance):
        raise UnsupportedGeometry("near_zero_projected_norm")
    u = r / norms[:, None]
    s = np.clip(u @ u.T, -1.0, 1.0)
    d = np.clip(1 - s, 0.0, 2.0)
    np.fill_diagonal(d, 0)
    return dict(
        raw_centroids=centroids.values,
        sample_counts=centroids.sample_counts,
        grand_mean=mean,
        neutral_mean=neutral.mean,
        components=neutral.components,
        explained_variance=neutral.explained_variance,
        explained_variance_ratio=neutral.explained_variance_ratio,
        threshold=np.array(0.5),
        R=r,
        U=u,
        S=s,
        D=d,
    )
