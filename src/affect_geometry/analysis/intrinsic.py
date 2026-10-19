from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

from affect_geometry.analysis.quality import upper_triangle


def eigenvalue_summary(eigenvalues: np.ndarray) -> dict[str, float | int]:
    positive = np.asarray(eigenvalues)[np.asarray(eigenvalues) > 0]
    total = positive.sum()
    cumulative = np.cumsum(positive) / total if total else np.zeros_like(positive)

    def dimension_for(fraction: float) -> int:
        return int(np.searchsorted(cumulative, fraction) + 1) if len(cumulative) else 0

    participation = float(total**2 / np.sum(positive**2)) if len(positive) else 0.0
    negative_mass = float(np.abs(eigenvalues[eigenvalues < 0]).sum())
    absolute_mass = float(np.abs(eigenvalues).sum())
    return {
        "dimensions_80": dimension_for(0.80),
        "dimensions_90": dimension_for(0.90),
        "dimensions_95": dimension_for(0.95),
        "participation_ratio": participation,
        "negative_eigenvalue_fraction": negative_mass / absolute_mass if absolute_mass else 0.0,
    }


def residual_variance(geodesic: np.ndarray, coordinates: np.ndarray) -> float:
    from scipy.spatial.distance import pdist, squareform

    embedded = squareform(pdist(coordinates))
    correlation = pearsonr(upper_triangle(geodesic), upper_triangle(embedded)).statistic
    return float(1.0 - correlation**2)


def elbow_dimension(dimensions: list[int], residuals: list[float]) -> int:
    """Return the largest perpendicular departure from the end-to-end line."""
    if len(dimensions) <= 2:
        return dimensions[-1]
    points = np.column_stack([dimensions, residuals]).astype(float)
    scale = np.ptp(points, axis=0)
    scale[scale == 0] = 1
    points = (points - points.min(axis=0)) / scale
    start, end = points[0], points[-1]
    line = end - start
    offsets = points - start
    distances = np.abs(line[0] * offsets[:, 1] - line[1] * offsets[:, 0])
    distances /= max(np.linalg.norm(line), 1e-12)
    return int(dimensions[int(np.argmax(distances))])
