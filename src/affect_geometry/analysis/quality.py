from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix)[np.triu_indices(len(matrix), 1)]


def metric_stress(target: np.ndarray, coordinates: np.ndarray) -> float:
    observed = squareform(pdist(coordinates))
    numerator = np.sum((upper_triangle(target) - upper_triangle(observed)) ** 2)
    denominator = np.sum(upper_triangle(target) ** 2)
    return float(np.sqrt(numerator / denominator)) if denominator else 0.0


def shepard_correlation(target: np.ndarray, coordinates: np.ndarray) -> float:
    observed = squareform(pdist(coordinates))
    return float(spearmanr(upper_triangle(target), upper_triangle(observed)).statistic)


def _ranks(distances: np.ndarray) -> np.ndarray:
    order = neighbor_order(distances)
    ranks = np.zeros_like(distances, dtype=int)
    ranks[np.arange(len(distances))[:, None], order] = np.arange(1, len(distances))
    return ranks


def neighbor_order(distances: np.ndarray) -> np.ndarray:
    """Order other observations, excluding self even when distances tie at zero."""
    values = np.array(distances, dtype=float, copy=True)
    np.fill_diagonal(values, np.inf)
    return np.argsort(values, axis=1, kind="stable")[:, : len(values) - 1]


def trustworthiness_continuity(
    original: np.ndarray,
    coordinates: np.ndarray,
    neighbors: int,
) -> tuple[float, float]:
    embedded = squareform(pdist(coordinates))
    n = len(original)
    if not 1 <= neighbors < n / 2:
        raise ValueError("neighbors must be positive and less than n_samples / 2")
    k = neighbors
    original_order = neighbor_order(original)[:, :k]
    embedded_order = neighbor_order(embedded)[:, :k]
    original_ranks, embedded_ranks = _ranks(original), _ranks(embedded)
    trust_penalty = 0.0
    continuity_penalty = 0.0
    for i in range(n):
        original_set = set(original_order[i])
        embedded_set = set(embedded_order[i])
        trust_penalty += sum(original_ranks[i, j] - k for j in embedded_set - original_set)
        continuity_penalty += sum(embedded_ranks[i, j] - k for j in original_set - embedded_set)
    normalizer = 2.0 / (n * k * (2 * n - 3 * k - 1))
    return 1.0 - normalizer * trust_penalty, 1.0 - normalizer * continuity_penalty


def geodesic_ratio_summary(direct: np.ndarray, geodesic: np.ndarray) -> dict[str, float]:
    direct_values, geodesic_values = upper_triangle(direct), upper_triangle(geodesic)
    valid = direct_values > np.finfo(float).eps
    ratios = geodesic_values[valid] / direct_values[valid]
    return {
        "geodesic_direct_ratio_mean": float(np.mean(ratios)),
        "geodesic_direct_ratio_median": float(np.median(ratios)),
        "geodesic_direct_ratio_p95": float(np.quantile(ratios, 0.95)),
        "geodesic_direct_ratio_max": float(np.max(ratios)),
    }


def embedding_quality(
    target: np.ndarray,
    coordinates: np.ndarray,
    neighbors: int,
) -> dict[str, float]:
    trust, continuity = trustworthiness_continuity(target, coordinates, neighbors)
    return {
        "stress": metric_stress(target, coordinates),
        "shepard_spearman": shepard_correlation(target, coordinates),
        "trustworthiness": trust,
        "continuity": continuity,
    }
