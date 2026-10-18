from __future__ import annotations

from collections.abc import Callable
from itertools import combinations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.covariance import LedoitWolf

from affect_geometry.logging import progress

ArrayGroups = list[np.ndarray]


def _pairwise(
    groups: ArrayGroups,
    function: Callable[[np.ndarray, np.ndarray], float],
    description: str,
) -> np.ndarray:
    result = np.zeros((len(groups), len(groups)), dtype=np.float64)
    pairs = list(combinations(range(len(groups)), 2))
    for i, j in progress(
        pairs,
        description=description,
        unit="pair",
        leave=False,
    ):
        result[i, j] = result[j, i] = function(groups[i], groups[j])
    return result


def centroid_euclidean(groups: ArrayGroups, **_: object) -> tuple[np.ndarray, np.ndarray]:
    centers = np.stack([group.mean(axis=0) for group in groups])
    matrix = cdist(centers, centers, metric="euclidean")
    return matrix, matrix.copy()


def centroid_cosine(groups: ArrayGroups, **_: object) -> tuple[np.ndarray, np.ndarray]:
    centers = np.stack([group.mean(axis=0) for group in groups])
    matrix = cdist(centers, centers, metric="cosine")
    matrix = np.nan_to_num(matrix, nan=0.0)
    np.fill_diagonal(matrix, 0.0)
    return matrix, matrix.copy()


def energy_distance(groups: ArrayGroups, **_: object) -> tuple[np.ndarray, np.ndarray]:
    def statistic(x: np.ndarray, y: np.ndarray) -> float:
        cross = cdist(x, y).mean()
        xx = cdist(x, x)
        yy = cdist(y, y)
        within_x = xx.sum() / max(len(x) * (len(x) - 1), 1)
        within_y = yy.sum() / max(len(y) * (len(y) - 1), 1)
        return float(2 * cross - within_x - within_y)

    raw = _pairwise(groups, statistic, "Computing Energy emotion pairs")
    return raw, np.sqrt(np.maximum(raw, 0.0))


def _median_bandwidth(groups: ArrayGroups, rng: np.random.Generator) -> float:
    combined = np.concatenate(groups)
    if len(combined) > 2048:
        combined = combined[rng.choice(len(combined), 2048, replace=False)]
    distances = cdist(combined, combined)
    positive = distances[np.triu_indices(len(combined), 1)]
    positive = positive[positive > 0]
    return float(np.median(positive)) if len(positive) else 1.0


def rbf_mmd(
    groups: ArrayGroups,
    *,
    seed: int = 17,
    bandwidth_multipliers: list[float] | None = None,
    **_: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Unbiased multi-bandwidth RBF MMD squared and its clipped square root."""
    rng = np.random.default_rng(seed)
    median = _median_bandwidth(groups, rng)
    multipliers = bandwidth_multipliers or [0.5, 1.0, 2.0]
    bandwidths = np.asarray(multipliers, dtype=float) * max(median, np.finfo(float).eps)

    def kernels(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        squared = cdist(x, y, metric="sqeuclidean")
        return np.mean(
            [np.exp(-squared / (2.0 * bandwidth**2)) for bandwidth in bandwidths],
            axis=0,
        )

    def statistic(x: np.ndarray, y: np.ndarray) -> float:
        kxx, kyy, kxy = kernels(x, x), kernels(y, y), kernels(x, y)
        xx = (kxx.sum() - np.trace(kxx)) / max(len(x) * (len(x) - 1), 1)
        yy = (kyy.sum() - np.trace(kyy)) / max(len(y) * (len(y) - 1), 1)
        return float(xx + yy - 2.0 * kxy.mean())

    raw = _pairwise(groups, statistic, "Computing RBF-MMD emotion pairs")
    return raw, np.sqrt(np.maximum(raw, 0.0))


def sliced_wasserstein(
    groups: ArrayGroups,
    *,
    seed: int = 17,
    projections: int = 256,
    **_: object,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(projections, groups[0].shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(min=1e-12)
    projected = [group @ directions.T for group in groups]

    def statistic(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean([wasserstein_distance(x[:, k], y[:, k]) for k in range(projections)]))

    matrix = _pairwise(projected, statistic, "Computing Sliced-Wasserstein pairs")
    return matrix, matrix.copy()


def crossnobis(
    groups: ArrayGroups,
    *,
    seed: int = 17,
    folds: int = 5,
    **_: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-validated squared Mahalanobis distance with pooled shrinkage precision."""
    min_count = min(map(len, groups))
    n_folds = min(folds, min_count)
    if n_folds < 2:
        raise ValueError("Crossnobis needs at least two samples per emotion")
    rng = np.random.default_rng(seed)
    assignments: list[np.ndarray] = []
    for group in groups:
        order = rng.permutation(len(group))
        assignment = np.empty(len(group), dtype=int)
        assignment[order] = np.arange(len(group)) % n_folds
        assignments.append(assignment)
    precisions = []
    for fold in progress(
        range(n_folds),
        description="Estimating Crossnobis precision",
        unit="fold",
        leave=False,
    ):
        residuals = []
        for group, assignment in zip(groups, assignments, strict=True):
            training = group[assignment != fold]
            residuals.append(training - training.mean(axis=0))
        precisions.append(LedoitWolf().fit(np.concatenate(residuals)).precision_)
    raw = np.zeros((len(groups), len(groups)), dtype=np.float64)
    pairs = list(combinations(range(len(groups)), 2))
    for i, j in progress(
        pairs,
        description="Computing Crossnobis emotion pairs",
        unit="pair",
        leave=False,
    ):
        estimates = []
        for fold in range(n_folds):
            delta_test = groups[i][assignments[i] == fold].mean(0) - groups[j][
                assignments[j] == fold
            ].mean(0)
            delta_train = groups[i][assignments[i] != fold].mean(0) - groups[j][
                assignments[j] != fold
            ].mean(0)
            estimates.append(
                float(delta_test @ precisions[fold] @ delta_train) / groups[i].shape[1]
            )
        raw[i, j] = raw[j, i] = float(np.mean(estimates))
    return raw, np.sqrt(np.maximum(raw, 0.0))


DISTANCE_FUNCTIONS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {
    "crossnobis": crossnobis,
    "rbf_mmd": rbf_mmd,
    "energy": energy_distance,
    "sliced_wasserstein": sliced_wasserstein,
    "cosine": centroid_cosine,
    "euclidean": centroid_euclidean,
}
