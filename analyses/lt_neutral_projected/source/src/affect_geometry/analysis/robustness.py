"""Geometry diagnostics with explicit fitting and randomization units."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, rankdata


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p values over one explicitly defined family."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def fit_similarity(source: np.ndarray, target: np.ndarray):
    source_mean, target_mean = source.mean(0), target.mean(0)
    x, y = source - source_mean, target - target_mean
    u, s, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    rotation = u @ vt
    scale = s.sum() / max(float((x * x).sum()), np.finfo(float).eps)
    return source_mean, target_mean, rotation, scale


def apply_similarity(source: np.ndarray, fit: tuple) -> np.ndarray:
    source_mean, target_mean, rotation, scale = fit
    return (source - source_mean) @ rotation * scale + target_mean


def procrustes_scores(sources: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Batched similarity R², allowing reflection, translation and one uniform scale."""
    x = sources - sources.mean(axis=1, keepdims=True)
    y = target - target.mean(0)
    s = np.linalg.svd(np.einsum("tni,nj->tij", x, y), compute_uv=False)
    return s.sum(axis=1) ** 2 / ((x * x).sum(axis=(1, 2)) * (y * y).sum())


def permutation_procrustes(sources: np.ndarray, target: np.ndarray, permutations: np.ndarray):
    """Shared label permutations allow max-statistic control over all supplied fits."""
    x = sources - sources.mean(axis=1, keepdims=True)
    y = target - target.mean(0)
    denominator = (x * x).sum(axis=(1, 2)) * (y * y).sum()
    observed = procrustes_scores(sources, target)
    null = np.empty((len(sources), len(permutations)))
    # Chunking avoids a large temporary tensor when increasing permutation count.
    for start in range(0, len(permutations), 250):
        yp = y[permutations[start : start + 250]]
        cross = np.einsum("tni,pnj->tpij", x, yp, optimize=True)
        null[:, start : start + len(yp)] = (
            np.linalg.svd(cross, compute_uv=False).sum(-1) ** 2 / denominator[:, None]
        )
    maxima = null.max(0)
    p = (1 + (null >= observed[:, None]).sum(1)) / (len(permutations) + 1)
    pmax = (1 + (maxima[None, :] >= observed[:, None]).sum(1)) / (len(permutations) + 1)
    return observed, p, pmax, null


def leave_one_emotion_out(source: np.ndarray, target: np.ndarray, *, affine=False):
    predictions, baselines = np.empty_like(target), np.empty_like(target)
    for i in range(len(source)):
        keep = np.arange(len(source)) != i
        baselines[i] = target[keep].mean(0)
        if affine:
            design = np.column_stack([np.ones(keep.sum()), source[keep]])
            coef = np.linalg.lstsq(design, target[keep], rcond=None)[0]
            predictions[i] = np.r_[1, source[i]] @ coef
        else:
            predictions[i] = apply_similarity(
                source[i : i + 1], fit_similarity(source[keep], target[keep])
            )[0]
    error = (predictions - target) ** 2
    baseline_error = (baselines - target) ** 2
    metrics = {"loo_joint_skill": float(1 - error.sum() / baseline_error.sum())}
    for j, axis in enumerate(["valence", "arousal"]):
        metrics[f"loo_{axis}_skill"] = float(1 - error[:, j].sum() / baseline_error[:, j].sum())
        metrics[f"loo_{axis}_pearson"] = float(pearsonr(predictions[:, j], target[:, j]).statistic)
    return metrics, predictions, baselines


def standardized_rank(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, axis=-1)
    ranks -= ranks.mean(axis=-1, keepdims=True)
    return ranks / np.linalg.norm(ranks, axis=-1, keepdims=True).clip(1e-15)


def partial_rank_correlation(first: np.ndarray, second: np.ndarray, controls: np.ndarray):
    x, y = rankdata(first), rankdata(second)
    design = np.column_stack([np.ones(len(x)), rankdata(controls, axis=0)])
    x -= design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y -= design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(pearsonr(x, y).statistic)


def rsa_permutations(matrices: np.ndarray, target: np.ndarray, permutations: np.ndarray):
    """QAP-style label randomization; 378 dyads are never treated as independent."""
    n = len(target)
    tri = np.triu_indices(n, 1)
    x = standardized_rank(matrices[:, tri[0], tri[1]])
    y = squareform(pdist(target))
    ranked_matrix = squareform(rankdata(y[tri]))
    ranked_matrix -= ranked_matrix[tri].mean()
    norm = np.linalg.norm(ranked_matrix[tri])
    yp = np.stack([ranked_matrix[p[:, None], p[None, :]][tri] / norm for p in permutations])
    observed = x @ (ranked_matrix[tri] / norm)
    null = x @ yp.T
    p = (1 + (null >= observed[:, None]).sum(1)) / (len(permutations) + 1)
    return observed, p
