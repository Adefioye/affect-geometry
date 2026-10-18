from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.stats import pearsonr


def align_similarity(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    rotation, singular_sum = orthogonal_procrustes(source_centered, target_centered)
    scale = singular_sum / max(np.sum(source_centered**2), np.finfo(float).eps)
    aligned = source_centered @ rotation * scale + target.mean(axis=0)
    sse = np.sum((target - aligned) ** 2)
    sst = np.sum(target_centered**2)
    return aligned, float(1.0 - sse / sst)


def va_recovery(
    labels: list[str],
    coordinates: np.ndarray,
    reference: pd.DataFrame,
    *,
    permutations: int = 0,
    seed: int = 17,
) -> tuple[dict[str, float], pd.DataFrame]:
    if coordinates.shape[1] != 2:
        raise ValueError("V-A recovery requires a two-dimensional embedding")
    lookup = reference.set_index("emotion")
    missing = sorted(set(labels) - set(lookup.index))
    if missing:
        raise ValueError(f"Missing V-A reference values: {missing}")
    target = lookup.loc[labels, ["valence", "arousal"]].to_numpy(float)
    aligned, r_squared = align_similarity(coordinates, target)
    valence_r = float(pearsonr(aligned[:, 0], target[:, 0]).statistic)
    arousal_r = float(pearsonr(aligned[:, 1], target[:, 1]).statistic)
    p_value = np.nan
    if permutations:
        rng = np.random.default_rng(seed)
        null = []
        for _ in range(permutations):
            _, score = align_similarity(coordinates, target[rng.permutation(len(target))])
            null.append(score)
        p_value = float((1 + np.sum(np.asarray(null) >= r_squared)) / (permutations + 1))
    table = pd.DataFrame(
        {
            "emotion": labels,
            "embedding_x": coordinates[:, 0],
            "embedding_y": coordinates[:, 1],
            "aligned_x": aligned[:, 0],
            "aligned_y": aligned[:, 1],
            "valence": target[:, 0],
            "arousal": target[:, 1],
        }
    )
    return {
        "va_procrustes_r2": r_squared,
        "valence_pearson": valence_r,
        "arousal_pearson": arousal_r,
        "va_permutation_p": p_value,
    }, table
