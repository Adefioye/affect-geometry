from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from affect_geometry.analysis.quality import upper_triangle
from affect_geometry.analysis.va import align_similarity


def rdm_spearman(first: np.ndarray, second: np.ndarray) -> float:
    return float(spearmanr(upper_triangle(first), upper_triangle(second)).statistic)


def embedding_procrustes_r2(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape[1] != second.shape[1]:
        dimensions = min(first.shape[1], second.shape[1])
        first, second = first[:, :dimensions], second[:, :dimensions]
    _, score = align_similarity(first, second)
    return score


def neighborhood_overlap(
    labels: list[str],
    first: np.ndarray,
    second: np.ndarray,
    neighbors: int,
) -> tuple[float, pd.DataFrame]:
    k = min(neighbors, len(labels) - 1)
    first_order = np.argsort(first, axis=1)[:, 1 : k + 1]
    second_order = np.argsort(second, axis=1)[:, 1 : k + 1]
    rows = []
    for i, emotion in enumerate(labels):
        a, b = set(first_order[i]), set(second_order[i])
        rows.append(
            {
                "emotion": emotion,
                "knn_overlap": len(a & b) / k,
                "neighborhood_change": 1.0 - len(a & b) / k,
                "neighbors_first": "|".join(labels[j] for j in first_order[i]),
                "neighbors_second": "|".join(labels[j] for j in second_order[i]),
            }
        )
    table = pd.DataFrame(rows).sort_values("neighborhood_change", ascending=False)
    return float(table["knn_overlap"].mean()), table
