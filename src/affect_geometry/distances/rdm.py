from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from affect_geometry.distances.metrics import DISTANCE_FUNCTIONS


@dataclass(frozen=True)
class RDMResult:
    labels: list[str]
    raw: np.ndarray
    distance: np.ndarray


def stratified_groups(
    activations: np.ndarray,
    emotions: pd.Series,
    max_samples: int,
    seed: int,
) -> tuple[list[str], list[np.ndarray], pd.DataFrame]:
    labels = sorted(emotions.unique().tolist())
    rng = np.random.default_rng(seed)
    groups: list[np.ndarray] = []
    records = []
    values = emotions.to_numpy()
    for label in labels:
        available = np.flatnonzero(values == label)
        selected = available
        if len(available) > max_samples:
            selected = np.sort(rng.choice(available, max_samples, replace=False))
        groups.append(np.asarray(activations[selected], dtype=np.float64))
        records.extend({"emotion": label, "activation_row": int(i)} for i in selected)
    return labels, groups, pd.DataFrame(records)


def reduce_groups(
    groups: list[np.ndarray],
    n_components: int,
    seed: int,
) -> tuple[list[np.ndarray], PCA | None]:
    combined = np.concatenate(groups)
    components = min(n_components, combined.shape[1], len(combined) - 1)
    if components >= combined.shape[1]:
        return groups, None
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    transformed = pca.fit_transform(combined)
    reduced, start = [], 0
    for group in groups:
        reduced.append(transformed[start : start + len(group)])
        start += len(group)
    return reduced, pca


def compute_rdm(
    name: str,
    labels: list[str],
    groups: list[np.ndarray],
    *,
    seed: int,
    folds: int,
    bandwidth_multipliers: list[float],
    projections: int,
) -> RDMResult:
    if name not in DISTANCE_FUNCTIONS:
        raise ValueError(f"Unknown distance {name!r}")
    raw, distance = DISTANCE_FUNCTIONS[name](
        groups,
        seed=seed,
        folds=folds,
        bandwidth_multipliers=bandwidth_multipliers,
        projections=projections,
    )
    return RDMResult(labels, raw, distance)


def save_rdm(
    result: RDMResult,
    destination: Path,
    context: dict[str, object],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination.with_suffix(".npz"),
        labels=np.asarray(result.labels),
        raw=result.raw,
        distance=result.distance,
    )
    rows = []
    for i, emotion_i in enumerate(result.labels):
        for j in range(i + 1, len(result.labels)):
            rows.append(
                {
                    **context,
                    "emotion_i": emotion_i,
                    "emotion_j": result.labels[j],
                    "raw_dissimilarity": result.raw[i, j],
                    "embedding_distance": result.distance[i, j],
                }
            )
    pd.DataFrame(rows).to_csv(destination.with_suffix(".csv"), index=False)


def load_rdm(path: Path) -> RDMResult:
    data = np.load(path, allow_pickle=False)
    return RDMResult(data["labels"].astype(str).tolist(), data["raw"], data["distance"])
