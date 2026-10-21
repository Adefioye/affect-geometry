"""Sample-level sensitivity checks using frozen activations at preselected depths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from affect_geometry.analysis.quality import upper_triangle
from affect_geometry.analysis.robustness import procrustes_scores
from affect_geometry.embeddings.isomap import geodesic_distances
from affect_geometry.embeddings.mds import classical_mds
from affect_geometry.logging import progress


def centroid_rdm(centers, distance):
    matrix = cdist(centers, centers, metric=distance)
    np.fill_diagonal(matrix, 0)
    return matrix


def sampling_checks(run: Path, destination: Path, seed=1701, repetitions=200):
    samples = pd.read_csv(run / "dataset/samples.csv")
    labels = sorted(samples.emotion.unique())
    reference = (
        pd.read_csv(destination / "reference.csv")
        .set_index("emotion")
        .loc[labels, ["valence", "arousal"]]
        .to_numpy()
    )
    indices = [np.flatnonzero(samples.emotion.to_numpy() == label) for label in labels]
    unique_comment = ~samples.text.duplicated(keep="first").to_numpy()
    rng = np.random.default_rng(seed)
    splitrows, balance, preprocessing = [], [], []
    layers = [0, 7, 14, 21, 27]
    for layer in progress(
        layers, description="Checking text splits and balanced sampling", unit="layer"
    ):
        source = np.load(run / f"activations/last_token/layer_{layer:03d}.npy", mmap_mode="r")
        checkpoint = np.load(run / f"checkpoints/last_token/layer_{layer:03d}_pca.npz")
        components, mean = checkpoint["components"], checkpoint["mean"]
        # Groupwise reads bound memory while preserving the original pooled PCA basis.
        reduced = [
            (np.asarray(source[ix], dtype=np.float32) - mean) @ components.T for ix in indices
        ]
        centers = np.stack([g.mean(0) for g in reduced])
        rawcenters = np.stack([np.asarray(source[ix], dtype=np.float32).mean(0) for ix in indices])
        original = {
            d: np.load(run / f"rdms/last_token/layer_{layer:03d}/{d}.npz")["distance"]
            for d in ["euclidean", "cosine"]
        }
        for transform, cent in [
            ("raw_1024", rawcenters),
            ("raw_1024_centered", rawcenters - mean),
            ("pca_64_uncentered", rawcenters @ components.T),
            ("pca_64_centered", centers),
        ]:
            for distance in ["cosine", "euclidean"]:
                matrix = centroid_rdm(cent, distance)
                preprocessing.append(
                    dict(
                        layer=layer,
                        distance=distance,
                        transform=transform,
                        rdm_spearman_to_original=spearmanr(
                            upper_triangle(matrix), upper_triangle(original[distance])
                        ).statistic,
                        va_r2=procrustes_scores(
                            classical_mds(matrix, 2).coordinates[None], reference
                        )[0],
                    )
                )
        equal_n = min(map(len, reduced))
        for repeat in range(repetitions):
            balanced = np.stack(
                [g[rng.choice(len(g), equal_n, replace=False)].mean(0) for g in reduced]
            )
            for distance in ["euclidean", "cosine"]:
                matrix = centroid_rdm(balanced, distance)
                balance.append(
                    dict(
                        layer=layer,
                        distance=distance,
                        repeat=repeat,
                        n_per_emotion=equal_n,
                        rdm_spearman_to_original=spearmanr(
                            upper_triangle(matrix), upper_triangle(original[distance])
                        ).statistic,
                        va_r2=procrustes_scores(
                            classical_mds(matrix, 2).coordinates[None], reference
                        )[0],
                    )
                )
        # Fit a new PCA exclusively on training comments; validation/test never enter fitting.
        splits = {
            split: [
                ix[(samples.iloc[ix].split.to_numpy() == split) & unique_comment[ix]]
                for ix in indices
            ]
            for split in ["train", "validation", "test"]
        }
        fit_indices = np.concatenate(
            [rng.choice(ix, min(256, len(ix)), replace=False) for ix in splits["train"]]
        )
        pca = PCA(n_components=64, svd_solver="randomized", random_state=seed).fit(
            np.asarray(source[fit_indices], dtype=np.float32)
        )
        splitmats = {}
        for split, group_indices in splits.items():
            cent = np.stack(
                [
                    pca.transform(np.asarray(source[ix], dtype=np.float32)).mean(0)
                    for ix in group_indices
                ]
            )
            for distance in ["euclidean", "cosine"]:
                splitmats[split, distance] = centroid_rdm(cent, distance)
        for (split, distance), matrix in splitmats.items():
            for method in ["mds", "isomap"]:
                target = matrix if method == "mds" else geodesic_distances(matrix, 5)[0]
                x = classical_mds(target, 2).coordinates
                splitrows.append(
                    dict(
                        layer=layer,
                        distance=distance,
                        method=method,
                        split=split,
                        pca_fit_split="train",
                        pca_fit_samples=len(fit_indices),
                        smallest_class=min(map(len, splits[split])),
                        retained_split_samples=sum(map(len, splits[split])),
                        duplicate_policy="exact text deduplicated globally, train first",
                        rdm_spearman_to_train=spearmanr(
                            upper_triangle(matrix), upper_triangle(splitmats["train", distance])
                        ).statistic,
                        va_r2=procrustes_scores(x[None], reference)[0],
                    )
                )
    context = pd.read_csv(destination / "quality.csv").iloc[0]
    for name, rows in [
        ("split_sensitivity", splitrows),
        ("balanced_sensitivity", balance),
        ("preprocessing_sensitivity", preprocessing),
    ]:
        frame = pd.DataFrame(rows)
        for key in [
            "model",
            "run_id",
            "pooling",
            "seed",
            "pca_components",
            "distribution_cap",
            "quality_neighbors",
        ]:
            frame[key] = context[key]
        if "split" not in frame:
            frame["split"] = "all"
        frame.to_csv(destination / f"{name}.csv", index=False)
