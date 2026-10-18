from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _coordinate_plots(plot_data: Path, destination: Path) -> None:
    coordinates = pd.read_csv(plot_data / "coordinates.csv")
    grouping = ["pooling", "layer", "distance", "method", "dimensions"]
    for key, frame in coordinates.groupby(grouping, sort=True):
        pooling, layer, distance_name, method, dimensions = key
        dimensions = int(dimensions)
        figure = plt.figure(figsize=(8, 6))
        title = f"{method.upper()} · {distance_name} · layer {layer} · {pooling}"
        if dimensions == 1:
            axis = figure.add_subplot(111)
            axis.scatter(
                frame["coordinate_1"], np.zeros(len(frame)), c=frame["valence"], cmap="coolwarm"
            )
            for row in frame.itertuples():
                axis.annotate(
                    row.emotion,
                    (row.coordinate_1, 0),
                    xytext=(0, 6),
                    textcoords="offset points",
                    rotation=45,
                )
            axis.set_yticks([])
            axis.set_xlabel("Coordinate 1")
        elif dimensions == 2:
            axis = figure.add_subplot(111)
            points = axis.scatter(
                frame["coordinate_1"], frame["coordinate_2"], c=frame["valence"], cmap="coolwarm"
            )
            for row in frame.itertuples():
                axis.annotate(row.emotion, (row.coordinate_1, row.coordinate_2), fontsize=7)
            axis.set(xlabel="Coordinate 1", ylabel="Coordinate 2")
            figure.colorbar(points, ax=axis, label="NRC valence")
        else:
            axis = figure.add_subplot(111, projection="3d")
            points = axis.scatter(
                frame["coordinate_1"],
                frame["coordinate_2"],
                frame["coordinate_3"],
                c=frame["valence"],
                cmap="coolwarm",
            )
            for row in frame.itertuples():
                axis.text(
                    row.coordinate_1, row.coordinate_2, row.coordinate_3, row.emotion, fontsize=6
                )
            axis.set(xlabel="Coordinate 1", ylabel="Coordinate 2", zlabel="Coordinate 3")
            figure.colorbar(points, ax=axis, label="NRC valence", shrink=0.7)
        axis.set_title(title)
        path = destination / "embeddings" / pooling / f"layer_{int(layer):03d}"
        _save(figure, path / f"{distance_name}_{method}_{dimensions}d.png")


def _layerwise_plots(plot_data: Path, destination: Path) -> None:
    frame = pd.read_csv(plot_data / "layerwise_metrics.csv")
    metrics = [
        "stress",
        "trustworthiness",
        "continuity",
        "va_procrustes_r2",
        "participation_ratio",
        "dimensions_90",
    ]
    for pooling, pooling_frame in frame.groupby("pooling"):
        for metric in metrics:
            if metric not in pooling_frame or pooling_frame[metric].isna().all():
                continue
            figure, axis = plt.subplots(figsize=(10, 6))
            relevant = pooling_frame
            if metric == "va_procrustes_r2":
                relevant = relevant.loc[relevant["dimensions"] == 2]
            for key, group in relevant.groupby(["distance", "method", "dimensions"]):
                distance_name, method, dimensions = key
                axis.plot(
                    group["layer"],
                    group[metric],
                    marker="o",
                    markersize=2,
                    label=f"{distance_name}/{method}/{int(dimensions)}D",
                )
            axis.set(
                xlabel="Layer",
                ylabel=metric.replace("_", " ").title(),
                title=f"{metric} · {pooling}",
            )
            axis.legend(fontsize=6, ncol=3)
            _save(figure, destination / "layerwise" / f"{pooling}_{metric}.png")


def _intrinsic_plots(plot_data: Path, destination: Path) -> None:
    residuals = pd.read_csv(plot_data / "isomap_residual_variance.csv")
    for (pooling, distance_name), frame in residuals.groupby(["pooling", "distance"]):
        figure, axis = plt.subplots(figsize=(9, 5))
        for layer, group in frame.groupby("layer"):
            axis.plot(
                group["dimensions"], group["residual_variance"], alpha=0.45, label=f"L{layer}"
            )
        axis.set(
            xlabel="Dimensions",
            ylabel="Residual variance",
            title=f"Isomap residual variance · {distance_name} · {pooling}",
        )
        if frame["layer"].nunique() <= 12:
            axis.legend(fontsize=7, ncol=2)
        _save(
            figure, destination / "intrinsic" / f"{pooling}_{distance_name}_residual_variance.png"
        )


def _agreement_plots(plot_data: Path, destination: Path) -> None:
    frame = pd.read_csv(plot_data / "distance_agreement.csv")
    for (pooling, layer), group in frame.groupby(["pooling", "layer"]):
        names = sorted(set(group["distance_i"]) | set(group["distance_j"]))
        matrix = np.eye(len(names))
        lookup = {name: index for index, name in enumerate(names)}
        for row in group.itertuples():
            i, j = lookup[row.distance_i], lookup[row.distance_j]
            matrix[i, j] = matrix[j, i] = row.spearman
        figure, axis = plt.subplots(figsize=(7, 6))
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
        axis.set_yticks(range(len(names)), names)
        axis.set_title(f"RDM Spearman agreement · layer {layer} · {pooling}")
        figure.colorbar(image, ax=axis)
        _save(figure, destination / "agreement" / f"{pooling}_layer_{int(layer):03d}.png")


def create_all_plots(plot_data: Path, destination: Path) -> None:
    """Create every plot exclusively from CSVs in ``plot_data``."""
    _coordinate_plots(plot_data, destination)
    _layerwise_plots(plot_data, destination)
    _intrinsic_plots(plot_data, destination)
    _agreement_plots(plot_data, destination)
