"""Geometry-only plots and a browsable local gallery, rendered from plot-ready CSVs."""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier

from affect_geometry.logging import progress
from affect_geometry.reporting.plots import create_all_plots

GROUP_COLORS = {
    "negative": "#c44e52",
    "positive": "#4c72b0",
    "ambiguous": "#ddad38",
    "neutral": "#555555",
}


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _regions(axis: plt.Axes, frame: pd.DataFrame) -> None:
    """Reconstruct taxonomy kNN shading exclusively from the exported coordinate table."""
    x, y = frame.coordinate_1.to_numpy(), frame.coordinate_2.to_numpy()
    span_x, span_y = max(np.ptp(x), 1e-8), max(np.ptp(y), 1e-8)
    xx, yy = np.meshgrid(
        np.linspace(x.min() - 0.12 * span_x, x.max() + 0.12 * span_x, 150),
        np.linspace(y.min() - 0.12 * span_y, y.max() + 0.12 * span_y, 150),
    )
    groups = list(GROUP_COLORS)
    classifier = KNeighborsClassifier(
        n_neighbors=min(int(frame.partition_neighbors.iloc[0]), len(frame)),
        weights="uniform",
    ).fit(np.column_stack([x, y]), frame.emotion_group.map(groups.index))
    zz = classifier.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    axis.contourf(
        xx,
        yy,
        zz,
        levels=np.arange(-0.5, len(groups) + 0.5),
        cmap=ListedColormap(list(GROUP_COLORS.values())),
        alpha=0.10,
    )


def render_projected_plots(root: Path) -> Path:
    """Save all coordinate plots and summaries; return an HTML gallery path."""
    destination = root / "plots"
    create_all_plots(root / "plot_data", destination, include_coordinates=False)
    frame = pd.read_csv(root / "plot_data" / "exploration_coordinates.csv")
    grouped = frame.groupby(["layer", "distance", "method", "dimensions"], sort=True)
    for (layer, distance, method, dimensions), group in progress(
        grouped,
        description="Rendering neutral-centered exploration",
        total=grouped.ngroups,
        unit="plot",
    ):
        dimensions = int(dimensions)
        figure = plt.figure(figsize=(12, 3.8) if dimensions == 1 else (10, 8))
        axis = figure.add_subplot(111, projection="3d" if dimensions == 3 else None)
        if dimensions == 2:
            _regions(axis, group)
            axis.axvline(0, color="grey", ls="--", lw=0.8)
            axis.axhline(0, color="grey", ls="--", lw=0.8)
            for x, y, name in [
                (0.97, 0.97, "I"),
                (0.03, 0.97, "II"),
                (0.03, 0.03, "III"),
                (0.97, 0.03, "IV"),
            ]:
                axis.text(x, y, name, transform=axis.transAxes, ha="center", alpha=0.5)
        for emotion_group, part in group.groupby("emotion_group", sort=True):
            coords = [part.coordinate_1]
            coords.append(part.coordinate_2 if dimensions >= 2 else np.zeros(len(part)))
            if dimensions == 3:
                coords.append(part.coordinate_3)
            axis.scatter(
                *coords,
                color=GROUP_COLORS[emotion_group],
                label=emotion_group,
                marker="*" if emotion_group == "neutral" else "o",
                s=160 if emotion_group == "neutral" else 40,
            )
        for index, row in enumerate(group.itertuples()):
            if dimensions == 3:
                axis.text(
                    row.coordinate_1, row.coordinate_2, row.coordinate_3, row.emotion, fontsize=7
                )
            else:
                y = row.coordinate_2 if dimensions == 2 else 0
                axis.annotate(
                    row.emotion,
                    (row.coordinate_1, y),
                    fontsize=7,
                    xytext=(4, 5 + 11 * (index % 3) if dimensions == 1 else 4),
                    textcoords="offset points",
                    rotation=45 if dimensions == 1 else 0,
                )
        axis.set_xlabel("Coordinate 1 (neutral = 0)")
        if dimensions >= 2:
            axis.set_ylabel("Coordinate 2 (neutral = 0)")
        else:
            axis.set_yticks([])
            axis.set_ylim(-0.03, 0.12)
            axis.axvline(0, color="grey", ls="--", lw=0.8)
        if dimensions == 2:
            axis.set_aspect("equal", adjustable="box")
        if dimensions == 3:
            axis.set_zlabel("Coordinate 3 (neutral = 0)")
            spans = np.ptp(group[[f"coordinate_{i}" for i in range(1, 4)]].to_numpy(), axis=0)
            axis.set_box_aspect(tuple(np.maximum(spans, 1e-8)))
        axis.set_title(
            f"{group.model.iloc[0]} · LT-NeutralProjected\n"
            f"{method.upper()} {dimensions}D · {distance} · L{layer}"
        )
        axis.legend(fontsize=8, loc="best")
        _save(
            figure,
            destination
            / "exploration"
            / f"layer_{int(layer):03d}"
            / f"{distance}_{method}_{dimensions}d.png",
        )

    recovery = pd.read_csv(root / "plot_data" / "vad_recovery.csv")
    figure, axis = plt.subplots(figsize=(11, 6))
    for (distance, method), group in recovery.groupby(["distance", "method"]):
        axis.plot(group.layer, group.vad_procrustes_r2, label=f"{distance}/{method}", marker="o")
    axis.set(
        xlabel="Layer",
        ylabel="NRC VAD similarity Procrustes R² (3D)",
        title="Separate reference evaluation · LT-NeutralProjected",
    )
    axis.legend(fontsize=7, ncol=3)
    _save(figure, destination / "layerwise" / "vad_recovery.png")
    return write_gallery(root)


def write_gallery(root: Path) -> Path:
    """Display saved figures with method/dimension/layer filters, without a web server."""
    paths = sorted((root / "plots").rglob("*.png"))
    cards = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        title = path.relative_to(root / "plots").as_posix()
        cards.append(
            f'<figure data-name="{html.escape(title)}"><figcaption>{html.escape(title)}'
            f'</figcaption><a href="{html.escape(relative)}"><img loading="lazy" '
            f'src="{html.escape(relative)}" alt="{html.escape(title)}"></a></figure>'
        )
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>LT-NeutralProjected exploration</title><style>
body{font:16px system-ui;margin:24px;background:#f6f7f9;color:#202124}
input{padding:10px;width:min(600px,90%)} main{display:grid;grid-template-columns:repeat(auto-fit,
minmax(430px,1fr));gap:16px}figure{margin:0;background:white;padding:12px}img{width:100%}
figcaption{overflow-wrap:anywhere}figure[hidden]{display:none}
</style><h1>LT-NeutralProjected</h1>
<p>All 27 emotions plus neutral; MDS and Isomap in 1D, 2D, and 3D. Coordinates are translated
to neutral without reference alignment. 2D axes cross at neutral; shading uses the GoEmotions
taxonomy and k nearest neighbors. Embedding axes and quadrants have no assigned V/A meaning.</p>
<p>All figures are recreated from the CSV files in <a href="plot_data/">plot_data/</a>.
Reference recovery appears only in separate summary figures. Click any plot for full size.</p>
<label>Filter plots (e.g. cosine isomap 2d or layer_017):
<input id="filter" type="search"></label><p id="count"></p><main>"""
    document += (
        "\n".join(cards)
        + r"""</main><script>
const cards=[...document.querySelectorAll('figure')];
function filter(){const words=document.querySelector('#filter').value.toLowerCase().split(/\s+/);
let count=0;for(const card of cards){card.hidden=!words.every(w=>card.dataset.name.includes(w));
if(!card.hidden)count++}document.querySelector('#count').textContent=count+' plots'}
document.querySelector('#filter').addEventListener('input',filter);filter();</script></html>"""
    )
    path = root / "exploration.html"
    path.write_text(document, encoding="utf-8")
    (root / "plot_manifest.json").write_text(
        json.dumps(
            {
                "exploration": {
                    "csv": "plot_data/exploration_coordinates.csv",
                    "figures": "plots/exploration/",
                    "origin": "neutral",
                },
                "vad_recovery": {
                    "csv": "plot_data/vad_recovery.csv",
                    "figure": "plots/layerwise/vad_recovery.png",
                },
                "gallery": "exploration.html",
                "figure_count": len(paths),
            },
            indent=2,
        )
        + "\n"
    )
    return path
