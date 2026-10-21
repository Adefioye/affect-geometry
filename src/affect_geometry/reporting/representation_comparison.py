from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save_pair(figure: plt.Figure, base: Path) -> tuple[Path, Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    png, pdf = base.with_suffix(".png"), base.with_suffix(".pdf")
    figure.savefig(png, dpi=180, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def create_representation_comparison_plots(
    plot_data: str | Path,
    destination: str | Path,
) -> list[Path]:
    """Render the layerwise and across-layer summaries from plot-ready CSVs."""

    plot_data, destination = Path(plot_data), Path(destination)
    layerwise = pd.read_csv(plot_data / "representation_comparison_layerwise.csv")
    summary = pd.read_csv(plot_data / "representation_comparison_summary.csv")
    required_layerwise = {"layer", "variant", "va_procrustes_r2"}
    required_summary = {"variant", "mean_va_procrustes_r2", "max_va_procrustes_r2"}
    if missing := sorted(required_layerwise - set(layerwise.columns)):
        raise ValueError(f"Layerwise plot data is missing columns: {missing}")
    if missing := sorted(required_summary - set(summary.columns)):
        raise ValueError(f"Summary plot data is missing columns: {missing}")

    paths: list[Path] = []
    figure, axis = plt.subplots(figsize=(11, 6))
    for variant, frame in layerwise.groupby("variant", sort=False):
        ordered = frame.sort_values("layer")
        axis.plot(
            ordered["layer"],
            ordered["va_procrustes_r2"],
            marker="o",
            markersize=3,
            linewidth=1.6,
            label=variant,
        )
    axis.set(
        xlabel="Decoder layer",
        ylabel="NRC V–A Procrustes R²",
        title="Representation alignment with NRC V–A across layers",
    )
    axis.set_xticks(sorted(layerwise["layer"].unique()))
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.2)
    axis.legend(ncol=2, fontsize=8)
    paths.extend(_save_pair(figure, destination / "representation_comparison_layerwise"))

    ordered_summary = summary.sort_values("mean_va_procrustes_r2", ascending=True)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    bars = axis.barh(
        ordered_summary["variant"],
        ordered_summary["mean_va_procrustes_r2"],
        color="#4C78A8",
    )
    for bar, value in zip(
        bars,
        ordered_summary["mean_va_procrustes_r2"],
        strict=True,
    ):
        axis.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:.3f}",
            va="center",
            fontsize=8,
        )
    axis.set(
        xlabel="Mean NRC V–A Procrustes R² across layers",
        ylabel="Representation variant",
        title="Mean alignment of activation representations",
    )
    axis.set_xlim(left=0)
    axis.grid(axis="x", alpha=0.2)
    paths.extend(_save_pair(figure, destination / "representation_comparison_mean"))
    return paths
