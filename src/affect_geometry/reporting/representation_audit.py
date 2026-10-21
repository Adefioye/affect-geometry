"""Figures rendered exclusively from the audit's self-contained plot CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

VARIANTS = [
    "LT-Raw",
    "LT-Centered",
    "LT-NeutralProjected",
    "Mean-Raw",
    "Mean-Centered",
    "Mean-NeutralProjected",
]


def plot_representation_audit(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(destination / f"{name}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    scores = pd.read_csv(source / "scores.csv")
    summary = pd.read_csv(source / "summary.csv")
    fig, ax = plt.subplots(figsize=(12, 5))
    for variant in VARIANTS:
        frame = scores[scores.variant == variant].sort_values("layer")
        ax.plot(frame.layer, frame.va_procrustes_r2, marker=".", label=variant)
    ax.set(
        xlabel="Decoder layer (zero-based)",
        ylabel="V–A Procrustes R²",
        title="Alignment across all 28 decoder layers",
        ylim=(0, 0.6),
    )
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.2)
    save(fig, "layerwise_alignment")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ordered = summary.sort_values("mean_r2")
    ax.barh(ordered.variant, ordered.mean_r2, color="#4c78a8")
    ax.scatter(ordered.max_r2, ordered.variant, color="#e45756", marker="D", label="Best layer")
    ax.set(
        xlabel="V–A Procrustes R²", title="Across-layer mean (bars) and selected maximum (diamonds)"
    )
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    save(fig, "ranked_alignment")
    fig, ax = plt.subplots(figsize=(12, 4.2))
    matrix = scores.pivot(index="variant", columns="layer", values="va_procrustes_r2").reindex(
        VARIANTS
    )
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=0.56, cmap="viridis")
    ax.set(
        yticks=range(6),
        yticklabels=VARIANTS,
        xticks=range(28),
        xlabel="Decoder layer (zero-based)",
        title="All 168 representation–layer evaluations",
    )
    fig.colorbar(im, ax=ax, label="V–A Procrustes R²")
    save(fig, "score_heatmap")

    contrasts = pd.read_csv(source / "contrasts.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, pooling in zip(axes, ["LT", "Mean"], strict=True):
        for name, frame in contrasts[contrasts.contrast.str.startswith(f"{pooling}:")].groupby(
            "contrast", sort=False
        ):
            ax.plot(frame.layer, frame.delta_r2, label=name.split(": ")[1], marker=".")
        ax.axhline(0, color="black", lw=0.8)
        ax.set(
            title=f"{pooling}: effect of transformation",
            xlabel="Layer (zero-based)",
            ylabel="Paired change in R²",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    save(fig, "transformation_effects")

    neutral = pd.read_csv(source / "neutral.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for pooling, frame in neutral.groupby("pooling", sort=False):
        axes[0].plot(frame.layer, frame.neutral_components_fitted, marker=".", label=pooling)
        axes[1].plot(frame.layer, frame.neutral_variance_explained, marker=".", label=pooling)
    axes[0].set(
        xlabel="Layer",
        ylabel="Removed principal components",
        yscale="log",
        title="Neutral subspace dimension (log scale)",
    )
    axes[1].axhline(0.5, color="black", ls="--", lw=0.8)
    axes[1].set(
        xlabel="Layer",
        ylabel="Neutral variance captured",
        ylim=(0.45, 1.03),
        title="Actual variance removed; threshold = 50%",
    )
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    save(fig, "neutral_subspace")

    geometry = pd.read_csv(source / "geometry.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for variant in ["LT-Centered", "LT-NeutralProjected", "Mean-Centered", "Mean-NeutralProjected"]:
        frame = geometry[geometry.variant == variant]
        axes[0].plot(frame.layer, frame.neutral_subspace_energy_share, label=variant)
        axes[1].plot(frame.layer, frame.mds_top1_positive_share, label=variant)
    axes[0].set(
        xlabel="Layer",
        ylabel="Fraction of centroid squared norm",
        title="Centroid energy in retained neutral directions",
        ylim=(-0.03, 1.03),
    )
    axes[1].set(
        xlabel="Layer",
        ylabel="Largest / total positive MDS eigenvalue",
        title="Concentration in one MDS direction",
        ylim=(0, 1.03),
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    save(fig, "geometry_diagnostics")

    coordinates = pd.read_csv(source / "selected_coordinates.csv")
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharex=True, sharey=True)
    for ax, variant in zip(axes.flat, VARIANTS, strict=True):
        frame = coordinates[coordinates.variant == variant]
        ax.scatter(frame.valence, frame.arousal, c="gray", marker="x", alpha=0.7, label="Reference")
        ax.scatter(
            frame.aligned_valence,
            frame.aligned_arousal,
            c=frame.valence,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            s=22,
            label="Aligned MDS",
        )
        ax.quiver(
            frame.aligned_valence,
            frame.aligned_arousal,
            frame.valence - frame.aligned_valence,
            frame.arousal - frame.aligned_arousal,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="gray",
            alpha=0.25,
            width=0.003,
        )
        ax.set(
            title=(
                f"{variant} · L{int(frame.layer.iloc[0])} · R²={frame.va_procrustes_r2.iloc[0]:.3f}"
            ),
            xlabel="Valence",
            ylabel="Arousal",
            xlim=(-1.15, 1.2),
            ylim=(-0.85, 1.0),
            aspect="equal",
        )
        ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Best layer of each variant: fitted geometry against the same reference", y=1.01)
    save(fig, "best_layer_maps")

    errors = pd.read_csv(source / "selected_errors.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, variant in zip(axes, ["LT-Centered", "Mean-NeutralProjected"], strict=True):
        frame = (
            errors[errors.variant == variant]
            .nlargest(10, "squared_error")
            .sort_values("squared_error")
        )
        ax.barh(frame.emotion, frame.squared_error, color="#4c78a8")
        ax.set(
            xlabel="Squared distance to V–A target",
            title=f"{variant} · best layer: ten largest residuals",
        )
        ax.grid(axis="x", alpha=0.2)
    save(fig, "emotion_residuals")

    counts = (
        pd.read_csv(source / "sample_counts.csv")
        .groupby("emotion")
        .sample_count.sum()
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(
        counts.index,
        counts.values,
        color=["#e45756" if e == "neutral" else "#4c78a8" for e in counts.index],
    )
    ax.set(
        xscale="log",
        xlabel="Samples across all splits (log scale)",
        title="Unequal evidence behind equally weighted emotion centroids",
    )
    ax.grid(axis="x", alpha=0.2)
    save(fig, "sample_counts")
