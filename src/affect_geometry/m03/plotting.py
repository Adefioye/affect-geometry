"""Figures reconstructed exclusively from saved, self-contained plot CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from affect_geometry.classification.protocol import EMOTIONS
from affect_geometry.m03.rules import LENSES


def plot(root: Path) -> Path:
    root = Path(root)
    source, target = root / "plot_data", root / "plots"
    target.mkdir(exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(target / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    for name, kind in (("direct_curves", "direct"), ("contrast_curves", "contrast")):
        frame = pd.read_csv(source / f"{name}.csv")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric in zip(axes, ("accuracy", "macro_f1"), strict=True):
            for lens in LENSES:
                part = frame.loc[(frame.lens == lens) & (frame.kind == kind)]
                if kind == "contrast":
                    part = part.loc[part.beta == 0.5]
                part = part.sort_values("layer")
                x = (part.layer + 1) / part.n_blocks
                (line,) = ax.plot(x, part[metric], label=lens)
                ax.fill_between(
                    x,
                    part[f"{metric}_ci_low"],
                    part[f"{metric}_ci_high"],
                    alpha=0.12,
                    color=line.get_color(),
                )
                headline = part.loc[part.headline]
                ax.scatter(
                    (headline.layer + 1) / headline.n_blocks,
                    headline[metric],
                    marker="*",
                    s=80,
                    color=line.get_color(),
                )
            for base, style in (("native", "--"), ("greedy", ":")):
                part = frame.loc[frame.arm == base]
                if len(part):
                    ax.axhline(part.iloc[0][metric], linestyle=style, color="grey", label=base)
            ax.set(
                xlabel="Normalized post-block depth",
                ylabel=metric.replace("_", " "),
                title=f"{kind.capitalize()} · {metric.replace('_', ' ')}",
            )
            ax.legend(fontsize=8)
        save(fig, name)
        if kind == "contrast":
            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            for ax, lens in zip(axes, LENSES, strict=True):
                grid = frame.loc[frame.lens == lens].pivot(
                    index="beta", columns="layer", values="accuracy"
                )
                im = ax.imshow(grid.values, aspect="auto", origin="lower")
                ax.set(title=lens, xlabel="Post-block layer", ylabel="Beta")
                positions = np.unique(
                    np.linspace(0, len(grid.columns) - 1, min(7, len(grid.columns)), dtype=int)
                )
                ax.set_xticks(positions, [grid.columns[i] for i in positions])
                ax.set_yticks(range(len(grid.index)), grid.index)
                fig.colorbar(im, ax=ax, label="Accuracy")
            save(fig, "contrast_beta_heatmap")
    classes = pd.read_csv(source / "class_metrics.csv")
    classes = classes.loc[
        (classes.kind == "direct")
        & (classes.variant == "canonical")
        & (classes.score_mode == "total")
    ]
    for lens in LENSES:
        grid = (
            classes.loc[classes.lens == lens]
            .pivot(index="label", columns="layer", values="recall")
            .reindex(EMOTIONS)
        )
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(grid.values, aspect="auto", vmin=0, vmax=1)
        ax.set_yticks(range(28), EMOTIONS, fontsize=8)
        positions = np.unique(
            np.linspace(0, len(grid.columns) - 1, min(10, len(grid.columns)), dtype=int)
        )
        ax.set_xticks(positions, [grid.columns[i] for i in positions])
        ax.set(xlabel="Post-block layer", title=f"{lens}: per-emotion recall")
        fig.colorbar(im, ax=ax)
        save(fig, f"recall_{lens}")
    repairs = pd.read_csv(source / "repairs.csv")
    repairs = repairs.loc[repairs.kind != "baseline"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, base in zip(axes, ("native", "greedy"), strict=True):
        x = np.arange(len(repairs))
        ax.bar(x - 0.2, repairs[f"repairs_{base}"], width=0.4, label="Repairs")
        ax.bar(x + 0.2, repairs[f"damages_{base}"], width=0.4, label="Damages")
        ax.set_xticks(x, repairs.arm, rotation=35, ha="right", fontsize=7)
        ax.set(title=f"Paired changes versus {base}", ylabel="Comments")
        ax.legend()
    save(fig, "repairs_and_damage")
    confusion = pd.read_csv(source / "confusion.csv")
    confusion = confusion.loc[
        (confusion.variant == "canonical") & (confusion.score_mode == "total")
    ]
    for arm, part in confusion.groupby("arm"):
        grid = part.pivot(index="gold", columns="prediction", values="n").reindex(
            index=EMOTIONS,
            columns=EMOTIONS + ["INVALID"],
            fill_value=0,
        )
        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(grid.values, aspect="auto")
        ax.set_xticks(range(29), EMOTIONS + ["INVALID"], rotation=90, fontsize=7)
        ax.set_yticks(range(28), EMOTIONS, fontsize=7)
        ax.set(title=arm, xlabel="Prediction", ylabel="Annotation")
        fig.colorbar(im, ax=ax, label="Comments")
        save(fig, "confusion_" + arm)
    if (source / "rank_trajectories.csv").exists():
        ranks = pd.read_csv(source / "rank_trajectories.csv")
        for sid, part in ranks.groupby("sample_id"):
            fig, ax = plt.subplots(figsize=(7, 3))
            for lens, series in part.groupby("lens"):
                series = series.sort_values("layer")
                ax.plot(series.layer, series.gold_rank, marker=".", label=lens)
            ax.set(
                xlabel="Post-block layer",
                ylabel="Gold label rank",
                ylim=(28.5, 0.5),
                title=f"Retrospective example {sid[:12]}",
            )
            ax.legend()
            save(fig, "rank_" + sid[:16])
    if (source / "vocabulary_kl.csv").exists():
        frame = pd.read_csv(source / "vocabulary_kl.csv")
        fig, ax = plt.subplots(figsize=(7, 4))
        for lens in LENSES:
            series = frame.loc[frame.lens == lens].groupby("layer").kl_native_to_lens.mean()
            ax.plot(series.index, series.values, label=lens)
        ax.set(
            xlabel="Post-block layer",
            ylabel="Mean KL(native || lens)",
            title="Fixed diagnostic prompt positions",
        )
        ax.legend()
        save(fig, "vocabulary_kl")
    return target
