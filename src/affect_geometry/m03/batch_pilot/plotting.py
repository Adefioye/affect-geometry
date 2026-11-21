"""Recreate the pilot report figure from the saved agreement table."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


def plot_flips(root):
    table = pd.read_csv(root / "plot_data" / "layer_agreement.csv")
    headings = {
        "total": "Label + stop token (primary)",
        "label_only": "Label tokens",
    }
    lenses = {
        "logit": ("Logit lens", "#2166ac", "o"),
        "tuned": ("Tuned lens", "#a43422", "^"),
        "jacobian": ("Jacobian lens", "#087c69", "s"),
        "native": ("Native readout", "#3d3d3d", "D"),
    }
    fig, axes = plt.subplots(len(headings), 1, sharex=True, figsize=(10, 5.2), layout="constrained")
    for (mode, title), ax in zip(headings.items(), axes, strict=True):
        for lens, (label, color, marker) in lenses.items():
            part = table.loc[(table.score_mode == mode) & (table.lens == lens)].sort_values("layer")
            ax.plot(
                part.layer,
                part.flip_count,
                label=label,
                color=color,
                linewidth=1.2,
                marker=marker,
                markersize=3.5,
                alpha=0.9,
            )
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("Flips / 64")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(-0.25, max(1.25, table.flip_count.max() + 0.5))
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=4, frameon=False
    )
    axes[-1].set_xlabel("Post-block layer index (zero-based)")
    target = root / "analysis-report" / "plots" / "top1-flips-by-layer.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160)
    plt.close(fig)
