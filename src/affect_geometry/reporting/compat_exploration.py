"""Figures consume standalone plot-ready CSVs saved by the notebook."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def model_figures(csv_dir, model, plot_dir):
    csv_dir, plot_dir = Path(csv_dir), Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    distribution = pd.read_csv(csv_dir / "distribution.csv").query("model == @model")
    per_gold = pd.read_csv(csv_dir / "per_gold.csv").query("model == @model and available")
    matrix = distribution.pivot(index="gold", columns="prediction", values="pct_of_failures")
    fig, ax = plt.subplots(figsize=(15, 8), layout="constrained")
    im = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0, vmax=100)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=65, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set(
        xlabel="First-word prediction",
        ylabel="Gold emotion",
        title=(
            f"{model}: incorrect responses (% within each gold label)\nBalanced activation cohort"
        ),
    )
    fig.colorbar(im, ax=ax, label="% of relaxed failures")
    fig.savefig(plot_dir / f"{model}-confusion.png", dpi=160)
    plt.show()

    fig, ax = plt.subplots(figsize=(13, 6), layout="constrained")
    bottom = np.zeros(len(per_gold))
    for col, label in [
        ("unparseable", "No valid first label"),
        ("wrong_label_with_format_issue", "Wrong label + extra formatting"),
        ("rescued_correct", "Correct first label; strict format rejected"),
    ]:
        values = per_gold[col].to_numpy()
        ax.bar(per_gold.gold, values, bottom=bottom, label=label)
        bottom += values
    ax.tick_params(axis="x", rotation=65)
    ax.set(
        ylabel="Response count",
        title=f"{model}: strict formatting issues by gold label\n"
        "Balanced activation cohort; rescued correct responses are not relaxed failures",
    )
    ax.legend(fontsize=8)
    fig.savefig(plot_dir / f"{model}-formatting.png", dpi=160)
    plt.show()

    wrong = distribution[
        (distribution.gold != distribution.prediction) & distribution.prediction.ne("[unparseable]")
    ]
    groups = list(wrong.groupby("gold", sort=True))
    fig, ax = plt.subplots(figsize=(13, 6), layout="constrained")
    ax.boxplot([g["count"] for _, g in groups], tick_labels=[k for k, _ in groups], showmeans=True)
    ax.tick_params(axis="x", rotation=65)
    ax.set(
        ylabel="Count per wrong emotion label",
        title=f"{model}: wrong-label count distribution\n"
        "27 possible wrong labels per gold label; zero-count labels included",
    )
    fig.savefig(plot_dir / f"{model}-wrong-label-boxplot.png", dpi=160)
    plt.show()


def population_boxplots(csv_dir, plot_dir):
    frame = pd.read_csv(Path(csv_dir) / "population.csv")
    groups = list(frame.groupby("model", sort=True))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for ax, metric, label in zip(
        axes,
        ["incorrect", "incorrect_pct"],
        ["Incorrect response count", "Incorrect responses (%)"],
    ):
        values = [g[metric].dropna().to_numpy() for _, g in groups]
        ax.boxplot(values, tick_labels=[k for k, _ in groups], showmeans=True)
        for i, points in enumerate(values, 1):
            ax.scatter(np.full(len(points), i), points, s=15, alpha=0.5)
        ax.set(ylabel=label, title="One observation per gold label")
    fig.suptitle("Full classification population: failures across all 28 gold labels")
    fig.savefig(Path(plot_dir) / "population-boxplots.png", dpi=160)
    plt.show()


def wrong_label_histograms(csv_dir, model, plot_dir):
    """One count/percentage figure per available gold; explicitly list absent golds."""
    from IPython.display import Markdown, display

    from affect_geometry.logging import progress

    csv_dir = Path(csv_dir)
    destination = Path(plot_dir) / "per-gold-histograms" / model
    destination.mkdir(parents=True, exist_ok=True)
    distribution = pd.read_csv(csv_dir / "distribution.csv").query("model == @model")
    coverage = pd.read_csv(csv_dir / "per_gold.csv").query("model == @model")
    absent = coverage.loc[~coverage.available, "gold"].tolist()
    display(
        Markdown(
            f"**{model}: gold labels without sampled response records:** "
            + (", ".join(absent) if absent else "none")
            + ". No histogram can be estimated for these labels; absence is not zero errors."
        )
    )
    available = coverage.loc[coverage.available, "gold"].tolist()
    for gold in progress(available, description=f"{model}: per-gold histograms", unit="label"):
        part = distribution.loc[distribution.gold.eq(gold)]
        wrong = part.loc[
            part.prediction.ne(gold) & part.prediction.ne("[unparseable]")
        ].sort_values(["count", "prediction"], ascending=[False, True])
        n_wrong = int(wrong["count"].sum())
        n_invalid = int(part.loc[part.prediction.eq("[unparseable]"), "count"].sum())
        fig, axes = plt.subplots(1, 2, figsize=(13, 9), sharey=True, layout="constrained")
        positions = np.arange(len(wrong))
        for ax, column, xlabel, color in zip(
            axes,
            ["count", "pct_of_wrong_labels"],
            ["Incorrect response count", "% of valid wrong-emotion responses"],
            ["#3566a1", "#d17a28"],
        ):
            values = wrong[column].to_numpy()
            bars = ax.barh(positions, values, color=color)
            ax.set_xlabel(xlabel)
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)
            if column == "count":
                ax.set_xlim(0, max(1, n_wrong and wrong["count"].max()) * 1.2)
                labels = [str(int(v)) if v else "" for v in values]
            else:
                ax.set_xlim(0, 110)
                labels = [f"{v:.2f}%" if pd.notna(v) and v > 0 else "" for v in values]
            ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
        axes[0].set_yticks(positions, wrong.prediction)
        axes[0].invert_yaxis()
        axes[0].set_ylabel("Generated wrong emotion (first-word parser)")
        fig.suptitle(
            f"{model} | gold: {gold}\n"
            f"{n_wrong:,} valid wrong labels; {n_invalid:,} unparseable failures excluded "
            "from percentage denominator\n"
            "Balanced activation sample · all 27 alternative labels shown, including zeros",
            fontsize=12,
        )
        if n_wrong == 0:
            axes[1].text(
                0.5,
                0.5,
                "No valid wrong labels; percentages undefined",
                transform=axes[1].transAxes,
                ha="center",
            )
        fig.savefig(destination / f"{gold}.png", dpi=140)
        plt.show()
        plt.close(fig)
