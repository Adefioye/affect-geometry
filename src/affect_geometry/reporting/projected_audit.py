"""Publication figures for the projected audit, rendered from exported tidy CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from affect_geometry.logging import progress

DISTANCES = ["cosine", "crossnobis", "energy", "euclidean", "rbf_mmd", "sliced_wasserstein"]
TITLES = dict(
    zip(
        DISTANCES,
        [
            "Cosine centroids",
            "Crossnobis",
            "Energy",
            "Euclidean centroids",
            "RBF MMD",
            "Sliced Wasserstein",
        ],
        strict=True,
    )
)
GROUP_COLORS = {
    "positive": "#d64b40",
    "negative": "#3779bd",
    "ambiguous": "#c79e29",
    "neutral": "#434343",
}


def render_projected_audit(root: Path):
    tables, data, figures = root / "tables", root / "plot_data", root / "figures"
    data.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    jobs, registry = [], []

    def read(name):
        return pd.read_csv(tables / f"{name}.csv")

    def add(name, frame, title, func):
        jobs.append((name, frame, title, func))

    def grid(d, value="value", series="series", x="layer", ylabel="", reference=None, ylim=None):
        fig, axs = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
        for ax, distance in zip(axs.flat, DISTANCES, strict=True):
            for label, g in d[d.distance == distance].groupby(series, sort=False):
                g = g.sort_values(x)
                ax.plot(g[x], g[value], label=str(label), lw=1.6)
            if reference is not None:
                ax.axhline(reference, color="grey", ls=":", lw=1)
            ax.set(
                title=TITLES[distance],
                xlabel="Decoder block (zero based)" if x == "layer" else x.replace("_", " "),
                ylabel=ylabel,
            )
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(alpha=0.2)
        fig.legend(*axs.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
        return fig

    def tidy(frame, values, labels=None):
        result = frame.melt(
            id_vars=[c for c in frame if c not in values],
            value_vars=values,
            var_name="metric",
            value_name="value",
        )
        result["series"] = (
            result.method + " / " + result.metric.map(labels or {v: v for v in values})
        )
        return result

    def support(d):
        d = d.sort_values("sample_count")
        fig, axs = plt.subplots(1, 2, figsize=(14, 10), layout="constrained")
        axs[0].barh(d.emotion, d.sample_count, color="#3779bd", label="all samples")
        axs[0].barh(
            d.emotion, d.distribution_sample_count, color="#c79e29", label="distribution cap"
        )
        axs[0].set(xscale="log", xlabel="Comments (log scale)")
        axs[0].legend()
        axs[1].barh(d.emotion, d.mean_words, color="#519e8a")
        axs[1].set(xlabel="Mean whitespace words")
        return fig

    add("01_sample_support", read("class_balance"), "Sample support and comment length", support)

    def projection(d):
        fig, axs = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
        axs[0].step(d.layer, d.neutral_components_removed, where="mid")
        axs[0].set(ylabel="Directions removed", xlabel="Block")
        axs[1].plot(d.layer, d.neutral_variance_explained, marker="o")
        axs[1].axhline(0.5, color="grey", ls=":")
        axs[1].set(ylabel="Neutral variance removed", xlabel="Block", ylim=(0, 1.05))
        return fig

    add(
        "02_projection",
        read("projection"),
        "The 50% neutral-variance rule changes with depth",
        projection,
    )
    va, vad = read("va_recovery"), read("vad_audit")
    for name, frame, score, title in [
        ("03_va_recovery", va, "va_procrustes_r2", "2D valence–arousal recovery"),
        ("04_vad_recovery", vad, "vad_procrustes_r2", "3D valence–arousal–dominance recovery"),
    ]:
        add(
            name,
            tidy(
                frame,
                [score, "loo_joint_skill"],
                {score: "fitted R²", "loo_joint_skill": "LOO skill"},
            ),
            title + ": fitted shape and held-out-label alignment",
            lambda d: grid(d, ylabel="R² / skill", reference=0),
        )
    axes = read("axis_diagnostics")
    for dim, name in [(2, "05_va_axis_skill"), (3, "06_vad_axis_skill")]:
        df = axes[axes.dimensions == dim].copy()
        df["series"] = df.method + " / " + df.axis
        add(
            name,
            df,
            f"{dim}D joint similarity fit: skill for each held-out reference axis",
            lambda d: grid(d, value="loo_skill", ylabel="LOO skill vs training mean", reference=0),
        )

    selection = read("selected_examples")[["layer", "distance", "method", "criterion"]]
    exploration = read("exploration")
    maps = exploration[exploration.dimensions == 2].merge(
        selection, on=["layer", "distance", "method"], validate="many_to_one"
    )

    def maps2d(d):
        fig, axs = plt.subplots(4, 3, figsize=(20, 23), layout="constrained")
        for ax, (method, distance) in zip(
            axs.flat, [(m, dist) for m in ["mds", "isomap"] for dist in DISTANCES], strict=True
        ):
            g = d[(d.method == method) & (d.distance == distance)]
            x = g[["coordinate_1", "coordinate_2"]].to_numpy()
            span = np.maximum(np.ptp(x, axis=0), 1e-8)
            lo, hi = x.min(0) - span * 0.18, x.max(0) + span * 0.18
            gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 160), np.linspace(lo[1], hi[1], 160))
            classes = list(GROUP_COLORS)
            model = KNeighborsClassifier(n_neighbors=5).fit(x, g.emotion_group.map(classes.index))
            z = model.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
            from matplotlib.colors import ListedColormap

            ax.contourf(
                gx,
                gy,
                z,
                levels=np.arange(5) - 0.5,
                cmap=ListedColormap(list(GROUP_COLORS.values())),
                alpha=0.12,
            )
            for group, part in g.groupby("emotion_group"):
                ax.scatter(
                    part.coordinate_1, part.coordinate_2, c=GROUP_COLORS[group], s=32, label=group
                )
            for i, r in enumerate(g.itertuples()):
                ax.annotate(
                    r.emotion,
                    (r.coordinate_1, r.coordinate_2),
                    xytext=(3, 3 if i % 2 else -9),
                    textcoords="offset points",
                    fontsize=7,
                )
            ax.axhline(0, c="grey", lw=0.7)
            ax.axvline(0, c="grey", lw=0.7)
            ax.set(
                title=f"{TITLES[distance]} / {method} / block {int(g.layer.iloc[0])}",
                xlabel="Unaligned coordinate 1; neutral at zero",
                ylabel="Coordinate 2",
            )
            ax.set_aspect("equal", adjustable="datalim")
        fig.legend(*axs.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
        return fig

    add(
        "07_neutral_centered_maps",
        maps,
        "Selected VA maxima: taxonomy colors and 5NN shading, with unaligned axes",
        maps2d,
    )

    def rank1(d):
        fig, axs = plt.subplots(2, 3, figsize=(17, 9), layout="constrained")
        for ax, distance in zip(axs.flat, DISTANCES, strict=True):
            for j, method in enumerate(["mds", "isomap"]):
                g = d[(d.distance == distance) & (d.method == method)]
                scale = max(g.coordinate_1.abs().max(), 1e-12)
                for i, r in enumerate(g.itertuples()):
                    xx = r.coordinate_1 / scale
                    yy = j + 0.065 * (i % 4 - 1.5)
                    ax.scatter(xx, yy, c=GROUP_COLORS[r.emotion_group], s=22)
                    ax.annotate(
                        r.emotion,
                        (xx, yy),
                        xytext=(2, 4),
                        textcoords="offset points",
                        rotation=45,
                        fontsize=6,
                    )
            ax.set(
                title=TITLES[distance],
                yticks=[0, 1],
                yticklabels=["MDS", "Isomap"],
                xlabel="Coordinate 1 / maximum |coordinate| in panel row",
                ylim=(-0.45, 1.65),
            )
            ax.axvline(0, c="grey", ls=":", lw=0.7)
        return fig

    add(
        "08_rank1",
        exploration[(exploration.dimensions == 1) & (exploration.layer == 17)],
        "Rank 1 at fixed block 17; vertical jitter only aids labels",
        rank1,
    )

    def rank3(d):
        fig = plt.figure(figsize=(18, 11), layout="constrained")
        for i, distance in enumerate(DISTANCES):
            ax = fig.add_subplot(2, 3, i + 1, projection="3d")
            g = d[d.distance == distance]
            for r in g.itertuples():
                ax.scatter(
                    r.coordinate_1,
                    r.coordinate_2,
                    r.coordinate_3,
                    c=GROUP_COLORS[r.emotion_group],
                    s=24,
                )
                ax.text(r.coordinate_1, r.coordinate_2, r.coordinate_3, r.emotion, fontsize=6)
            ax.set(
                title=TITLES[distance],
                xlabel="Coordinate 1",
                ylabel="Coordinate 2",
                zlabel="Coordinate 3",
            )
        return fig

    add(
        "09_rank3",
        exploration[
            (exploration.dimensions == 3)
            & (exploration.layer == 17)
            & (exploration.method == "mds")
        ],
        "3D MDS at fixed block 17; taxonomy colors and neutral origin",
        rank3,
    )
    residuals = read("emotion_residuals")
    aligned = residuals[residuals.dimensions == 2].merge(
        selection, on=["layer", "distance", "method"], validate="many_to_one"
    )

    def alignment(d):
        fig, axs = plt.subplots(4, 3, figsize=(19, 22), layout="constrained")
        for ax, (method, distance) in zip(
            axs.flat, [(m, dist) for m in ["mds", "isomap"] for dist in DISTANCES], strict=True
        ):
            g = d[(d.method == method) & (d.distance == distance)]
            ax.scatter(g.valence, g.arousal, c="black", marker="x", s=25, label="NRC target")
            ax.scatter(
                g.aligned_valence,
                g.aligned_arousal,
                c=g.emotion_group.map(GROUP_COLORS),
                s=26,
                label="aligned embedding",
            )
            for i, r in enumerate(g.itertuples()):
                ax.plot(
                    [r.valence, r.aligned_valence],
                    [r.arousal, r.aligned_arousal],
                    c="grey",
                    lw=0.6,
                    alpha=0.5,
                )
                ax.annotate(
                    r.emotion,
                    (r.aligned_valence, r.aligned_arousal),
                    xytext=(2, 3 if i % 2 else -8),
                    textcoords="offset points",
                    fontsize=6,
                )
            ax.set(
                title=f"{TITLES[distance]} / {method} / block {int(g.layer.iloc[0])}",
                xlabel="Valence",
                ylabel="Arousal",
                xlim=(-1.2, 1.2),
                ylim=(-1, 1.2),
            )
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
        fig.legend(*axs.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
        return fig

    add(
        "10_aligned_reference",
        aligned,
        "Separate reference evaluation: shared rotation, reflection, translation and scale",
        alignment,
    )
    quality = read("quality")
    for i, metric in enumerate(["stress", "trustworthiness", "shepard_spearman", "continuity"], 11):
        df = quality.copy()
        df["series"] = df.method + " / " + df.dimensions.astype(str) + "D"
        add(
            f"{i:02d}_{metric}",
            df,
            f"Embedding quality against direct RDM: {metric}",
            lambda d, m=metric: grid(d, value=m, ylabel=m.replace("_", " ")),
        )
    intrinsic = read("intrinsic")
    add(
        "15_dimensions",
        tidy(intrinsic, ["participation_ratio", "dimensions_90"]),
        "Emotion-level spectrum: participation ratio and dimensions for 90% positive mass",
        lambda d: grid(d, ylabel="Dimension / participation ratio"),
    )
    spectrum = read("spectrum")
    spectrum = (
        spectrum[spectrum.component <= 10]
        .groupby(
            [
                "distance",
                "method",
                "component",
                "model",
                "run_id",
                "variant",
                "source_run",
                "post_projection_pca",
                "neutral_variance_threshold",
            ],
            as_index=False,
        )[["positive_mass", "cumulative_positive_mass"]]
        .mean()
    )
    add(
        "16_spectrum",
        tidy(spectrum, ["positive_mass", "cumulative_positive_mass"]),
        "First ten modes: spectral mass averaged across all 28 blocks",
        lambda d: grid(d, x="component", ylabel="Positive mass fraction", ylim=(0, 1)),
    )
    gaps = read("eigengaps")
    gaps["series"] = "gap " + gaps.gap_rank.astype(str)
    add(
        "17_eigengaps",
        gaps,
        "Off-diagonal shuffle eigengap tests; BH over 840 tests",
        lambda d: grid(d, value="bh_q", ylabel="BH q value", reference=0.05, ylim=(0, 1)),
    )
    comparison = quality.pivot(
        index=["layer", "distance", "dimensions"],
        columns="method",
        values=["trustworthiness", "stress"],
    ).reset_index()
    comparison.columns = ["_".join(c).rstrip("_") for c in comparison.columns]
    for metric in ["trustworthiness", "stress"]:
        comparison[metric] = comparison[metric + "_isomap"] - comparison[metric + "_mds"]
    ctx = [
        "model",
        "run_id",
        "variant",
        "source_run",
        "neutral_variance_threshold",
        "post_projection_pca",
    ]
    comparison = comparison.merge(
        quality[["layer", "distance", "dimensions"] + ctx].drop_duplicates(),
        on=["layer", "distance", "dimensions"],
    )
    df = comparison.melt(
        id_vars=[c for c in comparison if c not in ["trustworthiness", "stress"]],
        value_vars=["trustworthiness", "stress"],
        var_name="metric",
        value_name="value",
    )
    df["series"] = df.metric + " / " + df.dimensions.astype(str) + "D"
    add(
        "18_isomap_changes",
        df,
        "Isomap minus MDS: trustworthiness higher, stress lower is better",
        lambda d: grid(d, ylabel="Quality difference", reference=0),
    )
    geo = read("geodesic_pairs")
    geo = (
        geo.groupby(["distance", "layer"] + ctx)
        .ratio.quantile([0.1, 0.5, 0.9])
        .rename("ratio")
        .reset_index()
        .rename(columns={f"level_{2 + len(ctx)}": "quantile"})
    )
    geo["series"] = geo["quantile"].map(
        {0.1: "10th percentile", 0.5: "median", 0.9: "90th percentile"}
    )
    add(
        "19_geodesic_ratios",
        geo,
        "Geodesic / direct ratios: zero direct distances omitted",
        lambda d: grid(d, value="ratio", ylabel="Distance ratio", reference=1),
    )

    def agreement(d):
        fig, axs = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
        for ax, value, label in zip(
            axs,
            ["spearman", "knn_overlap", "procrustes_r2"],
            ["Direct RDM Spearman", "5-neighbor overlap", "MDS shape R²"],
            strict=True,
        ):
            matrix = np.eye(6)
            for r in d.itertuples():
                i, j = DISTANCES.index(r.distance_i), DISTANCES.index(r.distance_j)
                matrix[i, j] = matrix[j, i] = getattr(r, value)
            im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
            for (i, j), v in np.ndenumerate(matrix):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    color="white" if v < 0.5 else "black",
                )
            ax.set(
                title=label,
                xticks=range(6),
                yticks=range(6),
                xticklabels=["Cos", "Cross", "Energy", "Euc", "MMD", "SW"],
                yticklabels=["Cos", "Cross", "Energy", "Euc", "MMD", "SW"],
            )
            fig.colorbar(im, ax=ax, shrink=0.7)
        return fig

    ag = read("distance_agreement")
    ag = (
        ag[ag.method == "mds"]
        .groupby(["distance_i", "distance_j"] + ctx, as_index=False)[
            ["spearman", "knn_overlap", "procrustes_r2"]
        ]
        .mean()
    )
    add("20_distance_agreement", ag, "Distance agreement averaged over matched layers", agreement)

    def crossheat(d):
        fig, axs = plt.subplots(2, 6, figsize=(24, 11), layout="constrained")
        for row, method in enumerate(["mds", "isomap"]):
            for col, distance in enumerate(DISTANCES):
                matrix = np.eye(28)
                for r in d[(d.method == method) & (d.distance == distance)].itertuples():
                    matrix[r.layer_i, r.layer_j] = matrix[r.layer_j, r.layer_i] = r.procrustes_r2
                ax = axs[row, col]
                im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
                ax.set(
                    title=f"{TITLES[distance]}\n{method}",
                    xlabel="Block",
                    ylabel="Block",
                    xticks=[0, 7, 14, 21, 27],
                    yticks=[0, 7, 14, 21, 27],
                )
        fig.colorbar(im, ax=axs, shrink=0.6, label="2D Procrustes R²")
        return fig

    add(
        "21_cross_layer",
        read("cross_layer"),
        "Evolution of emotion geometry through depth",
        crossheat,
    )
    sweep = read("neighbor_sweep")
    df = sweep.groupby(["distance", "requested_neighbors"] + ctx, as_index=False)[
        ["va_r2", "trustworthiness", "stress"]
    ].mean()
    df["method"] = "isomap"
    add(
        "22_graph_sensitivity",
        tidy(df, ["va_r2", "trustworthiness", "stress"]),
        "Isomap graph sensitivity: mean across 28 blocks; evaluation neighborhoods fixed at 5",
        lambda d: grid(d, x="requested_neighbors", ylabel="Score / stress"),
    )
    rsa = read("reference_rsa")
    df = rsa.melt(
        id_vars=[c for c in rsa if c not in ["spearman", "partial_spearman"]],
        value_vars=["spearman", "partial_spearman"],
        var_name="control",
        value_name="value",
    )
    df["series"] = df.reference_axis + " / " + df.control
    add(
        "23_reference_rsa",
        df,
        "Reference RDM alignment: raw and nuisance-adjusted rank correlation",
        lambda d: grid(d, ylabel="Spearman / partial Spearman", reference=0),
    )

    def samplechecks(d):
        fig, axs = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
        for col, distance in enumerate(["cosine", "euclidean"]):
            g = d[(d.distance == distance) & (d.check == "balanced")]
            p = g.pivot(index="layer", columns="quantile", values="va_r2")
            axs[0, col].fill_between(
                p.index, p[0.025], p[0.975], alpha=0.2, label="2.5–97.5% of balanced draws"
            )
            axs[0, col].plot(p.index, p[0.5], marker="o", label="balanced median")
            full = g.drop_duplicates("layer")
            axs[0, col].plot(full.layer, full.full_sample_r2, marker="s", label="all samples")
            for split, s in d[(d.distance == distance) & (d.check == "split")].groupby("split"):
                axs[1, col].plot(s.layer, s.va_r2, marker="o", label=split)
            axs[0, col].set(title=TITLES[distance], ylabel="2D MDS VA R²", ylim=(0, 0.8))
            axs[1, col].set(xlabel="Block", ylabel="2D MDS VA R²", ylim=(0, 0.8))
            axs[0, col].legend(fontsize=8)
            axs[1, col].legend(fontsize=8)
        return fig

    balanced = read("balanced_draws")
    quant = (
        balanced.groupby(["layer", "distance"] + ctx)
        .va_r2.quantile([0.025, 0.5, 0.975])
        .rename("va_r2")
        .reset_index()
        .rename(columns={f"level_{2 + len(ctx)}": "quantile"})
    )
    quant = quant.merge(
        va[va.method == "mds"][["layer", "distance", "va_procrustes_r2"]].rename(
            columns={"va_procrustes_r2": "full_sample_r2"}
        ),
        on=["layer", "distance"],
    )
    quant["check"] = "balanced"
    split = read("frozen_split_sensitivity").assign(check="split")
    add(
        "24_sample_sensitivity",
        pd.concat([quant, split], ignore_index=True),
        "Sample sensitivity conditional on the frozen all-split projection",
        samplechecks,
    )

    def nonmetric(d):
        fig, axs = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
        for ax, metric in zip(
            axs, ["triangle_violation_fraction", "negative_eigenvalue_fraction"], strict=True
        ):
            for distance, g in d.groupby("distance"):
                ax.plot(g.layer, g[metric], label=TITLES[distance])
            ax.set(xlabel="Block", ylabel=metric.replace("_", " "))
        fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
        return fig

    audit = read("metric_audit").merge(
        intrinsic[intrinsic.method == "mds"][["layer", "distance", "negative_eigenvalue_fraction"]],
        on=["layer", "distance"],
    )
    add(
        "25_metric_assumptions",
        audit,
        "Non-Euclidean diagnostics of the input dissimilarities",
        nonmetric,
    )

    def chord(d):
        fig, axs = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
        for (transform, method), g in d.groupby(["transform", "method"]):
            axs[0].plot(g.layer, g.va_r2, label=transform + " / " + method)
            axs[1].plot(g.layer, g.negative_eigenvalue_fraction)
        axs[0].set(xlabel="Block", ylabel="VA R²")
        axs[1].set(xlabel="Block", ylabel="Negative spectral mass fraction")
        fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
        return fig

    add(
        "26_cosine_chord",
        read("cosine_metric_sensitivity"),
        "Same normalized centroids: cosine dissimilarity versus Euclidean chord distance",
        chord,
    )

    def heatneigh(d):
        matrix = d.pivot(index="emotion", columns="layer", values="neighborhood_change")
        matrix = matrix.loc[matrix.mean(1).sort_values(ascending=False).index]
        fig, ax = plt.subplots(figsize=(15, 10), layout="constrained")
        im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="magma")
        ax.set(
            yticks=range(28),
            yticklabels=matrix.index,
            xticks=range(28),
            xticklabels=range(28),
            xlabel="Block",
        )
        fig.colorbar(im, ax=ax, label="Mean 1 − 5-neighbor overlap over 15 distance pairs")
        return fig

    neigh = (
        read("neighborhood_change")
        .groupby(["emotion", "layer"] + ctx, as_index=False)["neighborhood_change"]
        .mean()
    )
    add(
        "27_neighborhood_instability",
        neigh,
        "Emotion-specific sensitivity to distance choice",
        heatneigh,
    )
    df = read("taxonomy_structure").assign(series="positive vs negative")
    add(
        "28_taxonomy",
        df,
        "Positive/negative 5NN purity in direct RDMs; only 23 signed labels",
        lambda d: grid(
            d, value="positive_negative_5nn_purity", ylabel="Neighbor purity", ylim=(0, 1)
        ),
    )
    df = read("residual_variance")
    df = df[df.layer.isin([0, 7, 14, 17, 21, 27])].copy()
    df["series"] = "block " + df.layer.astype(str)
    add(
        "29_isomap_residuals",
        df,
        "Isomap geodesic residual variance at fixed depths",
        lambda d: grid(
            d, value="residual_variance", x="dimensions", ylabel="1 − Pearson²", ylim=(0, 1)
        ),
    )

    def shepard(d):
        fig, axs = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
        for ax, distance in zip(axs.flat, DISTANCES, strict=True):
            g = d[d.distance == distance]
            scale = g.direct_distance.max()
            for method, part in g.groupby("method"):
                ax.scatter(
                    part.direct_distance / scale,
                    part.embedded_distance / scale,
                    s=8,
                    alpha=0.3,
                    label=method,
                )
            ax.plot([0, 1], [0, 1], c="grey", ls=":")
            ax.set(title=TITLES[distance], xlabel="Direct / max direct", ylabel="2D / max direct")
        fig.legend(*axs.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
        return fig

    add(
        "30_shepard",
        read("shepard_pairs").query("layer == 14"),
        "All 378 emotion pairs at fixed block 14",
        shepard,
    )
    for name, frame, title, function in progress(
        jobs, description="Rendering projected audit", unit="figure"
    ):
        path = data / f"{name}.csv"
        frame.to_csv(path, index=False)
        # Every plot consumes its persisted, self-contained plot-ready table.
        fig = function(pd.read_csv(path))
        fig.suptitle(title, fontsize=14)
        fig.savefig(figures / f"{name}.png", dpi=160, bbox_inches="tight")
        fig.savefig(figures / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        registry.append(
            dict(
                name=name,
                title=title,
                csv=str(path.relative_to(root)),
                png=f"figures/{name}.png",
                pdf=f"figures/{name}.pdf",
                rows=len(frame),
            )
        )
    pd.DataFrame(registry).to_csv(root / "figure_index.csv", index=False)
    (root / "figure_index.json").write_text(json.dumps(registry, indent=2) + "\n")
