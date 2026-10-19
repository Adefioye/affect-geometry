"""Static report figures, each rendered solely from its own tidy CSV."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from affect_geometry.logging import progress

DISTANCES = ["cosine", "crossnobis", "energy", "euclidean", "rbf_mmd", "sliced_wasserstein"]
TITLES = {
    "cosine": "Cosine (centered PCA)",
    "crossnobis": "Crossnobis",
    "energy": "Energy",
    "euclidean": "Euclidean centroids",
    "rbf_mmd": "RBF MMD",
    "sliced_wasserstein": "Sliced Wasserstein",
}
METHODS = {"mds": "#20639b", "isomap": "#d95f02"}


def render_report_figures(report: Path):
    source = report / "tables"
    plot_data = report / "plot_data"
    plot_data.mkdir(parents=True, exist_ok=True)
    figures = report / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    registry = []

    def read(name):
        return pd.read_csv(source / f"{name}.csv")

    def draw(name, frame, title, function):
        csv = plot_data / f"figure_{name}.csv"
        frame.to_csv(csv, index=False)
        # Deliberately reload the exported plot-ready artifact.
        fig = function(pd.read_csv(csv))
        fig.suptitle(title, fontsize=15, y=1.04)
        fig.savefig(figures / f"{name}.png", dpi=170, bbox_inches="tight")
        fig.savefig(figures / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        registry.append(
            dict(
                figure=f"figures/{name}.png",
                vector_figure=f"figures/{name}.pdf",
                data=f"plot_data/figure_{name}.csv",
                title=title,
            )
        )

    def lines(frame, value, ylabel, *, x="layer", series="method", ylim=None, reference=None):
        fig, axs = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
        for ax, distance in zip(axs.flat, DISTANCES):
            part = frame[frame.distance == distance]
            for i, (label, g) in enumerate(part.groupby(series, sort=False)):
                g = g.sort_values(x)
                ax.plot(g[x], g[value], label=str(label), lw=1.7, color=METHODS.get(label, None))
            if reference is not None:
                ax.axhline(reference, color="grey", lw=0.8, ls=":")
            ax.set(
                title=TITLES[distance],
                xlabel="Decoder block (zero based)" if x == "layer" else x.replace("_", " "),
                ylabel=ylabel,
            )
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(alpha=0.2)
        handles, labels = axs.flat[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="outside lower center", ncol=min(6, len(labels)), frameon=False
        )
        return fig

    va = read("va_recovery")
    quality = read("quality")
    intrinsic = read("intrinsic")
    context = [
        "model",
        "run_id",
        "pooling",
        "split",
        "seed",
        "pca_components",
        "distribution_cap",
        "quality_neighbors",
    ]
    jobs = []
    recovery = va.melt(
        id_vars=[c for c in va.columns if c not in ["va_procrustes_r2", "loo_joint_skill"]],
        value_vars=["va_procrustes_r2", "loo_joint_skill"],
        var_name="score_type",
        value_name="score",
    )
    recovery["series"] = (
        recovery.method
        + " / "
        + recovery.score_type.map(
            {"va_procrustes_r2": "fitted", "loo_joint_skill": "held-out label"}
        )
    )
    jobs.append(
        (
            "01_va_recovery",
            recovery,
            "V–A shape recovery across all layers",
            lambda d: lines(
                d, "score", "Joint R² / predictive skill", series="series", reference=0
            ),
        )
    )
    axis = va.melt(
        id_vars=["layer", "distance", "method"] + context,
        value_vars=["valence_pearson", "arousal_pearson"],
        var_name="axis",
        value_name="pearson",
    )
    axis["series"] = axis.method + " / " + axis.axis.str.replace("_pearson", "")
    jobs.append(
        (
            "02_axis_correlations",
            axis,
            "Valence and arousal correlations after joint similarity alignment",
            lambda d: lines(d, "pearson", "Aligned Pearson r", series="series", ylim=(0, 1)),
        )
    )
    selected = read("selected_examples")
    aligned = read("va_alignment").merge(
        selected[["distance", "method", "layer", "criterion"]], on=["distance", "method", "layer"]
    )

    def maps(d):
        fig, axs = plt.subplots(4, 3, figsize=(20, 23), layout="constrained")
        for index, (distance, method) in enumerate(
            (distance, method) for method in ["mds", "isomap"] for distance in DISTANCES
        ):
            ax = axs.flat[index]
            g = d[(d.distance == distance) & (d.method == method)]
            for row in g.itertuples():
                ax.plot(
                    [row.valence, row.aligned_valence],
                    [row.arousal, row.aligned_arousal],
                    color="#999999",
                    alpha=0.45,
                    lw=0.8,
                )
            ax.scatter(g.valence, g.arousal, s=24, marker="x", c="black", label="NRC target")
            ax.scatter(
                g.aligned_valence,
                g.aligned_arousal,
                c=g.valence,
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                s=38,
                edgecolor="white",
                lw=0.4,
                label="aligned embedding",
            )
            # Deterministic stagger, labels name both ends through the connecting segment.
            for i, r in enumerate(g.itertuples()):
                ax.annotate(
                    r.emotion,
                    (r.aligned_valence, r.aligned_arousal),
                    xytext=(3, 3 if i % 2 else -8),
                    textcoords="offset points",
                    fontsize=7,
                )
            ax.set(
                title=f"{TITLES[distance]} / {method} / block {int(g.layer.iloc[0])}",
                xlabel="Valence",
                ylabel="Arousal",
                xlim=(
                    min(-1.1, g.aligned_valence.min() - 0.2),
                    max(1.1, g.aligned_valence.max() + 0.2),
                ),
                ylim=(
                    min(-0.9, g.aligned_arousal.min() - 0.2),
                    max(1.0, g.aligned_arousal.max() + 0.2),
                ),
            )
            ax.set_aspect("equal")
            ax.grid(alpha=0.15)
        fig.legend(*axs.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
        return fig

    jobs.append(
        (
            "03_aligned_maps",
            aligned,
            "Selected maxima of fitted V–A R² (exploratory layer selection)",
            maps,
        )
    )
    for name, label in [
        ("stress", "Normalized metric stress ↓"),
        ("trustworthiness", "Trustworthiness at k=5 ↑"),
        ("shepard_spearman", "Shepard Spearman ρ ↑"),
        ("continuity", "Continuity at k=5 ↑"),
    ]:
        jobs.append(
            (
                f"04_{name}",
                quality[quality.dimensions == 2],
                f"2D embedding quality against the direct RDM: {label}",
                lambda d, n=name, axis_label=label: lines(d, n, axis_label),
            )
        )
    dimension = intrinsic[intrinsic.method == "mds"].melt(
        id_vars=["layer", "distance"] + context,
        value_vars=["dimensions_80", "dimensions_90", "dimensions_95", "participation_ratio"],
        var_name="estimate",
        value_name="dimension",
    )
    elbow = intrinsic[intrinsic.method == "isomap"].copy()
    elbow["estimate"] = "Isomap residual elbow (1–10)"
    elbow["dimension"] = elbow.isomap_residual_elbow
    dimension = pd.concat([dimension, elbow], ignore_index=True)
    jobs.append(
        (
            "05_intrinsic_dimensions",
            dimension,
            "Dimensions of emotion-level geometry: positive MDS spectrum and Isomap elbow",
            lambda d: lines(
                d, "dimension", "Dimension / participation ratio", series="estimate", ylim=(0, 27)
            ),
        )
    )
    spectrum = read("spectrum")
    for value, title in [
        ("positive_mass", "Positive classical-MDS eigenvalue spectrum"),
        ("cumulative_positive_mass", "Cumulative positive spectral mass"),
    ]:
        summary = spectrum.groupby(["distance", "method", "component"] + context, as_index=False)[
            value
        ].mean()
        jobs.append(
            (
                f"06_{value}",
                summary,
                f"{title}: average over 28 blocks",
                lambda d, v=value: lines(
                    d, v, "Fraction of positive mass", x="component", ylim=(0, 1)
                ),
            )
        )
    res = read("residual_variance")
    res = res[res.layer.isin([0, 7, 14, 21, 27])].copy()
    res["series"] = "block " + res.layer.astype(str)
    jobs.append(
        (
            "07_isomap_residuals",
            res,
            "Isomap residual variance at fixed representative depths",
            lambda d: lines(
                d,
                "residual_variance",
                "1 − Pearson² to geodesic distances",
                x="dimensions",
                series="series",
                ylim=(0, 1),
            ),
        )
    )
    q = quality.pivot(
        index=["layer", "distance", "dimensions"] + context,
        columns="method",
        values=["stress", "trustworthiness", "shepard_spearman", "continuity"],
    ).reset_index()
    q.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in q.columns]
    for metric in ["stress", "trustworthiness", "shepard_spearman", "continuity"]:
        q[metric + "_difference"] = q[metric + "_isomap"] - q[metric + "_mds"]
    deltas = q[q.dimensions == 2].melt(
        id_vars=["layer", "distance"] + context,
        value_vars=[
            m + "_difference"
            for m in ["stress", "trustworthiness", "shepard_spearman", "continuity"]
        ],
        var_name="metric",
        value_name="difference",
    )
    jobs.append(
        (
            "08_mds_isomap_difference",
            deltas,
            "Isomap minus classical MDS in 2D (negative stress difference favors Isomap)",
            lambda d: lines(d, "difference", "Isomap − MDS", series="metric", reference=0),
        )
    )
    dimq = q.groupby(["distance", "dimensions"] + context, as_index=False)[
        ["stress_difference", "trustworthiness_difference"]
    ].mean()
    dimq = dimq.melt(
        id_vars=["distance", "dimensions"] + context, var_name="metric", value_name="difference"
    )
    jobs.append(
        (
            "09_dimension_method_difference",
            dimq,
            "Mean Isomap − MDS quality difference by embedding dimension",
            lambda d: lines(
                d,
                "difference",
                "Mean difference across blocks",
                x="dimensions",
                series="metric",
                reference=0,
            ),
        )
    )
    geo = read("geodesic_pairs")
    quantiles = (
        geo.groupby(["distance", "layer"] + context)
        .ratio.quantile([0.05, 0.5, 0.95])
        .rename("ratio")
        .reset_index()
    )
    quantiles = quantiles.rename(
        columns={next(c for c in quantiles if c.startswith("level_")): "quantile"}
    )
    quantiles["series"] = quantiles["quantile"].map(
        {0.05: "5th percentile", 0.5: "median", 0.95: "95th percentile"}
    )
    jobs.append(
        (
            "10_geodesic_ratios",
            quantiles,
            "Geodesic / direct distances (zero direct distances omitted)",
            lambda d: lines(d, "ratio", "Distance ratio", series="series", reference=1),
        )
    )
    agreement = read("distance_agreement")
    ag = agreement.groupby(["distance_i", "distance_j", "method"] + context, as_index=False)[
        ["spearman", "procrustes_r2", "knn_overlap"]
    ].mean()

    def agreement_heat(d):
        fig, axs = plt.subplots(2, 2, figsize=(14, 12), layout="constrained")
        panels = [
            ("spearman", "mds", "RDM Spearman ρ"),
            ("knn_overlap", "mds", "5-neighbor overlap"),
            ("procrustes_r2", "mds", "MDS shape R²"),
            ("procrustes_r2", "isomap", "Isomap shape R²"),
        ]
        short = ["cos", "cross", "energy", "euc", "MMD", "SW"]
        for ax, (value, method, title) in zip(axs.flat, panels):
            matrix = np.eye(6)
            for row in d[d.method == method].itertuples():
                i, j = DISTANCES.index(row.distance_i), DISTANCES.index(row.distance_j)
                matrix[i, j] = matrix[j, i] = getattr(row, value)
            im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
            for (i, j), v in np.ndenumerate(matrix):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    color="white" if v < 0.55 else "black",
                    fontsize=10,
                )
            ax.set(
                xticks=range(6), yticks=range(6), xticklabels=short, yticklabels=short, title=title
            )
            fig.colorbar(im, ax=ax, shrink=0.8)
        return fig

    jobs.append(
        (
            "11_distance_agreement",
            ag,
            "Agreement across distances: average of matched blocks",
            agreement_heat,
        )
    )
    neighborhood = read("neighborhood_change")
    neigh = neighborhood.groupby(
        ["emotion", "layer"] + context, as_index=False
    ).neighborhood_change.mean()

    def neighborhood_heat(d):
        matrix = d.pivot(index="emotion", columns="layer", values="neighborhood_change")
        matrix = matrix.loc[matrix.mean(1).sort_values(ascending=False).index]
        fig, ax = plt.subplots(figsize=(15, 10), layout="constrained")
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="magma", aspect="auto")
        ax.set(
            xticks=range(28),
            xticklabels=range(28),
            yticks=range(28),
            yticklabels=matrix.index,
            xlabel="Decoder block",
            ylabel="Emotion (ordered by mean change)",
        )
        fig.colorbar(im, ax=ax, label="1 − 5-neighbor overlap, mean over 15 distance pairs")
        return fig

    jobs.append(
        (
            "12_neighborhood_change",
            neigh,
            "Which emotions depend most on the distance definition?",
            neighborhood_heat,
        )
    )
    cross = read("cross_layer")

    def cross_heat(d):
        fig, axs = plt.subplots(2, 6, figsize=(23, 8), layout="constrained")
        for row, method in enumerate(["mds", "isomap"]):
            for col, distance in enumerate(DISTANCES):
                g = d[(d.distance == distance) & (d.method == method)]
                mat = np.eye(28)
                for r in g.itertuples():
                    mat[r.layer_i, r.layer_j] = mat[r.layer_j, r.layer_i] = r.procrustes_r2
                ax = axs[row, col]
                im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
                ax.set(
                    title=f"{TITLES[distance]}\n{method}",
                    xticks=[0, 7, 14, 21, 27],
                    yticks=[0, 7, 14, 21, 27],
                    xlabel="Block",
                    ylabel="Block",
                )
        fig.colorbar(im, ax=axs, shrink=0.7, label="Cross-block 2D Procrustes R²")
        return fig

    jobs.append(
        (
            "13_cross_layer_similarity",
            cross,
            "Evolution of 2D emotion geometry across model depth",
            cross_heat,
        )
    )
    sweep = read("neighbor_sweep")
    sw = sweep.groupby(["distance", "requested_neighbors"] + context, as_index=False)[
        ["stress", "trustworthiness", "va_r2", "geodesic_ratio_median"]
    ].mean()

    def sweep_plot(d):
        fig, axs = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
        for ax, distance in zip(axs.flat, DISTANCES):
            g = d[d.distance == distance]
            ax.plot(g.requested_neighbors, g.va_r2, label="V–A R²", marker="o")
            ax.plot(g.requested_neighbors, g.trustworthiness, label="trustworthiness", marker="s")
            ax.plot(g.requested_neighbors, g.stress, label="stress", marker="^")
            ax.set(
                title=TITLES[distance],
                xlabel="Requested graph neighbors k",
                ylabel="Mean score over 28 blocks",
                ylim=(0, 1),
            )
            ax.grid(alpha=0.2)
        fig.legend(*axs.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
        return fig

    jobs.append(
        (
            "14_graph_sensitivity",
            sw,
            "Isomap graph sensitivity (quality evaluated at k=5)",
            sweep_plot,
        )
    )
    rsa = read("reference_rsa")
    rdf = rsa.melt(
        id_vars=["layer", "distance", "reference_axis"] + context,
        value_vars=["spearman", "partial_spearman"],
        var_name="control",
        value_name="rho",
    )
    rdf["series"] = (
        rdf.reference_axis
        + " / "
        + rdf.control.map({"spearman": "raw", "partial_spearman": "partial"})
    )
    jobs.append(
        (
            "15_reference_rsa",
            rdf,
            "Reference-distance agreement before and after nuisance controls",
            lambda d: lines(d, "rho", "Rank correlation", series="series", reference=0),
        )
    )

    def sampling_plot(d):
        fig, axs = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
        for col, distance in enumerate(["cosine", "euclidean"]):
            for split, g in d[(d.distance == distance) & (d.method == "mds")].groupby("split"):
                axs[0, col].plot(g.layer, g.va_r2, marker="o", label=split)
                axs[1, col].plot(g.layer, g.rdm_spearman_to_train, marker="o", label=split)
            axs[0, col].set(title=TITLES[distance], ylabel="2D MDS V–A R²", ylim=(0, 1))
            axs[1, col].set(ylabel="RDM Spearman ρ to train", xlabel="Block", ylim=(0, 1))
        fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
        return fig

    jobs.append(
        (
            "16_split_sensitivity",
            read("split_sensitivity"),
            "Disjoint text splits using PCA fitted on training text only",
            sampling_plot,
        )
    )
    balance = read("balanced_sensitivity")
    quant = (
        balance.groupby(["distance", "layer"] + [c for c in context if c in balance])
        .va_r2.quantile([0.025, 0.5, 0.975])
        .rename("va_r2")
        .reset_index()
    )
    quant = quant.rename(columns={next(c for c in quant if c.startswith("level_")): "quantile"})
    orig = va[
        (va.method == "mds")
        & va.layer.isin([0, 7, 14, 21, 27])
        & va.distance.isin(["cosine", "euclidean"])
    ][["distance", "layer", "va_procrustes_r2"]].rename(
        columns={"va_procrustes_r2": "full_sample_r2"}
    )
    quant = quant.merge(orig, on=["distance", "layer"])

    def balance_plot(d):
        fig, axs = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
        for ax, distance in zip(axs, ["cosine", "euclidean"]):
            g = d[d.distance == distance].pivot(index="layer", columns="quantile", values="va_r2")
            ax.fill_between(
                g.index, g[0.025], g[0.975], alpha=0.2, label="2.5–97.5% across balanced draws"
            )
            ax.plot(g.index, g[0.5], marker="o", label="balanced median")
            full = d[d.distance == distance].drop_duplicates("layer")
            ax.plot(full.layer, full.full_sample_r2, marker="s", label="original full centroids")
            ax.set(title=TITLES[distance], xlabel="Block", ylabel="2D MDS V–A R²", ylim=(0, 0.8))
        fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
        return fig

    jobs.append(
        (
            "17_balanced_sampling",
            quant,
            "Equal sample counts: 200 draws of 47 comments per emotion",
            balance_plot,
        )
    )
    audit = read("metric_audit").merge(
        intrinsic[intrinsic.method == "mds"][["distance", "layer", "negative_eigenvalue_fraction"]],
        on=["distance", "layer"],
    )
    aud = audit.melt(
        id_vars=["distance", "layer"] + context,
        value_vars=["triangle_violation_fraction", "negative_eigenvalue_fraction"],
        var_name="diagnostic",
        value_name="fraction",
    )
    jobs.append(
        (
            "18_non_euclidean_audit",
            aud,
            "Metric assumptions: triangle violations and negative MDS spectral mass",
            lambda d: lines(d, "fraction", "Fraction", series="diagnostic", ylim=(0, 0.3)),
        )
    )

    def class_plot(d):
        d = d.sort_values("sample_count")
        fig, axs = plt.subplots(1, 2, figsize=(15, 10), layout="constrained")
        axs[0].barh(d.emotion, d.sample_count, color="#20639b", label="all single-label examples")
        axs[0].barh(
            d.emotion,
            d.distribution_sample_count,
            color="#e8a83e",
            label="distribution-distance subsample",
        )
        axs[0].set(xlabel="Comments (log scale)", xscale="log")
        axs[0].legend(fontsize=8)
        axs[1].barh(d.emotion, d.mean_words, color="#519e8a")
        axs[1].set(xlabel="Mean whitespace word count")
        return fig

    jobs.append(
        (
            "19_dataset_balance",
            read("class_balance"),
            "Sample support and comment length by emotion",
            class_plot,
        )
    )
    shepard = read("shepard_pairs")
    shepard = shepard[shepard.layer == 14]

    def shepard_plot(d):
        fig, axs = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
        for ax, distance in zip(axs.flat, DISTANCES):
            g = d[d.distance == distance]
            maximum = g.direct_distance.max()
            for method, part in g.groupby("method"):
                ax.scatter(
                    part.direct_distance / maximum,
                    part.embedded_distance / maximum,
                    s=9,
                    alpha=0.4,
                    label=method,
                    color=METHODS[method],
                )
            ax.plot([0, 1], [0, 1], c="grey", ls=":")
            ax.set(
                title=TITLES[distance],
                xlabel="Direct RDM / maximum direct",
                ylabel="2D distance / maximum direct",
            )
        fig.legend(*axs.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
        return fig

    jobs.append(
        (
            "20_shepard_diagrams",
            shepard,
            "Shepard diagrams at fixed block 14: all 378 emotion pairs",
            shepard_plot,
        )
    )

    def preprocessing_plot(d):
        fig, axs = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
        for col, distance in enumerate(["cosine", "euclidean"]):
            for transform, group in d[d.distance == distance].groupby("transform"):
                axs[0, col].plot(group.layer, group.va_r2, marker="o", label=transform)
                axs[1, col].plot(
                    group.layer, group.rdm_spearman_to_original, marker="o", label=transform
                )
            axs[0, col].set(title=TITLES[distance], ylabel="2D MDS V–A R²", ylim=(0, 0.65))
            axs[1, col].set(
                xlabel="Block", ylabel="RDM Spearman ρ to pooled PCA result", ylim=(0, 1.05)
            )
        fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
        return fig

    jobs.append(
        (
            "21_preprocessing_sensitivity",
            read("preprocessing_sensitivity"),
            "Cosine depends strongly on centering and PCA; Euclidean geometry is more stable",
            preprocessing_plot,
        )
    )
    skill = va.melt(
        id_vars=["layer", "distance", "method"] + context,
        value_vars=["loo_valence_skill", "loo_arousal_skill"],
        var_name="axis",
        value_name="predictive_skill",
    )
    skill["series"] = (
        skill.method + " / " + skill.axis.str.replace("loo_", "").str.replace("_skill", "")
    )
    jobs.append(
        (
            "22_held_out_axes",
            skill,
            "Held-out emotion predictions relative to training-mean baselines",
            lambda d: lines(
                d,
                "predictive_skill",
                "Predictive skill (negative = worse than baseline)",
                series="series",
                reference=0,
            ),
        )
    )
    affine = read("affine_decoding")
    affine = affine.groupby(["distance", "method", "dimensions"] + context, as_index=False)[
        ["loo_joint_skill", "loo_arousal_skill"]
    ].mean()
    affine = affine.melt(
        id_vars=["distance", "method", "dimensions"] + context,
        value_vars=["loo_joint_skill", "loo_arousal_skill"],
        var_name="axis",
        value_name="skill",
    )
    affine["series"] = (
        affine.method + " / " + affine.axis.str.replace("loo_", "").str.replace("_skill", "")
    )
    jobs.append(
        (
            "23_affine_decoding",
            affine,
            "Separate affine decoder: average held-out label skill over blocks",
            lambda d: lines(
                d, "skill", "LOO predictive skill", x="dimensions", series="series", reference=0
            ),
        )
    )

    def chord_plot(d):
        fig, axs = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
        for (transform, method), group in d.groupby(["transform", "method"]):
            label = transform + " / " + method
            axs[0].plot(group.layer, group.va_r2, label=label)
            axs[1].plot(group.layer, group.negative_eigenvalue_fraction, label=label)
        axs[0].set(xlabel="Block", ylabel="2D V–A R²")
        axs[1].set(xlabel="Block", ylabel="Negative / absolute spectral mass")
        fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
        return fig

    jobs.append(
        (
            "24_cosine_chord",
            read("cosine_metric_sensitivity"),
            "Cosine dissimilarity versus Euclidean chord distance on the same normalized centroids",
            chord_plot,
        )
    )
    for name, frame, title, function in progress(
        jobs, description="Rendering comprehensive report figures", unit="figure"
    ):
        draw(name, frame, title, function)
    pd.DataFrame(registry).to_csv(report / "figure_index.csv", index=False)
