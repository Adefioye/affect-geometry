"""All EXP3 figures read self-contained CSVs; no hidden analysis state."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from affect_geometry.analysis.geometry_comparisons import METHODS


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _metric_curve(ax, group, x, y, label):
    """Missing layers are gaps; crosses at the axis edge are not zero estimates."""
    group = group.sort_values(x)
    values = pd.to_numeric(group[y], errors="coerce").copy()
    valid = np.isfinite(values)
    if "status" in group:
        valid &= group.status.eq("ok")
    values.loc[~valid] = np.nan
    (line,) = ax.plot(group[x], values, ".-", label=label, alpha=0.7)
    if (~valid).any():
        ax.scatter(
            group.loc[~valid, x],
            np.full((~valid).sum(), 0.02),
            transform=ax.get_xaxis_transform(),
            marker="x",
            color=line.get_color(),
        )
        ax.text(
            0.01,
            0.98,
            "× at lower edge = unsupported",
            transform=ax.transAxes,
            va="top",
            fontsize=6,
        )


def metric_curve_table(metrics, scores):
    """Persist missing evaluated layers explicitly, keeping emotion-specific scatter separate."""
    if metrics.empty:
        return metrics.copy()
    metrics = metrics.copy()
    if "emotion" not in metrics:
        metrics["emotion"] = ""
    metrics["emotion"] = metrics.emotion.fillna("")
    keys = ["cohort", "method", "dimension", "setting", "reference", "metric", "emotion"]
    depth = scores.drop_duplicates("layer").set_index("layer").depth.to_dict()
    rows = []
    for _, group in metrics.groupby(keys, dropna=False):
        for layer in sorted(depth):
            present = group.loc[group.layer.eq(layer)]
            if len(present):
                rows.extend(present.to_dict("records"))
            else:
                row = group.iloc[0].to_dict()
                row.update(
                    layer=layer,
                    depth=depth[layer],
                    value=np.nan,
                    status="unsupported",
                    reason="missing_layer_metric",
                )
                rows.append(row)
    return pd.DataFrame(rows)


def render(root):
    root = Path(root)
    plot_data, plots = root / "plot_data", root / "plots"
    plot_data.mkdir(exist_ok=True)
    plots.mkdir(exist_ok=True)
    # Each exported table carries its own model/run, ordering and experiment context.
    for path in (root / "metrics").glob("*.csv"):
        frame = pd.read_csv(path)
        frame.to_csv(plot_data / path.name, index=False)
    for name in ("support", "counts"):
        pd.read_csv(root / f"cohorts/{name}.csv").to_csv(plot_data / f"{name}.csv", index=False)
    samples = pd.read_csv(root / "cohorts/samples.csv")
    if len(samples):
        composition = samples.groupby(["cohort", "emotion", "source_split"], dropna=False).agg(
            n_samples=("sample_id", "size"),
            mean_text_length=("text_length", "mean"),
            compatibility_invalid=("compat_valid_label", lambda s: int((~s.astype(bool)).sum())),
        )
        composition = composition.reset_index()
        context = pd.read_csv(root / "metrics/scores.csv").iloc[0]
        for key in ("model", "run_id", "emotion_set_id", "draw_id"):
            composition[key] = context[key]
        composition.to_csv(plot_data / "composition.csv", index=False)
    scores = pd.read_csv(plot_data / "scores.csv")
    for (cohort, dimension), part in scores.groupby(["cohort", "dimension"]):
        fig, ax = plt.subplots(figsize=(8, 4))
        for method, group in part.groupby("method"):
            group = group.sort_values("layer")
            (line,) = ax.plot(group.depth, group.cv_r2, label=f"{method}: held-out")
            ax.plot(
                group.depth,
                group.full_r2,
                "--",
                color=line.get_color(),
                label=f"{method}: full-data",
            )
            failed = group.loc[group.status.ne("ok")]
            if len(failed):
                ax.scatter(
                    failed.depth,
                    np.full(len(failed), 0.02),
                    marker="x",
                    transform=ax.get_xaxis_transform(),
                    color=line.get_color(),
                )
                ax.text(
                    0.01,
                    0.98,
                    "× at lower edge = unsupported",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=6,
                )
        ax.set(
            xlabel="Normalized block depth",
            ylabel="Procrustes R²",
            title=f"{cohort} · {dimension}D",
        )
        ax.legend(fontsize=7)
        _save(fig, plots / f"alignment-{cohort}-{dimension}d.png")
        fig, ax = plt.subplots(figsize=(8, 3))
        matrix = part.pivot(index="method", columns="layer", values="cv_r2").reindex(METHODS)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("lightgray")
        image = ax.imshow(np.ma.masked_invalid(matrix.to_numpy()), aspect="auto", cmap=cmap)
        ax.set_yticks(range(3), METHODS)
        ax.set_xticks(range(len(matrix.columns)), matrix.columns)
        ax.set(
            xlabel="Native block index",
            title=f"{cohort} {dimension}D held-out R²; gray = unsupported",
        )
        fig.colorbar(image, ax=ax)
        _save(fig, plots / f"layer-method-{cohort}-{dimension}d.png")
    cka = pd.read_csv(plot_data / "cka.csv")
    for (cohort, dimension), part in cka.groupby(["cohort", "dimension"]):
        fig, ax = plt.subplots(figsize=(5, 4))
        matrix = part.pivot(index="method_left", columns="method_right", values="cka").reindex(
            index=METHODS, columns=METHODS
        )
        image = ax.imshow(np.ma.masked_invalid(matrix.to_numpy()), vmin=0, vmax=1)
        labels = []
        for method in METHODS:
            row = part.loc[part.method_left.eq(method)].iloc[0]
            labels.append(
                f"{method}\nlayer {row.layer_left:g}"
                if pd.notna(row.layer_left)
                else f"{method}\nunsupported"
            )
        ax.set_xticks(range(3), labels, fontsize=8)
        ax.set_yticks(range(3), labels, fontsize=8)
        for i in range(3):
            for j in range(3):
                value = matrix.iloc[i, j]
                ax.text(
                    j, i, f"{value:.2f}" if np.isfinite(value) else "N/A", ha="center", va="center"
                )
        ax.set_title(f"{cohort} · {dimension}D · K={int(part.K.iloc[0])}")
        fig.colorbar(image, ax=ax)
        _save(fig, plots / f"cka-{cohort}-{dimension}d.png")
    coordinates = pd.read_csv(plot_data / "coordinates.csv")
    native = coordinates.loc[coordinates.coordinate_type.eq("native")]
    for (cohort, method, dim, setting), part in native.groupby(
        ["cohort", "method", "dimension", "setting"]
    ):
        layers = sorted(part.layer.unique())
        cols = min(4, len(layers))
        fig, axes = plt.subplots(
            int(np.ceil(len(layers) / cols)),
            cols,
            figsize=(cols * 4, int(np.ceil(len(layers) / cols)) * 3.5),
            squeeze=False,
        )
        for ax, layer in zip(axes.flat, layers):
            group = part.loc[part.layer.eq(layer)]
            good = group.loc[group.status.eq("ok")]
            if not len(good):
                ax.text(0.5, 0.5, "Unsupported", ha="center", transform=ax.transAxes)
            elif dim == 2:
                ax.scatter(good.x, good.y, s=12)
                for row in good.itertuples():
                    ax.annotate(row.emotion, (row.x, row.y), fontsize=6)
                neutral = good.loc[good.emotion.eq("neutral")]
                if len(neutral):
                    ax.axhline(neutral.y.iloc[0], color="gray", lw=0.5)
                    ax.axvline(neutral.x.iloc[0], color="gray", lw=0.5)
                ax.set(xlabel="Native coordinate 1", ylabel="Native coordinate 2")
            else:
                good = good.sort_values("x")
                ax.scatter(good.x, range(len(good)), s=12)
                ax.set_yticks(range(len(good)), good.emotion, fontsize=6)
                ax.set_xlabel("Native 1D coordinate")
            ax.set_title(f"Block {layer}")
        for ax in list(axes.flat)[len(layers) :]:
            ax.set_visible(False)
        fig.suptitle(f"{cohort} · {method} · {dim}D · {setting}", y=1.01)
        _save(fig, plots / f"coordinates-{cohort}-{method}-{dim}d-{setting}.png")
    aligned = coordinates.loc[
        coordinates.coordinate_type.eq("full_data_aligned") & coordinates.dimension.eq(1)
    ]
    if len(aligned):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for ax, cohort in zip(axes, ("correct", "incorrect", "all")):
            for (method, emotion), group in aligned.loc[aligned.cohort.eq(cohort)].groupby(
                ["method", "emotion"]
            ):
                ax.plot(group.layer, group.x, alpha=0.5, lw=0.8, label=f"{method}/{emotion}")
            ax.set(
                title=f"{cohort}: full-data valence aligned",
                xlabel="Block",
                ylabel="Aligned valence",
            )
        _save(fig, plots / "aligned-1d.png")
    # Metric-specific curves preserve common/native reference and matched setting labels.
    metrics = pd.read_csv(plot_data / "metrics.csv")
    metric_curve_table(metrics, scores).to_csv(plot_data / "metric_curves.csv", index=False)
    metrics = pd.read_csv(plot_data / "metric_curves.csv")
    if len(metrics):
        for (metric, reference, dimension), part in metrics.groupby(
            ["metric", "reference", "dimension"]
        ):
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            for ax, cohort in zip(axes, ("correct", "incorrect", "all")):
                subset = part.loc[part.cohort.eq(cohort)]
                for (method, setting, emotion), group in subset.groupby(
                    ["method", "setting", "emotion"], dropna=False
                ):
                    label = f"{method}/{setting}"
                    if pd.notna(emotion) and emotion:
                        label += f"/{emotion}"
                    _metric_curve(ax, group, "layer", "value", label)
                ax.set(title=cohort, xlabel="Block", ylabel=metric)
            if len(part.groupby(["method", "setting"])) <= 12:
                axes[-1].legend(fontsize=5)
            fig.suptitle(f"{metric} · {reference} · {dimension}D")
            _save(fig, plots / f"metric-{metric}-{reference}-{dimension}d.png")
    aggregates = pd.read_csv(plot_data / "aggregates.csv")
    for dim, part in aggregates.groupby("dimension"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, value in zip(axes, ("A_M", "delta_A_M")):
            labels = part.cohort + "/" + part.method
            ax.bar(range(len(part)), part[value])
            for i, row in enumerate(part.itertuples()):
                ax.text(i, 0, f"{row.completed_layers}/{row.layer_count}", rotation=90, fontsize=6)
                if not np.isfinite(getattr(row, value)):
                    ax.text(
                        i,
                        0.5,
                        "N/A",
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        rotation=90,
                        fontsize=7,
                    )
            ax.set_xticks(range(len(part)), labels, rotation=70, fontsize=7)
            ax.set_ylabel(value)
        fig.suptitle(f"{dim}D · equal weights over prespecified layers; counts show coverage")
        _save(fig, plots / f"aggregates-{dim}d.png")
    spectra = pd.read_csv(plot_data / "spectra.csv")
    if len(spectra):
        for (cohort, method, setting), part in spectra.groupby(["cohort", "method", "setting"]):
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            for layer, group in part.groupby("layer"):
                axes[0].plot(group["mode"], group.eigenvalue, label=f"Block {layer}")
                mass = group.sort_values("weight", ascending=False)
                axes[1].plot(range(1, len(mass) + 1), mass.cumulative_mass)
            axes[0].set(xlabel="Mode", ylabel="Eigenvalue")
            axes[1].set(
                xlabel="Contribution rank", ylabel="Cumulative positive mass / diffusion energy"
            )
            fig.suptitle(f"{cohort}/{method}/{setting}")
            _save(fig, plots / f"spectra-{cohort}-{method}-{setting}.png")
    diagrams, betti = (
        pd.read_csv(plot_data / "persistence.csv"),
        pd.read_csv(plot_data / "betti.csv"),
    )
    if len(diagrams):
        for (cohort, layer), part in diagrams.groupby(["cohort", "layer"]):
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            bound = part.filtration_bound.iloc[0]
            for dim, group in part.groupby("dimension"):
                axes[0].scatter(
                    group.birth, group.death.replace(np.inf, bound * 1.05), label=f"H{dim}", s=15
                )
                curve = betti.loc[
                    betti.cohort.eq(cohort) & betti.layer.eq(layer) & betti.dimension.eq(dim)
                ]
                axes[1].step(curve.filtration, curve.betti, where="post", label=f"H{dim}")
            axes[0].plot([0, bound], [0, bound], "--", color="gray")
            axes[0].set(xlabel="Birth", ylabel="Death (∞ above bound)")
            axes[1].set(xlabel="Cosine filtration", ylabel="Betti number")
            axes[1].legend()
            fig.suptitle(f"{cohort}/block {layer} · nonmetric cosine clique filtration")
            _save(fig, plots / f"topology-{cohort}-{layer}.png")
    render_support_and_predictions(root)
    # Data-derived summary only; scientific interpretation is a separate report task.
    report = root / "reports"
    report.mkdir(exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    (report / "run.md").write_text(
        "# EXP3 run inventory\n\n"
        f"Mode: **{manifest['inference_label']}**.\n\n"
        f"Model: {manifest['model']}. Layers: {manifest['layers']}.\n\n"
        f"Supported emotion order: {', '.join(manifest['emotion_order'])}.\n\n"
        f"Successful score cells: {int(scores.status.eq('ok').sum())}/{len(scores)}.\n\n"
        "Scores evaluate transductive representations against held-out NRC ratings. "
        "Full-data maps and selected-layer CKA are descriptive. Cohort and method contrasts "
        "are descriptive; all overlaps both balanced subsets. No prompt-sampling confidence "
        "intervals are reported.\n\n"
        "See `../metrics/failures.csv` for unsupported settings, `../metrics/aggregates.csv` "
        "for coverage, and `../inference/` for correction-family completion or skip status. "
        "Each figure reads its corresponding self-contained table under `../plot_data/`.\n"
    )


def compare_models(roots, destination, failures=()):
    """Inventory support differences; no unmatched-support cross-model score contrasts."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifests = [json.loads((Path(root) / "manifest.json").read_text()) for root in roots]
    intersection = (
        sorted(set.intersection(*(set(m["emotion_order"]) for m in manifests))) if manifests else []
    )
    records = []
    for root, manifest in zip(roots, manifests, strict=True):
        support = pd.read_csv(Path(root) / "cohorts/support.csv")
        for row in support.to_dict("records"):
            records.append(
                row
                | dict(
                    model=manifest["model"],
                    run_id=manifest["config"]["run_id"],
                    in_common_intersection=row["emotion"] in intersection,
                )
            )
    pd.DataFrame(records).to_csv(destination / "model_support.csv", index=False)
    (destination / "manifest.json").write_text(
        json.dumps(
            dict(
                common_emotions=intersection,
                model_emotions={m["model"]: m["emotion_order"] for m in manifests},
                source_runs=[str(p) for p in roots],
                failures=failures,
                comparison_policy="model_specific_balanced_draws; common emotion intersection only",
            ),
            indent=2,
        )
    )


def render_support_and_predictions(root):
    root = Path(root)
    data, plots = root / "plot_data", root / "plots"
    predictions = [
        pd.read_csv(p) for p in sorted((root / "validation").glob("*/*/layer_*/*/predictions.csv"))
    ]
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(
            data / "held_out_predictions.csv", index=False
        )
        frame = pd.read_csv(data / "held_out_predictions.csv")
        for (cohort, dimension), group in frame.groupby(["cohort", "dimension"]):
            fig, axes = plt.subplots(
                1, int(dimension), figsize=(5 * int(dimension), 4), squeeze=False
            )
            for ax, axis in zip(axes.flat, ["valence", "arousal"][: int(dimension)]):
                for method, part in group.groupby("method"):
                    ax.scatter(
                        part[f"observed_{axis}"],
                        part[f"predicted_{axis}"],
                        s=10,
                        alpha=0.4,
                        label=method,
                    )
                ax.set(xlabel=f"Observed {axis}", ylabel=f"Held-out predicted {axis}", title=cohort)
                ax.legend(fontsize=7)
            _save(fig, plots / f"held-out-{cohort}-{dimension}d.png")
    scores = pd.read_csv(data / "scores.csv")
    for axis in ("valence_pearson", "arousal_pearson"):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, cohort in zip(axes, ("correct", "incorrect", "all")):
            for (method, dim), part in scores.loc[scores.cohort.eq(cohort)].groupby(
                ["method", "dimension"]
            ):
                if axis == "arousal_pearson" and dim == 1:
                    continue
                _metric_curve(ax, part, "depth", axis, f"{method}/{dim}D")
            ax.set(title=cohort, xlabel="Normalized depth", ylabel=axis)
        if axes[-1].get_legend_handles_labels()[0]:
            axes[-1].legend(fontsize=6)
        _save(fig, plots / f"{axis}.png")
    counts = pd.read_csv(data / "counts.csv")
    for phase, frame in counts.groupby("phase"):
        fig, ax = plt.subplots(figsize=(12, 4))
        grouped = frame.groupby(["emotion", "cohort"]).n_samples.sum().unstack("cohort")
        grouped.plot.bar(ax=ax)
        ax.set(title=f"{phase} sampling: source counts", ylabel="Samples")
        _save(fig, plots / f"support-{phase}.png")
    if (data / "composition.csv").exists():
        frame = pd.read_csv(data / "composition.csv")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        frame.groupby(["emotion", "cohort"]).compatibility_invalid.sum().unstack("cohort").plot.bar(
            ax=axes[0]
        )
        frame.groupby(["source_split", "cohort"]).n_samples.sum().unstack("cohort").plot.bar(
            ax=axes[1]
        )
        axes[0].set_ylabel("Compatibility-invalid sampled outputs")
        axes[1].set_ylabel("Sample count by source split")
        _save(fig, plots / "composition.png")
    frame = pd.read_csv(data / "contrasts.csv")
    for dim, group in frame.groupby("dimension"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for key, part in group.groupby(
            ["cohort_left", "method_left", "cohort_right", "method_right"]
        ):
            ax = axes[0] if key[0] == key[2] else axes[1]
            _metric_curve(ax, part, "layer", "delta_r2", "/".join(key))
        for ax, title in zip(axes, ["Method contrasts", "Cohort contrasts"]):
            ax.axhline(0, color="gray", lw=0.5)
            ax.set(title=title, xlabel="Block", ylabel="Paired CV R² difference")
            ax.legend(fontsize=5)
        _save(fig, plots / f"contrasts-{dim}d.png")
