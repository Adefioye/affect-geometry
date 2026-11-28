"""Paired strict-accuracy reports; no edited membership inference from two scored paths."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from affect_geometry.classification.protocol import EMOTIONS


def prediction_row(record):
    generation = record["generation"]
    gold, rival = record["emotion"], record["rival"]
    values = {
        k: record[k]
        for k in (
            "sample_id",
            "emotion",
            "duplicate_group",
            "community_id",
            "arm",
            "alpha",
            "donor_id",
            "donor_prompt_hash",
            "abstention",
            "prompt_hash",
            "rival",
        )
    }
    values.update(
        {
            k: generation[k]
            for k in (
                "raw_response",
                "generated_token_ids",
                "predicted_emotion",
                "correct",
                "valid_label",
                "outcome",
                "generation_finish_reason",
            )
        }
    )
    for name, label in (("gold", gold), ("rival", rival)):
        for part in ("total", "label_score", "stop_score"):
            values[f"{name}_{part}"] = record["scores"][label][part]
    values["margin"] = values["gold_total"] - values["rival_total"]
    return values


def export_predictions(store, ledger):
    import pyarrow as pa

    string_fields = [
        "sample_id",
        "emotion",
        "duplicate_group",
        "community_id",
        "arm",
        "donor_id",
        "donor_prompt_hash",
        "abstention",
        "prompt_hash",
        "rival",
        "raw_response",
        "predicted_emotion",
        "outcome",
        "generation_finish_reason",
    ]
    schema = pa.schema(
        [(v, pa.string()) for v in string_fields]
        + [
            ("alpha", pa.float64()),
            ("generated_token_ids", pa.list_(pa.int64())),
            ("correct", pa.bool_()),
            ("valid_label", pa.bool_()),
        ]
        + [
            (f"{n}_{p}", pa.float64())
            for n in ("gold", "rival")
            for p in ("total", "label_score", "stop_score")
        ]
        + [("margin", pa.float64())]
    )
    predictions = store.stream_parquet(
        "predictions.parquet", map(prediction_row, ledger.rows()), schema
    )

    def path_rows():
        for record in ledger.rows():
            for label, scores in record["scores"].items():
                yield dict(
                    sample_id=record["sample_id"],
                    community_id=record["community_id"],
                    arm=record["arm"],
                    label=label,
                    **scores,
                )

    paths = store.stream_parquet(
        "path_scores.parquet",
        path_rows(),
        pa.schema(
            [(v, pa.string()) for v in ("sample_id", "community_id", "arm", "label")]
            + [(v, pa.float64()) for v in ("total", "label_score", "stop_score")]
            + [("label_tokens", pa.int64()), ("token_ids", pa.list_(pa.int64()))]
        ),
    )

    def norm_rows():
        for record in ledger.rows():
            for purpose, rows in (
                ("generation", record["generation"].get("edit_audit", [])),
                ("scoring", record.get("score_edit_audit", [])),
            ):
                for row in rows:
                    yield dict(
                        sample_id=record["sample_id"],
                        community_id=record["community_id"],
                        arm=record["arm"],
                        purpose=purpose,
                        **row,
                    )

    norms = store.stream_parquet(
        "edit_norms.parquet",
        norm_rows(),
        pa.schema(
            [(v, pa.string()) for v in ("sample_id", "community_id", "arm", "purpose")]
            + [(v, pa.int64()) for v in ("layer", "batch_row", "position", "count", "forward")]
            + [("norm", pa.float64())]
        ),
    )
    return [predictions, paths, norms]


def paired_bootstrap(differences, groups, replicates=2000, seed=42):
    """All contrasts share recipient-group draws; duplicate groups are resampled as units."""
    differences = np.asarray(differences, dtype=float)
    if differences.ndim == 1:
        differences = differences[:, None]
    if not len(differences):
        raise ValueError("Cannot bootstrap an empty population")
    codes, unique = pd.factorize(np.asarray(groups), sort=True)
    counts = np.bincount(codes, minlength=len(unique))
    sums = np.zeros((len(unique), differences.shape[1]))
    np.add.at(sums, codes, differences)
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, differences.shape[1]))
    for i in range(replicates):
        indices = rng.integers(0, len(unique), len(unique))
        samples[i] = sums[indices].sum(0) / counts[indices].sum()
    return samples


def outcome_metrics(frame, baseline, population="full_test"):
    if frame.sample_id.duplicated().any() or set(frame.sample_id) != set(baseline.sample_id):
        raise ValueError("Incomplete or duplicated paired prediction population")
    frame = frame.set_index("sample_id").loc[baseline.sample_id].reset_index()
    before, after = baseline.correct.to_numpy(bool), frame.correct.to_numpy(bool)
    repair, damage = ~before & after, before & ~after
    return dict(
        population=population,
        n=len(frame),
        accuracy=float(after.mean()),
        baseline_accuracy=float(before.mean()),
        accuracy_delta=float(after.mean() - before.mean()),
        repaired=int(repair.sum()),
        damaged=int(damage.sum()),
        unchanged_correct=int((before & after).sum()),
        unchanged_wrong=int((~before & ~after).sum()),
        repair_rate=float(repair.sum() / (~before).sum()) if (~before).any() else None,
        damage_rate=float(damage.sum() / before.sum()) if before.any() else None,
        invalid_rate=float((~frame.valid_label).mean()),
        format_repairs=int((repair & ~baseline.valid_label.to_numpy(bool)).sum()),
        valid_wrong_repairs=int((repair & baseline.valid_label.to_numpy(bool)).sum()),
        donor_coverage=float(frame.donor_id.notna().mean()),
        macro_f1=float(
            f1_score(
                frame.emotion,
                frame.predicted_emotion.fillna("<invalid>"),
                labels=EMOTIONS,
                average="macro",
                zero_division=0,
            )
        ),
    ), frame


def diagnostic_summaries(store, ordered, baseline):
    """Summarize prespecified score contrasts and measured norms without accuracy proxies."""
    summaries = []
    for (cid, arm), frame in ordered.items():
        for group, mask in (
            ("all", np.ones(len(baseline), dtype=bool)),
            ("baseline_correct", baseline.correct.to_numpy(bool)),
            ("baseline_incorrect", ~baseline.correct.to_numpy(bool)),
        ):
            if not mask.any():
                continue
            summaries.append(
                dict(
                    community_id=cid,
                    arm=arm,
                    outcome_group=group,
                    n=int(mask.sum()),
                    **{
                        f"mean_{column}_delta": float(
                            (frame[column].to_numpy() - baseline[column].to_numpy())[mask].mean()
                        )
                        for column in ("margin", "gold_label_score", "gold_stop_score")
                    },
                )
            )
    paths = [store.csv("score_diagnostics.csv", pd.DataFrame(summaries))]
    effects = pd.read_parquet(store.root / "test_ablation_effects.parquet")
    ablations = []
    if len(effects):
        expanded = pd.concat([effects, effects.assign(emotion="__all__")], ignore_index=True)
        for (cid, label), group in expanded.groupby(["community_id", "emotion"], sort=True):
            row = dict(community_id=cid, emotion=label, n=group.sample_id.nunique())
            for column in ("margin_effect", "gold_label_effect", "gold_stop_effect"):
                paired = group.pivot(index="sample_id", columns="arm", values=column)
                if set(paired.columns) != {"community", "random"} or paired.isna().any().any():
                    raise ValueError("Held-out ablation arms are not completely paired")
                row.update(
                    {
                        f"mean_{arm}_{column}": float(paired[arm].mean())
                        for arm in ("community", "random")
                    }
                )
                row[f"mean_net_{column}"] = float((paired.community - paired.random).mean())
            ablations.append(row)
    effect_columns = [
        f"mean_{arm}_{column}"
        for column in ("margin_effect", "gold_label_effect", "gold_stop_effect")
        for arm in ("community", "random", "net")
    ]
    paths.append(
        store.csv(
            "heldout_ablation_summary.csv",
            pd.DataFrame(ablations, columns=["community_id", "emotion", "n", *effect_columns]),
        )
    )
    norms = pd.read_parquet(store.root / "edit_norms.parquet")
    columns = ["community_id", "arm", "purpose", "layer", "n", "mean_norm", "p95_norm"]
    summary = pd.DataFrame(columns=columns)
    if len(norms):
        summary = (
            norms.groupby(columns[:4], sort=True)["norm"]
            .agg(n="size", mean_norm="mean", p95_norm=lambda values: values.quantile(0.95))
            .reset_index()
        )
    paths.append(store.csv("edit_norm_summary.csv", summary))
    return paths


def write_report(store, config):
    shortlist = store.read_json("shortlist.json")
    communities = shortlist["communities"]
    if not communities:
        path = store._write(
            "report.md",
            lambda p: p.write_text(
                "# EXP5 discovery finding\n\nNo supported communities qualified on training data.\n"
                "Test interventions were not run. This does not establish an absence "
                "of sparse mechanisms.\n"
            ),
        )
        return [path]
    data = pd.read_parquet(store.root / "predictions.parquet")
    baseline = data[data.arm.eq("baseline")].sort_values("sample_id").reset_index(drop=True)
    expected = store.read_json("frozen.json")["payload"]["test_ids"]
    if baseline.sample_id.tolist() != sorted(expected):
        raise ValueError("Report population differs from the frozen test IDs")
    metrics, per_label, differences, identifiers = [], [], [], []
    ordered = {}
    for community in communities:
        cid = community["community_id"]
        for arm in ("community", "random", "unclustered", "wrong_label"):
            frame = data[data.community_id.eq(cid) & data.arm.eq(arm)]
            summary, aligned = outcome_metrics(frame, baseline)
            metrics.append(dict(community_id=cid, arm=arm, **summary))
            ordered[(cid, arm)] = aligned
            differences.append(aligned.correct.to_numpy(float) - baseline.correct.to_numpy(float))
            identifiers.append((cid, arm))
            for label in EMOTIONS:
                mask = aligned.emotion.eq(label)
                support = int(mask.sum())
                per_label.append(
                    dict(
                        community_id=cid,
                        arm=arm,
                        emotion=label,
                        n=support,
                        correct=int(aligned.loc[mask, "correct"].sum()),
                        recall=float(aligned.loc[mask, "correct"].mean()) if support else None,
                        donor_coverage=float(aligned.loc[mask, "donor_id"].notna().mean())
                        if support
                        else None,
                    )
                )
        common = np.logical_and.reduce(
            [
                ordered[(cid, arm)].donor_id.notna().to_numpy()
                for arm in ("community", "random", "unclustered", "wrong_label")
            ]
        )
        if common.any():
            for arm in ("community", "random", "unclustered", "wrong_label"):
                summary, _ = outcome_metrics(
                    ordered[(cid, arm)].loc[common], baseline.loc[common], "common_donor_coverage"
                )
                metrics.append(dict(community_id=cid, arm=arm, **summary))
    primary = shortlist["primary"]
    primary_after = ordered[(primary, "community")].correct.to_numpy(float)
    for arm in ("random", "unclustered"):
        differences.append(primary_after - ordered[(primary, arm)].correct.to_numpy(float))
    samples = paired_bootstrap(
        np.column_stack(differences),
        baseline.duplicate_group,
        config.bootstrap_replicates,
        config.seed,
    )
    intervals = {}
    for i, identifier in enumerate(identifiers):
        intervals[identifier] = np.quantile(samples[:, i], [0.025, 0.975])
    for row in metrics:
        if row["population"] == "full_test":
            row["ci_low"], row["ci_high"] = intervals[(row["community_id"], row["arm"])]
    primary_index = identifiers.index((primary, "community"))
    contrast_indices = (primary_index, len(identifiers), len(identifiers) + 1)
    contrasts = []
    for name, index in zip(("baseline", "random", "unclustered"), contrast_indices, strict=True):
        low, high = np.quantile(samples[:, index], [0.05 / 6, 1 - 0.05 / 6])
        contrasts.append(
            dict(
                community_id=primary,
                comparator=name,
                delta=float(np.mean(differences[index])),
                ci_low=float(low),
                ci_high=float(high),
                confidence=1 - 0.05 / 3,
                positive=bool(low > 0),
                scope=store.manifest["inference_scope"],
            )
        )
    metric_frame = pd.DataFrame(metrics)
    paths = [
        store.csv("metrics.csv", metric_frame),
        store.csv("per_label_metrics.csv", pd.DataFrame(per_label)),
        store.csv("primary_contrasts.csv", pd.DataFrame(contrasts)),
    ]
    paths.extend(diagnostic_summaries(store, ordered, baseline))
    usage = data[data.donor_id.notna()].groupby(["community_id", "arm", "donor_id"]).size()
    paths.append(store.csv("donor_usage.csv", usage.rename("n_recipients").reset_index()))
    plot_path = store.csv("plot_data/accuracy_deltas.csv", metric_frame)
    paths.append(plot_path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot = pd.read_csv(plot_path)
    plot = plot[plot.population.eq("full_test") & plot.arm.eq("community")]
    fig, ax = plt.subplots(figsize=(8, max(3, len(plot) * 0.32)))
    y = np.arange(len(plot))
    point = plot.accuracy_delta.to_numpy() * 100
    errors = np.vstack(
        (
            np.maximum(0, point - plot.ci_low.to_numpy() * 100),
            np.maximum(0, plot.ci_high.to_numpy() * 100 - point),
        )
    )
    ax.errorbar(point, y, xerr=errors, fmt="o", capsize=3)
    ax.set_yticks(y, [v[:10] for v in plot.community_id])
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_xlabel("Strict accuracy change (percentage points); descriptive 95% intervals")
    ax.set_title("Oracle-guided community transfer")
    fig.tight_layout()
    target = store.root / "figures/accuracy_deltas.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160)
    plt.close(fig)
    paths.append(target)
    primary_row = metric_frame[
        (metric_frame.population.eq("full_test"))
        & metric_frame.community_id.eq(primary)
        & metric_frame.arm.eq("community")
    ]
    result = primary_row.iloc[0]
    text = (
        "# EXP5: oracle-guided transfer\n\n"
        f"Scope: {store.manifest['inference_scope']}. Test rows: {len(baseline)}. "
        f"Frozen communities: {len(communities)}. Primary: `{primary}`.\n\n"
        f"Primary accuracy change: {100 * result.accuracy_delta:.3f} percentage points; "
        f"repairs: {int(result.repaired)}; damage: {int(result.damaged)}; "
        f"donor coverage: {100 * result.donor_coverage:.1f}%.\n\n"
        "The three primary contrasts use Bonferroni-adjusted paired duplicate-group bootstrap "
        "intervals in primary_contrasts.csv. Other community comparisons are exploratory. "
        "All test rows, invalid outputs, and donor abstentions remain in the primary "
        "denominator.\n\n"
        "Gold-guided donors do not demonstrate autonomous classification improvement. Two edited "
        "paths measure a fixed contrast, not 28-way membership accuracy. Held-out ablations are "
        "score effects, not measured label flips or full necessity/sufficiency certification. "
        "One random group per community and reused donors limit generalization over donor/control "
        "populations. Location specificity additionally requires norm-matched controls.\n\n"
        "See metrics.csv, per_label_metrics.csv, primary_contrasts.csv, donor_usage.csv, "
        "score_diagnostics.csv, heldout_ablation_summary.csv, edit_norm_summary.csv, "
        "and the raw condition/path/norm records.\n"
    )
    paths.append(store._write("report.md", lambda p: p.write_text(text)))
    return paths
