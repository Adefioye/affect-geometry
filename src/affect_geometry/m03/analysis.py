"""Evaluation of frozen predictions. No function here selects model settings."""

from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from affect_geometry.classification.protocol import EMOTIONS, digest
from affect_geometry.classification.storage import atomic_json
from affect_geometry.logging import progress
from affect_geometry.m03.configuration import Config, fingerprint
from affect_geometry.m03.metrics import (
    GroupBootstrap,
    class_counts,
    intervals,
    neutral_transitions,
    paired,
    ranks_and_margins,
    summarize,
)
from affect_geometry.m03.pipeline import cached, setup, versions
from affect_geometry.m03.rules import LENSES, Arm, arms, log_prob


def label_ids(labels):
    return np.asarray([EMOTIONS.index(x) if x in EMOTIONS else -1 for x in labels])


class ParquetSink:
    def __init__(self, path: Path):
        self.path, self.writer = path, None
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, frame: pd.DataFrame):
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path.with_suffix(".tmp"), table.schema)
        self.writer.write_table(table.cast(self.writer.schema))

    def close(self, *, success: bool):
        if self.writer is not None:
            self.writer.close()
            if success:
                self.path.with_suffix(".tmp").replace(self.path)


def _context(config, run_id, variant, mode):
    return dict(
        model=config.model_id,
        model_revision=config.model_revision,
        dataset_revision=config.dataset_revision,
        split="test",
        run_id=run_id,
        variant=variant,
        score_mode=mode,
        bootstrap_seed=config.seed,
        bootstrap_replicates=config.bootstrap_replicates,
        interval_type="pointwise duplicate-text-group bootstrap",
        n_blocks=config.n_blocks,
        test_history=config.test_history,
    )


def _annotate(frame, context):
    return frame.assign(**context)


def _deduplicated(gold, prediction, rows, duplicates):
    cross = set(duplicates.loc[duplicates.cross_split_duplicate, "text_hash"])
    seen, keep = set(), []
    for i, row in enumerate(rows):
        h = row["text_hash"]
        if h not in cross and h not in seen:
            keep.append(i)
            seen.add(h)
    return (
        dict(
            deduplicated_n=len(keep),
            **{
                f"deduplicated_{k}": v
                for k, v in summarize(gold[keep], prediction[keep]).items()
                if k != "n"
            },
        )
        if keep
        else dict(deduplicated_n=0, deduplicated_accuracy=None, deduplicated_macro_f1=None)
    )


def _diagnostics(scores, plan, rows, gold, native, greedy):
    transitions, availability, stability, agreement = [], [], [], []
    for lens in LENSES:
        layers = plan["coverage"][lens]
        panel = np.stack([np.argmax(scores[(lens, layer)], axis=-1) for layer in layers], axis=1)
        correct = panel == gold[:, None]
        headline = panel[:, layers.index(plan["headline_layer"])]
        for i, row in enumerate(rows):
            found = np.flatnonzero(correct[i])
            transitions.append(
                dict(
                    sample_id=row["sample_id"],
                    lens=lens,
                    gold=row["emotion"],
                    first_correct_layer=layers[found[0]] if len(found) else -1,
                    native_final_correct=bool(native[i] == gold[i]),
                    correct_to_wrong=int((correct[i, :-1] & ~correct[i, 1:]).sum()),
                    wrong_to_correct=int((~correct[i, :-1] & correct[i, 1:]).sum()),
                    prediction_changes=int((panel[i, :-1] != panel[i, 1:]).sum()),
                    layer_count=len(layers),
                )
            )
        for j in range(1, len(layers)):
            stability.append(
                dict(
                    lens=lens,
                    layer=layers[j],
                    previous_layer=layers[j - 1],
                    layer_gap=layers[j] - layers[j - 1],
                    prediction_change_rate=float((panel[:, j] != panel[:, j - 1]).mean()),
                )
            )
        any_correct = correct.any(axis=1)
        for name, base in (("native", native), ("greedy", greedy)):
            errors = base != gold
            availability.append(
                dict(
                    lens=lens,
                    baseline=name,
                    layer_count=len(layers),
                    layers=json.dumps(layers),
                    availability=float(any_correct.mean()),
                    baseline_errors=int(errors.sum()),
                    baseline_error_availability=float(any_correct[errors].mean())
                    if errors.any()
                    else None,
                    predeclared_direct_accuracy=float((headline == gold).mean()),
                    gap_to_predeclared_direct=float(any_correct.mean() - (headline == gold).mean()),
                )
            )
    for layer in plan["common_layers"]:
        for a, b in combinations(LENSES, 2):
            left = np.argmax(scores[(a, layer)], axis=-1)
            right = np.argmax(scores[(b, layer)], axis=-1)
            for name, base in (("native", native), ("greedy", greedy)):
                lr = (left == gold) & (base != gold)
                rr = (right == gold) & (base != gold)
                union = int((lr | rr).sum())
                agreement.append(
                    dict(
                        left_lens=a,
                        right_lens=b,
                        layer=layer,
                        baseline=name,
                        prediction_agreement=float((left == right).mean()),
                        left_repairs=int(lr.sum()),
                        right_repairs=int(rr.sum()),
                        shared_repairs=int((lr & rr).sum()),
                        union_repairs=union,
                        repair_jaccard=float((lr & rr).sum() / union) if union else None,
                    )
                )
    return dict(
        trajectories=pd.DataFrame(transitions),
        availability=pd.DataFrame(availability),
        stability=pd.DataFrame(stability),
        agreement=pd.DataFrame(agreement),
    )


def _evaluate(
    root, config, run_id, variant, mode, rows, keys, values, frozen, duplicates, sinks, tables
):
    context = _context(config, run_id, variant, mode)
    gold = label_ids([r["emotion"] for r in rows])
    greedy = label_ids([r["predicted_emotion"] for r in rows])
    scores = {}
    for i, key in enumerate(keys):
        total = np.stack([v["label_scores"][i] for v in values])
        if mode != "label_only":
            total += np.stack([v["stop_scores"][i] for v in values])
        if mode == "length_normalized":
            total /= np.stack([v["token_counts"][i] + 1 for v in values])
        scores[key] = total
    plan = frozen["plan"]
    native_scores = scores[("native", plan["n_blocks"] - 1)]
    native = native_scores.argmax(-1)
    candidates = arms(scores, plan)
    if variant != "canonical" or mode != "total":
        candidates = [a for a in candidates if a.headline]
    candidates = [Arm("greedy", "greedy", "baseline", -1, 0, native_scores, True)] + candidates
    bootstrap = GroupBootstrap(
        gold, [r["text_hash"] for r in rows], config.bootstrap_replicates, config.seed
    )
    baseline_predictions = dict(native=native, greedy=greedy)
    distributions = {k: bootstrap.distribution(v) for k, v in baseline_predictions.items()}
    baseline_metrics = {k: summarize(gold, v) for k, v in baseline_predictions.items()}
    direct_distributions = {}
    thresholds = None
    if config.calibrated and variant == "canonical" and mode == "total":
        thresholds = json.loads((root / "thresholds.json").read_text())
        if thresholds["freeze_hash"] != fingerprint(frozen):
            raise ValueError("Reference thresholds differ from frozen artifacts")
    ranks_panel = []
    frozen_predictions = {}
    # A predeclared retrospective rule: first 2 hashes in each greedy outcome stratum.
    example_ids = set()
    for condition in (greedy == gold, (greedy != gold) & (greedy >= 0), greedy < 0):
        ids = [rows[i]["sample_id"] for i in np.flatnonzero(condition)]
        example_ids.update(sorted(ids, key=lambda s: digest(f"{config.seed}:{s}"))[:2])
    for arm in progress(candidates, description=f"M03 metrics {variant}/{mode}", unit="arm"):
        prediction = greedy if arm.name == "greedy" else arm.scores.argmax(-1)
        if arm.headline:
            frozen_predictions[arm.name] = prediction.copy()
        common = dict(
            arm=arm.name,
            lens=arm.lens,
            kind=arm.kind,
            layer=arm.layer,
            beta=float(arm.beta),
            headline=arm.headline,
            primary=arm.headline
            and arm.kind in {"direct", "contrast"}
            and variant == "canonical"
            and mode == "total",
            greedy_prompt_match=variant == "canonical",
        )
        point = summarize(gold, prediction)
        distribution = (
            distributions[arm.name]
            if arm.name in distributions
            else bootstrap.distribution(prediction)
        )
        metric = dict(
            **context,
            **common,
            **point,
            **intervals(distribution, distributions),
            **_deduplicated(gold, prediction, rows, duplicates),
        )
        comparisons = {}
        if arm.kind in {"contrast", "masked", "dynamic", "within"}:
            comparisons["predeclared_direct"] = scores[(arm.lens, plan["headline_layer"])].argmax(
                -1
            )
            if arm.kind == "dynamic":
                panel = np.stack(
                    [scores[(arm.lens, layer)].argmax(-1) for layer in plan["dynamic_layers"]],
                    axis=1,
                )
                offsets = np.searchsorted(plan["dynamic_layers"], arm.selected_layers)
                comparisons["same_layer_direct"] = panel[np.arange(len(rows)), offsets]
            else:
                comparisons["same_layer_direct"] = scores[(arm.lens, arm.layer)].argmax(-1)
            if arm.kind == "within":
                comparisons["upper_direct"] = scores[(arm.lens, plan["upper_layer"])].argmax(-1)
        for name, reference in comparisons.items():
            key = (arm.lens, name, reference.tobytes())
            if key not in direct_distributions:
                direct_distributions[key] = bootstrap.distribution(reference)
            metric.update(intervals(distribution, {name: direct_distributions[key]}))
            differences = paired(gold, prediction, reference)
            metric[f"delta_accuracy_{name}_pp"] = differences.pop("delta_accuracy_pp")
            metric.update({f"{k}_{name}": v for k, v in differences.items()})
            metric[f"delta_macro_f1_{name}_pp"] = 100 * (
                point["macro_f1"] - summarize(gold, reference)["macro_f1"]
            )
        for name, base in baseline_predictions.items():
            differences = paired(gold, prediction, base)
            metric[f"delta_accuracy_{name}_pp"] = differences.pop("delta_accuracy_pp")
            metric.update({f"{k}_{name}": v for k, v in differences.items()})
            metric[f"delta_macro_f1_{name}_pp"] = 100 * (
                point["macro_f1"] - baseline_metrics[name]["macro_f1"]
            )
        count = class_counts(gold, prediction)
        for label, name in enumerate(EMOTIONS):
            tables["class_metrics"].append(
                dict(
                    **context,
                    **common,
                    label=name,
                    **{k: float(v[label]) for k, v in count.items()},
                )
            )
        ranks, margin, top_margin = ranks_and_margins(arm.scores, gold)
        if arm.name != "greedy":
            metric.update(
                mrr=float((1 / ranks).mean()),
                recall_at_1=float((ranks <= 1).mean()),
                recall_at_3=float((ranks <= 3).mean()),
                recall_at_5=float((ranks <= 5).mean()),
            )
        if arm.kind == "masked":
            excluded = ~np.isfinite(arm.scores[np.arange(len(gold)), gold])
            metric.update(
                gold_excluded=int(excluded.sum()), gold_exclusion_rate=float(excluded.mean())
            )
        tables["metrics"].append(metric)
        frame = pd.DataFrame(
            dict(
                sample_id=[r["sample_id"] for r in rows],
                gold=[r["emotion"] for r in rows],
                prediction=[EMOTIONS[p] if p >= 0 else "INVALID" for p in prediction],
                correct=prediction == gold,
                native_prediction=[EMOTIONS[p] for p in native],
                greedy_prediction=[EMOTIONS[p] if p >= 0 else "INVALID" for p in greedy],
                greedy_valid=greedy >= 0,
                native_correct=native == gold,
                greedy_correct=greedy == gold,
                gold_rank=ranks if arm.name != "greedy" else np.full(len(rows), -1),
                gold_margin=margin if arm.name != "greedy" else np.full(len(rows), np.nan),
                top_two_margin=top_margin if arm.name != "greedy" else np.full(len(rows), np.nan),
                selected_layer=arm.selected_layers
                if arm.selected_layers is not None
                else np.full(len(rows), arm.layer),
                selected_beta=arm.selected_beta
                if arm.selected_beta is not None
                else np.full(len(rows), arm.beta, dtype=float),
            )
        ).assign(**context, **common)
        for name, base in baseline_predictions.items():
            frame[f"repair_{name}"] = (prediction == gold) & (base != gold)
            frame[f"damage_{name}"] = (prediction != gold) & (base == gold)
            for outcome, keep in (
                ("repair", (prediction == gold) & (base != gold)),
                ("damage", (prediction != gold) & (base == gold)),
                ("unchanged_error", (prediction != gold) & (prediction == base)),
            ):
                if arm.headline and arm.kind != "baseline":
                    indices = sorted(
                        np.flatnonzero(keep),
                        key=lambda i: digest(f"{config.seed}:{rows[i]['sample_id']}"),
                    )[:3]
                    for i in indices:
                        tables["examples"].append(
                            dict(
                                **context,
                                **common,
                                baseline=name,
                                outcome=outcome,
                                sample_id=rows[i]["sample_id"],
                                text=rows[i]["text"],
                                gold=rows[i]["emotion"],
                                prediction=EMOTIONS[prediction[i]],
                                baseline_prediction=EMOTIONS[base[i]]
                                if base[i] >= 0
                                else "INVALID",
                            )
                        )
        sinks["predictions"].write(frame)
        if arm.kind == "dynamic":
            selected = (
                frame.groupby(["selected_layer", "selected_beta"]).size().reset_index(name="n")
            )
            tables["dynamic_selection"].extend(
                selected.assign(**context, **common).to_dict("records")
            )
        if arm.kind == "masked":
            excluded = ~np.isfinite(arm.scores[np.arange(len(gold)), gold])
            for label, name in enumerate(EMOTIONS):
                mask = gold == label
                tables["mask_exclusion"].append(
                    dict(
                        **context,
                        **common,
                        label=name,
                        support=int(mask.sum()),
                        excluded=int(excluded[mask].sum()),
                        exclusion_rate=float(excluded[mask].mean()) if mask.any() else None,
                    )
                )
        if arm.headline:
            for g in range(28):
                for p in range(-1, 28):
                    tables["confusion"].append(
                        dict(
                            **context,
                            **common,
                            gold=EMOTIONS[g],
                            prediction=EMOTIONS[p] if p >= 0 else "INVALID",
                            n=int(((gold == g) & (prediction == p)).sum()),
                        )
                    )
            lengths = np.asarray(values[0]["token_counts"])[0]
            for length in sorted(set(lengths)):
                mask = lengths[gold] == length
                tables["token_length"].append(
                    dict(
                        **context,
                        **common,
                        gold_label_token_count=int(length),
                        n=int(mask.sum()),
                        accuracy=float((prediction[mask] == gold[mask]).mean())
                        if mask.any()
                        else None,
                    )
                )
            for name, base in baseline_predictions.items():
                for transition, counts in neutral_transitions(gold, prediction, base).items():
                    tables["neutral_transitions"].append(
                        dict(
                            **context,
                            **common,
                            baseline=name,
                            transition=transition,
                            **counts,
                        )
                    )
        if thresholds and arm.name in thresholds["thresholds"]:
            probs = np.exp(log_prob(arm.scores))
            threshold = np.asarray(thresholds["thresholds"][arm.name])
            exceeds = probs > threshold
            for label, name in enumerate(EMOTIONS):
                mask = gold == label
                tables["calibration"].append(
                    dict(
                        **context,
                        **common,
                        label=name,
                        threshold=threshold[label],
                        support=int(mask.sum()),
                        gold_frequency=float(mask.mean()),
                        exceed_fraction=float(exceeds[:, label].mean()),
                        calibrated_readout_statistic=float(
                            exceeds[np.arange(len(gold)), gold].mean()
                        ),
                        class_calibrated_statistic=float(exceeds[mask, label].mean())
                        if mask.any()
                        else None,
                        multiple_exceed_fraction=float((exceeds.sum(-1) > 1).mean()),
                        zero_exceed_fraction=float((exceeds.sum(-1) == 0).mean()),
                    )
                )
        if arm.kind == "direct" and variant == "canonical" and mode == "total":
            ranks_panel.extend(frame.loc[frame.sample_id.isin(example_ids)].to_dict("records"))
    if variant == "canonical" and mode == "total":
        for name, frame in _diagnostics(scores, plan, rows, gold, native, greedy).items():
            tables[name].extend(_annotate(frame, context).to_dict("records"))
        tables["rank_trajectories"].extend(ranks_panel)
        for i, row in enumerate(rows):
            if "kl_native_to_lens" in values[i]:
                for k, (lens, layer) in enumerate(keys):
                    tables["vocabulary_kl"].append(
                        dict(
                            **context,
                            sample_id=row["sample_id"],
                            lens=lens,
                            layer=layer,
                            prefix="empty candidate prefix; final prompt token",
                            kl_native_to_lens=float(values[i]["kl_native_to_lens"][k]),
                            prompt_hash=row["prompt_hash"],
                        )
                    )

    return frozen_predictions


def export_scores(root: Path, config: Config, variant="canonical") -> Path:
    rows, keys, values, _ = cached(root, "test", variant, config)
    sink = ParquetSink(root / "tables" / f"complete_label_scores_{variant}.parquet")
    success = False
    try:
        native = np.stack([v["label_scores"][0] + v["stop_scores"][0] for v in values])
        for k, (lens, layer) in enumerate(progress(keys, description="Exporting label scores")):
            label = np.stack([v["label_scores"][k] for v in values])
            stop = np.stack([v["stop_scores"][k] for v in values])
            count = np.stack([v["token_counts"][k] for v in values])
            sink.write(
                pd.DataFrame(
                    dict(
                        sample_id=np.repeat([r["sample_id"] for r in rows], 28),
                        source_split="test",
                        model=config.model_id,
                        model_revision=config.model_revision,
                        dataset_revision=config.dataset_revision,
                        variant=variant,
                        lens=lens,
                        layer=layer,
                        label=np.tile(EMOTIONS, len(rows)),
                        label_token_score=label.ravel(),
                        stop_score=stop.ravel(),
                        total_score=(label + stop).ravel(),
                        label_token_count=count.ravel(),
                        native_reference_score=native.ravel(),
                    )
                )
            )
        success = True
    finally:
        sink.close(success=success)
    return sink.path


def analyze(config: Config, run_id: str, *, export_complete_scores=False) -> Path:
    started = time.perf_counter()
    root = setup(config, run_id, "m03.analyze")
    atomic_json(root / "analysis_manifest.json", dict(complete=False, status="running"))
    duplicates = pd.read_csv(root / "prepared" / "duplicate_audit.csv")
    names = (
        "metrics",
        "class_metrics",
        "confusion",
        "examples",
        "mask_exclusion",
        "dynamic_selection",
        "token_length",
        "neutral_transitions",
        "calibration",
        "trajectories",
        "availability",
        "stability",
        "agreement",
        "rank_trajectories",
        "vocabulary_kl",
    )
    tables = {name: [] for name in names}
    sensitivities = []
    canonical_predictions, canonical_ids = None, None
    sinks = dict(predictions=ParquetSink(root / "tables" / "predictions.parquet"))
    success = False
    variants = ["canonical"] + (["alternate_menu"] if config.alternate_menu else [])
    variants += ["alternate_prompt"] if config.alternate_prompt else []
    try:
        for variant in variants:
            rows, keys, values, frozen = cached(root, "test", variant, config)
            modes = ["total"]
            if variant == "canonical":
                modes += ["label_only"] + (
                    ["length_normalized"] if config.length_normalized else []
                )
            for mode in modes:
                predictions = _evaluate(
                    root,
                    config,
                    run_id,
                    variant,
                    mode,
                    rows,
                    keys,
                    values,
                    frozen,
                    duplicates,
                    sinks,
                    tables,
                )
                ids = [row["sample_id"] for row in rows]
                if variant == "canonical" and mode == "total":
                    canonical_predictions, canonical_ids = predictions, ids
                else:
                    lookup = {sid: i for i, sid in enumerate(canonical_ids)}
                    offsets = [lookup[sid] for sid in ids]
                    gold = label_ids([row["emotion"] for row in rows])
                    for arm, prediction in predictions.items():
                        reference = canonical_predictions[arm][offsets]
                        sensitivities.append(
                            dict(
                                **_context(config, run_id, variant, mode),
                                arm=arm,
                                n=len(rows),
                                n_changed=int((prediction != reference).sum()),
                                changed_fraction=float((prediction != reference).mean()),
                                comparison="canonical prompt with label+stop total scores",
                                **paired(gold, prediction, reference),
                            )
                        )
            if export_complete_scores:
                export_scores(root, config, variant)
        success = True
    except Exception as exc:
        atomic_json(
            root / "analysis_manifest.json", dict(complete=False, status="failed", error=str(exc))
        )
        raise
    finally:
        for sink in sinks.values():
            sink.close(success=success)
    target = root / "tables"
    for name, records in tables.items():
        if records:
            pd.DataFrame(records).to_csv(target / f"{name}.csv", index=False)
    metrics = pd.DataFrame(tables["metrics"])
    primary = metrics.loc[
        (metrics.variant == "canonical") & (metrics.score_mode == "total") & metrics.headline
    ]
    primary.to_csv(target / "main_results.csv", index=False)
    pd.DataFrame(sensitivities).to_csv(target / "scoring_and_prompt_sensitivity.csv", index=False)
    metrics.loc[
        (metrics.variant == "canonical")
        & (metrics.score_mode == "total")
        & metrics.layer.isin(frozen["plan"]["common_layers"])
    ].to_csv(
        target / "common_layer_panel.csv",
        index=False,
    )
    # Required figures consume these independent, self-contained CSVs.
    plot_data = root / "plot_data"
    plot_data.mkdir(exist_ok=True)
    for name, kinds in (
        ("direct_curves", ["direct", "baseline"]),
        ("contrast_curves", ["contrast", "baseline"]),
    ):
        metrics.loc[
            metrics.kind.isin(kinds)
            & (metrics.variant == "canonical")
            & (metrics.score_mode == "total")
        ].to_csv(plot_data / f"{name}.csv", index=False)
    for name in ("class_metrics", "confusion", "rank_trajectories", "agreement", "vocabulary_kl"):
        if tables[name]:
            pd.DataFrame(tables[name]).to_csv(plot_data / f"{name}.csv", index=False)
    primary.to_csv(plot_data / "repairs.csv", index=False)
    timings = [json.loads(p.read_text()) for p in sorted((root / "runtime").glob("*.json"))]
    pd.DataFrame(timings).to_csv(target / "runtime.csv", index=False)
    atomic_json(
        root / "analysis_manifest.json",
        dict(
            complete=True,
            protocol="m03_label_free_v1",
            variants=variants,
            bootstrap_replicates=config.bootstrap_replicates,
            seed=config.seed,
            intervals="pointwise; paired text-group bootstrap; no significance tests",
            selection="No test/validation accuracy used to choose configurations",
            deduplicated_rule=(
                "one row per test text hash; exclude text shared with train/validation"
            ),
            retrospective_examples="first three SHA256(seed:sample_id) per arm/baseline/outcome",
            trajectory_examples="first two SHA256(seed:sample_id) per greedy outcome stratum",
            test_history=config.test_history,
            source_freeze_hash=fingerprint(frozen),
            cached_analysis_seconds=time.perf_counter() - started,
            analysis_versions=versions(),
        ),
    )
    return root
