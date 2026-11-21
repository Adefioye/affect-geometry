"""Validate and compare paired canonical pilot caches without annotation labels."""

import json

import numpy as np
import pandas as pd

from affect_geometry.classification.protocol import EMOTIONS
from affect_geometry.classification.storage import atomic_json
from affect_geometry.logging import get_logger, progress
from affect_geometry.m03 import pipeline
from affect_geometry.m03.configuration import fingerprint
from affect_geometry.m03.rules import layer_plan

from .diagnostic import read_checks

BATCH_FIELDS = {"prefix_batch_size", "projection_batch_size"}


def validate_protocols(left, right):
    def comparable(manifest):
        return dict(
            manifest,
            config={k: v for k, v in manifest["config"].items() if k not in BATCH_FIELDS},
        )

    if comparable(left) != comparable(right):
        raise ValueError("Paired protocol manifests differ beyond the two batch settings")
    for manifest in (left, right):
        if tuple(manifest["labels"]) != tuple(EMOTIONS):
            raise ValueError("Unexpected candidate labels or ordering")


def _modes(arrays, normalized):
    total = arrays["label_scores"] + arrays["stop_scores"]
    modes = {"total": total, "label_only": arrays["label_scores"]}
    if normalized:
        modes["length_normalized"] = total / (arrays["token_counts"] + 1)
    return modes


def _stats(left, right, tolerance):
    absolute = np.abs(left - right)
    return dict(
        score_count=absolute.size,
        median_abs_difference=np.median(absolute),
        p95_abs_difference=np.quantile(absolute, 0.95),
        max_abs_difference=absolute.max(),
        count_above_tolerance=int((absolute > tolerance).sum()),
    )


def layer_tables(ids, keys, arrays1, arrays8, *, normalized, context):
    agreement, differences, flips = [], [], []
    modes1 = _modes(arrays1, normalized)
    modes8 = _modes(arrays8, normalized)
    for k, (lens, layer) in progress(
        list(enumerate(keys)), description="Comparing paired M03 readouts", unit="readout"
    ):
        row = dict(context, lens=lens, layer=layer, comment_count=len(ids))
        for mode, scores1 in modes1.items():
            left, right = scores1[:, k], modes8[mode][:, k]
            pred1, pred8 = left.argmax(-1), right.argmax(-1)
            changed = np.flatnonzero(pred1 != pred8)
            agreement.append(
                dict(
                    row,
                    score_mode=mode,
                    flip_count=len(changed),
                    top1_agreement_count=len(ids) - len(changed),
                    top1_agreement=1 - len(changed) / len(ids),
                    **_stats(left, right, context["tolerance"]),
                )
            )
            top_two = np.partition(left, -2, axis=-1)[:, -2:]
            for i in changed:
                flips.append(
                    dict(
                        row,
                        score_mode=mode,
                        sample_id=ids[i],
                        bs1_prediction=EMOTIONS[pred1[i]],
                        bs8_prediction=EMOTIONS[pred8[i]],
                        bs1_top_two_margin=top_two[i, 1] - top_two[i, 0],
                    )
                )
        for component in ("label_scores", "stop_scores"):
            differences.append(
                dict(
                    row,
                    score_component=component,
                    **_stats(
                        arrays1[component][:, k], arrays8[component][:, k], context["tolerance"]
                    ),
                )
            )
    flip_columns = [
        *context,
        "lens",
        "layer",
        "comment_count",
        "score_mode",
        "sample_id",
        "bs1_prediction",
        "bs8_prediction",
        "bs1_top_two_margin",
    ]
    return (
        pd.DataFrame(agreement),
        pd.DataFrame(differences),
        pd.DataFrame(flips, columns=flip_columns),
    )


def _runtime(values, checks, config):
    run = checks["runtime"]
    if (
        run["prefix_batch_size"] != config.prefix_batch_size
        or run["projection_batch_size"] != config.projection_batch_size
        or run["examples"] != len(values)
        or run["newly_extracted"] + run["reused"] != len(values)
    ):
        raise ValueError("Pilot runtime counts or batch sizes disagree with the cache")
    seconds = sum(float(value["extraction_seconds"]) for value in values)
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Missing positive per-comment scorer timings")
    elapsed = run["extraction_seconds"]
    return dict(
        run,
        batch_size=config.prefix_batch_size,
        committed_scorer_seconds=seconds,
        committed_comments_per_second=len(values) / seconds,
        new_comments_per_extraction_second=run["newly_extracted"] / elapsed if elapsed else None,
    )


def compare(configs, run_ids, root):
    roots = [cfg.root(run_id) for cfg, run_id in zip(configs, run_ids, strict=True)]
    manifests = [pipeline.protocol(path, cfg) for path, cfg in zip(roots, configs, strict=True)]
    validate_protocols(*manifests)
    caches = [
        pipeline.cached(path, "pilot", "canonical", cfg)
        for path, cfg in zip(roots, configs, strict=True)
    ]
    (rows1, keys1, values1, freeze1), (rows8, keys8, values8, freeze8) = caches
    ids = [row["sample_id"] for row in rows1]
    if (
        len(ids) != 64
        or len(set(ids)) != 64
        or ids != [row["sample_id"] for row in rows8]
        or ids != manifests[0]["selected_ids"]["pilot"]
    ):
        raise ValueError("Expected the same 64 unique canonical pilot IDs in frozen order")
    paths = [path / "scores" / "pilot" / "canonical" / "index.json" for path in roots]
    index = json.loads(paths[0].read_text())
    if index != json.loads(paths[1].read_text()):
        raise ValueError("Canonical prompt or candidate token-path identity differs")
    expected = dict(freeze1, protocol_hash=fingerprint(manifests[1]))
    if freeze8 != expected or freeze1["protocol_hash"] != fingerprint(manifests[0]):
        raise ValueError("Frozen lens artifacts, software or protocol provenance differ")
    plan = freeze1["plan"]
    if (
        tuple(freeze1["labels"]) != tuple(EMOTIONS)
        or keys1 != keys8
        or len(set(keys1)) != len(keys1)
        or plan != freeze1["readouts"]["plan"]
        or plan["n_blocks"] != configs[0].n_blocks
        or plan != layer_plan(plan["coverage"], plan["n_blocks"])
    ):
        raise ValueError("Lens/layer identifiers or candidate labels do not match the frozen plan")
    arrays1, arrays8 = [
        {
            score: np.stack([value[score] for value in values])
            for score in ("label_scores", "stop_scores", "token_counts")
        }
        for values in (values1, values8)
    ]
    if not np.array_equal(arrays1["token_counts"], arrays8["token_counts"]):
        raise ValueError("Candidate label token counts differ")
    checks1 = read_checks(roots[0], diagnostic=False)
    checks8 = read_checks(roots[1], diagnostic=True)
    if any(
        check["pilot_samples"] != len(ids) or check["candidate_score_atol"] != configs[0].score_atol
        for check in (checks1, checks8)
    ):
        raise ValueError("Technical checks and paired protocol disagree")
    context = dict(
        comparison_id=root.name,
        model_id=configs[0].model_id,
        dataset_id=configs[0].dataset_id,
        bs1_run_id=run_ids[0],
        bs8_run_id=run_ids[1],
        variant="canonical",
        tolerance=configs[0].score_atol,
        bs1_prefix_batch_size=configs[0].prefix_batch_size,
        bs1_projection_batch_size=configs[0].projection_batch_size,
        bs8_prefix_batch_size=configs[1].prefix_batch_size,
        bs8_projection_batch_size=configs[1].projection_batch_size,
    )
    agreement, differences, flips = layer_tables(
        ids,
        keys1,
        arrays1,
        arrays8,
        normalized=configs[0].length_normalized,
        context=context,
    )
    runtime = pd.DataFrame(
        [
            dict(context, **_runtime(values1, checks1, configs[0])),
            dict(context, **_runtime(values8, checks8, configs[1])),
        ]
    )
    (root / "plot_data").mkdir(exist_ok=True)
    agreement.to_csv(root / "plot_data" / "layer_agreement.csv", index=False)
    differences.to_csv(root / "component_differences.csv", index=False)
    flips.to_csv(root / "prediction_flips.csv", index=False)
    runtime.to_csv(root / "runtime.csv", index=False)
    atomic_json(
        root / "comparison_manifest.json",
        dict(
            context,
            complete=True,
            gold_labels_used=False,
            sample_ids=ids,
            readouts=keys1,
            labels=EMOTIONS,
            configs=[cfg.to_dict() for cfg in configs],
            protocol_hashes=[fingerprint(manifest) for manifest in manifests],
            score_index_hash=fingerprint(index),
            bs1_candidate_score_agreement_passed=(
                checks1["shared_prefix_equivalence"] and checks1["individual_batch_equivalence"]
            ),
            bs8_candidate_score_agreement_passed=checks8["candidate_score_agreement_passed"],
        ),
    )
    get_logger().info("m03.batch_pilot.compared comments=%s readouts=%s", len(ids), len(keys1))
