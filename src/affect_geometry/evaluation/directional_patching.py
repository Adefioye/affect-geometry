"""Donor-oriented logit recovery and literal target-directed sign flips."""

import numpy as np
import pandas as pd

from affect_geometry.logging import progress


def evaluate_pairs(patcher, pairs, direction, layer, *, random_count=5, seed=42):
    rng = np.random.default_rng(seed)
    controls = {"pc1": direction, "no_op": None, "full": None}
    for i in range(random_count):
        d = rng.standard_normal(len(direction))
        controls[f"random_{i}"] = d / np.linalg.norm(d)
    records = []
    # Each baseline is computed once per case, shared across all controls.
    for case in progress(pairs.to_dict("records"), description="Directional patching", unit="case"):
        recipient_delta, _ = patcher.run(case["recipient_prompt"], layer)
        donor_delta, donor = patcher.run(case["donor_prompt"], layer)
        sign = 2 * case["donor_label"] - 1
        recipient_margin, donor_margin = sign * recipient_delta, sign * donor_delta
        denom = donor_margin - recipient_margin
        for control, vector in controls.items():
            if control == "no_op":
                patched_delta = recipient_delta
            else:
                patched_delta, _ = patcher.run(
                    case["recipient_prompt"],
                    layer,
                    donor=donor,
                    direction=vector,
                    full=control == "full",
                )
            patched_margin = sign * patched_delta
            records.append(
                {
                    **case,
                    "control": control,
                    "layer": layer,
                    "recipient_logit_diff": recipient_delta,
                    "donor_logit_diff": donor_delta,
                    "patched_logit_diff": patched_delta,
                    "recipient_margin": recipient_margin,
                    "donor_margin": donor_margin,
                    "patched_margin": patched_margin,
                    "denominator": denom,
                    "recovery_percent": 100 * (patched_margin - recipient_margin) / denom
                    if abs(denom) > 1e-8
                    else np.nan,
                    "target_sign_flip": recipient_delta * patched_delta < 0 and patched_margin > 0,
                    "both_baselines_correct": recipient_margin < 0 and donor_margin > 0,
                    "recipient_correct": recipient_margin < 0,
                    "donor_correct": donor_margin > 0,
                    "patched_target_correct": patched_margin > 0,
                }
            )
    return pd.DataFrame(records)


def summarize(records):
    summaries = []
    for control, rows in records.groupby("control", sort=False):
        for subset, mask in (
            ("all_pairs", np.ones(len(rows), dtype=bool)),
            ("both_baselines_correct", rows.both_baselines_correct.to_numpy()),
        ):
            frame = rows.loc[mask]
            denominator = frame.denominator.mean()
            gain = (frame.patched_margin - frame.recipient_margin).mean()
            sign = 2 * frame.donor_label - 1
            centered_accuracies = []
            for column in ("recipient_logit_diff", "donor_logit_diff", "patched_logit_diff"):
                centered = frame[column] - frame[column].mean()
                centered_accuracies.append(float((sign * centered > 0).mean()))
            base_acc, donor_acc, patched_acc = centered_accuracies
            accuracy_gap = donor_acc - base_acc
            calibrated_flip = (
                100 * (patched_acc - base_acc) / accuracy_gap
                if (np.isfinite(accuracy_gap) and abs(accuracy_gap) > 1e-8)
                else np.nan
            )
            summaries.append(
                dict(
                    control=control,
                    subset=subset,
                    n_cases=len(frame),
                    n_pairs=frame.pair_id.nunique(),
                    recovery_percent=100 * gain / denominator
                    if np.isfinite(denominator) and abs(denominator) > 1e-8
                    else np.nan,
                    mean_case_recovery_percent=frame.recovery_percent.mean(),
                    calibrated_logit_flip_percent=calibrated_flip,
                    sign_flip_percent=100 * frame.target_sign_flip.mean(),
                    patched_target_accuracy=frame.patched_target_correct.mean(),
                    recipient_accuracy=frame.recipient_correct.mean(),
                    donor_accuracy=frame.donor_correct.mean(),
                    near_zero_denominators=int(frame.denominator.abs().le(1e-8).sum()),
                )
            )
    return pd.DataFrame(summaries)
