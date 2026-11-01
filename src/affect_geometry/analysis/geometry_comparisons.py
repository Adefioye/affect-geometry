"""Descriptive paired contrasts, fixed-layer aggregates, and selected-layer CKA."""

from itertools import combinations

import numpy as np
import pandas as pd

METHODS = ("mds", "isomap", "diffusion_maps")
COHORTS = ("correct", "incorrect", "all")


def linear_cka(x, y):
    x, y = x - x.mean(0), y - y.mean(0)
    denominator = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    return float(np.linalg.norm(x.T @ y) ** 2 / denominator) if denominator > 1e-10 else np.nan


def aggregate_scores(scores, layers):
    rows = []
    for (cohort, method, dimension), group in scores.groupby(["cohort", "method", "dimension"]):
        valid = (
            set(group.layer) == set(layers)
            and len(group) == len(layers)
            and group.cv_r2.notna().all()
            and group.status.eq("ok").all()
        )
        rows.append(
            dict(
                cohort=cohort,
                method=method,
                dimension=dimension,
                A_M=float(group.cv_r2.mean()) if valid else np.nan,
                layer_count=len(layers),
                completed_layers=int(group.cv_r2.notna().sum()),
                status="ok" if valid else "unsupported",
                reason="" if valid else "incomplete_fixed_layer_support",
            )
        )
    result = pd.DataFrame(rows)
    result["delta_A_M"] = np.nan
    for idx, row in result.iterrows():
        baseline = result.loc[
            result.cohort.eq(row.cohort)
            & result.dimension.eq(row.dimension)
            & result.method.eq("mds"),
            "A_M",
        ]
        if len(baseline):
            result.loc[idx, "delta_A_M"] = row.A_M - baseline.iloc[0]
    return result


def selected_cka(scores, coordinates):
    rows = []
    for cohort in COHORTS:
        for dimension in (1, 2):
            best = {}
            for method in METHODS:
                group = scores.loc[
                    scores.cohort.eq(cohort)
                    & scores.dimension.eq(dimension)
                    & scores.method.eq(method)
                ]
                # No best-layer selection from an incomplete prespecified layer panel.
                if len(group) and group.status.eq("ok").all() and group.cv_r2.notna().all():
                    best[method] = group.sort_values(
                        ["cv_r2", "layer"], ascending=[False, True]
                    ).iloc[0]
            for left in METHODS:
                for right in METHODS:
                    a, b = best.get(left), best.get(right)
                    value = np.nan
                    if a is not None and b is not None:
                        value = linear_cka(
                            coordinates[(cohort, left, int(a.layer), dimension)],
                            coordinates[(cohort, right, int(b.layer), dimension)],
                        )
                    rows.append(
                        dict(
                            cohort=cohort,
                            dimension=dimension,
                            method_left=left,
                            method_right=right,
                            cka=value,
                            layer_left=None if a is None else int(a.layer),
                            layer_right=None if b is None else int(b.layer),
                            setting_left=None if a is None else a.setting,
                            setting_right=None if b is None else b.setting,
                            score_left=None if a is None else a.cv_r2,
                            score_right=None if b is None else b.cv_r2,
                            status="ok" if np.isfinite(value) else "unsupported",
                            reason="" if np.isfinite(value) else "incomplete_or_degenerate",
                        )
                    )
    return pd.DataFrame(rows)


def contrasts(scores, predictions):
    rows, errors = [], []
    for dimension in (1, 2):
        for layer in sorted(scores.layer.unique()):
            part = scores.loc[scores.dimension.eq(dimension) & scores.layer.eq(layer)]
            pairs = [((c, m), (c, "mds")) for c in COHORTS for m in METHODS[1:]]
            pairs += [((a, m), (b, m)) for a, b in combinations(COHORTS, 2) for m in METHODS]
            for left, right in pairs:
                a = part.loc[part.cohort.eq(left[0]) & part.method.eq(left[1])].iloc[0]
                b = part.loc[part.cohort.eq(right[0]) & part.method.eq(right[1])].iloc[0]
                context = dict(
                    dimension=dimension,
                    layer=int(layer),
                    cohort_left=left[0],
                    method_left=left[1],
                    cohort_right=right[0],
                    method_right=right[1],
                )
                difference = a.cv_r2 - b.cv_r2
                rows.append(
                    dict(
                        **context,
                        delta_r2=difference,
                        status="ok" if np.isfinite(difference) else "unsupported",
                        reason="" if np.isfinite(difference) else "missing_paired_score",
                    )
                )
                ka, kb = (*left, int(layer), dimension), (*right, int(layer), dimension)
                if ka in predictions and kb in predictions:
                    for i, (ea, eb) in enumerate(
                        zip(predictions[ka]["errors"], predictions[kb]["errors"], strict=True)
                    ):
                        errors.append(
                            dict(
                                **context,
                                emotion_index=i,
                                error_left=ea,
                                error_right=eb,
                                delta_error=ea - eb,
                                status="ok",
                                reason="",
                            )
                        )
    return pd.DataFrame(rows), pd.DataFrame(errors)


def emergence(scores, minimum_interval=3, transition_threshold=0.1):
    rows = []
    for keys, group in scores.groupby(["cohort", "method", "dimension"]):
        group = group.sort_values("layer")
        valid = group.status.eq("ok").all() and group.cv_r2.notna().all()
        significant = group.corrected_p.le(0.05).to_numpy()
        intervals, start = [], None
        for i, flag in enumerate(np.r_[significant, False]):
            if flag and start is None:
                start = i
            if not flag and start is not None:
                if i - start >= minimum_interval:
                    intervals.append([int(group.iloc[start].layer), int(group.iloc[i - 1].layer)])
                start = None
        peak = group.sort_values(["cv_r2", "layer"], ascending=[False, True]).iloc[0]
        changes = group.cv_r2.diff()
        rows.append(
            dict(
                cohort=keys[0],
                method=keys[1],
                dimension=keys[2],
                first_significant_layer=(
                    int(group.loc[group.corrected_p.le(0.05), "layer"].min())
                    if significant.any()
                    else None
                ),
                descriptive_peak_layer=int(peak.layer) if valid else None,
                stable_significant_intervals=intervals,
                abrupt_transition_layers=group.loc[
                    changes.abs().ge(transition_threshold), "layer"
                ].tolist(),
                decline_layers=group.loc[changes.le(-transition_threshold), "layer"].tolist(),
                status="ok" if valid else "unsupported",
            )
        )
    return rows
