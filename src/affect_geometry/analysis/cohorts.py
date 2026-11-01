"""RQ3's single balanced draw; gold labels, never predicted labels, define nodes."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Cohorts:
    samples: pd.DataFrame
    support: pd.DataFrame
    unresolved: pd.DataFrame
    counts: pd.DataFrame
    labels: tuple[str, ...]


def balanced_cohorts(rows: pd.DataFrame, labels, seed: int) -> Cohorts:
    required = {"sample_id", "emotion", "status", "compat_correct", "source_split"}
    if required - set(rows):
        raise ValueError(f"Missing cohort columns: {required - set(rows)}")
    if rows.sample_id.isna().any() or rows.sample_id.duplicated().any():
        raise ValueError("Sample IDs must be unique and non-null")
    if not set(rows.emotion) <= set(labels):
        raise ValueError("Unknown gold emotion")
    resolved = rows.status.eq("completed") & rows.compat_correct.notna()
    if not rows.loc[resolved, "compat_correct"].isin([True, False]).all():
        raise ValueError("Compatibility correctness must be boolean")
    completed = rows.loc[resolved].sort_values("sample_id").copy()
    completed["cohort"] = np.where(completed.compat_correct, "correct", "incorrect")
    rng = np.random.default_rng(seed)
    support, selected = [], []
    for label in sorted(labels):
        pools = [
            completed.loc[completed.emotion.eq(label) & completed.cohort.eq(c)]
            for c in ("correct", "incorrect")
        ]
        nc, ni = map(len, pools)
        n = min(nc, ni) if nc > 100 else 0
        reason = (
            "correct_count_not_greater_than_100"
            if nc <= 100
            else ("no_incorrect_support" if ni == 0 else "")
        )
        support.append(
            dict(
                emotion=label,
                correct=nc,
                incorrect=ni,
                sampled_per_subset=n,
                eligible=nc > 100,
                status="unsupported" if reason else "ok",
                reason=reason,
            )
        )
        if n:
            selected.extend(p.iloc[np.sort(rng.choice(len(p), n, replace=False))] for p in pools)
    subsets = pd.concat(selected, ignore_index=True) if selected else completed.iloc[:0].copy()
    union = subsets.assign(cohort="all")
    sampled = pd.concat([subsets, union], ignore_index=True)
    for label in sampled.emotion.unique():
        part = sampled.loc[sampled.emotion.eq(label)]
        ids = {
            c: set(part.loc[part.cohort.eq(c), "sample_id"])
            for c in ("correct", "incorrect", "all")
        }
        if ids["correct"] & ids["incorrect"] or ids["correct"] | ids["incorrect"] != ids["all"]:
            raise AssertionError("Cohort partition failed")
        if len(ids["correct"]) != len(ids["incorrect"]):
            raise AssertionError("Cohort balance failed")
    # Include zero cells so exclusions remain visible in diagnostics.
    index = pd.MultiIndex.from_product(
        [("correct", "incorrect", "all"), sorted(rows.source_split.unique()), sorted(labels)],
        names=["cohort", "source_split", "emotion"],
    )
    before = pd.concat([completed, completed.assign(cohort="all")])
    tables = []
    for phase, frame in [("before", before), ("after", sampled)]:
        counts = frame.groupby(list(index.names)).size().reindex(index, fill_value=0)
        tables.append(counts.rename("n_samples").reset_index().assign(phase=phase))
    return Cohorts(
        sampled,
        pd.DataFrame(support),
        rows.loc[~resolved].copy(),
        pd.concat(tables, ignore_index=True),
        tuple(sorted(subsets.emotion.unique())),
    )
