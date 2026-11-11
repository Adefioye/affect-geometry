"""Pure first-word classification diagnostics with explicit population scopes."""

import numpy as np
import pandas as pd

from affect_geometry.classification.protocol import EMOTIONS, parse_compat, parse_strict


def analyze_source(source):
    rows = source["rows"].copy()
    required = {
        "sample_id",
        "emotion",
        "raw_response",
        "compat_correct",
        "compat_predicted_emotion",
        "format_compliant",
        "status",
    }
    if required - set(rows):
        raise ValueError(f"Missing columns: {required - set(rows)}")
    if rows.sample_id.eq("").any() or rows.sample_id.duplicated().any():
        raise ValueError("Missing or duplicate sample IDs")
    if not rows.status.eq("completed").all():
        raise ValueError("Unexpected unresolved row in activation cohort")
    for col in ["compat_correct", "format_compliant", "compat_valid_label"]:
        if not rows[col].isin([True, False]).all():
            raise ValueError(f"{col} must contain booleans")
        rows[col] = rows[col].astype(bool)
    if not rows.emotion.isin(EMOTIONS).all():
        raise ValueError("Unknown gold labels")
    predicted = rows.raw_response.map(parse_compat).fillna("")
    checks = {
        "prediction": predicted.eq(rows.compat_predicted_emotion),
        "correctness": predicted.eq(rows.emotion).eq(rows.compat_correct),
        "validity": predicted.ne("").eq(rows.compat_valid_label),
        "format": rows.raw_response.map(parse_strict).notna().eq(rows.format_compliant),
    }
    for name, check in checks.items():
        if not check.all():
            raise ValueError(f"{name} disagrees with saved scoring on {(~check).sum()} rows")
    rows["incorrect"] = ~rows.compat_correct
    rows["unparseable"] = predicted.eq("")
    rows["strict_format_issue"] = ~rows.format_compliant
    rows["rescued_correct"] = rows.strict_format_issue & rows.compat_correct
    rows["format_issue_among_failures"] = rows.strict_format_issue & rows.incorrect
    rows["wrong_label_with_format_issue"] = rows.format_issue_among_failures & ~rows.unparseable
    rows["wrong_label"] = rows.incorrect & ~rows.unparseable
    rows["prediction"] = predicted.replace("", "[unparseable]")
    flags = [
        "incorrect",
        "unparseable",
        "strict_format_issue",
        "rescued_correct",
        "format_issue_among_failures",
        "wrong_label_with_format_issue",
        "wrong_label",
    ]
    per_gold = rows.groupby("emotion")[flags].sum().reindex(EMOTIONS)
    per_gold["n"] = rows.groupby("emotion").size().reindex(EMOTIONS, fill_value=0)
    per_gold["available"] = per_gold.n.gt(0)
    per_gold["incorrect_pct"] = 100 * per_gold.incorrect / per_gold.n.replace(0, np.nan)
    per_gold["unparseable_pct_of_failures"] = (
        100 * per_gold.unparseable / per_gold.incorrect.replace(0, np.nan)
    )
    per_gold = per_gold.rename_axis("gold").reset_index()
    failures = rows[rows.incorrect]
    index = pd.MultiIndex.from_product(
        [sorted(rows.emotion.unique()), EMOTIONS + ["[unparseable]"]], names=["gold", "prediction"]
    )
    distribution = (
        failures.groupby(["emotion", "prediction"])
        .size()
        .rename_axis(["gold", "prediction"])
        .reindex(index, fill_value=0)
        .rename("count")
        .reset_index()
    )
    totals = failures.groupby("emotion").size()
    valid_totals = failures[~failures.unparseable].groupby("emotion").size()
    distribution["n_failures"] = distribution.gold.map(totals).fillna(0)
    distribution["pct_of_failures"] = (
        100 * distribution["count"] / distribution.n_failures.replace(0, np.nan)
    )
    distribution["pct_of_wrong_labels"] = np.where(
        distribution.prediction.eq("[unparseable]"),
        np.nan,
        100 * distribution["count"] / distribution.gold.map(valid_totals).replace(0, np.nan),
    )
    wrong = distribution[
        (distribution.prediction != distribution.gold) & distribution.prediction.ne("[unparseable]")
    ]
    stats = wrong.groupby("gold")["count"].agg(["min", "max", "mean", "median"]).reset_index()
    population = source["support"].rename(columns={"emotion": "gold"}).copy()
    population["n"] = population.correct + population.incorrect
    population["incorrect_pct"] = 100 * population.incorrect / population.n.replace(0, np.nan)
    expected = population.set_index("gold").sampled_per_subset * 2
    observed = rows.groupby("emotion").size().reindex(expected.index, fill_value=0)
    if not observed.eq(expected).all():
        raise ValueError("Cohort rows disagree with saved support counts")
    if not rows.groupby("emotion").compat_correct.mean().eq(0.5).all():
        raise ValueError("Expected balanced activation cohort")
    for frame in [per_gold, distribution, stats, population]:
        for key, value in source["provenance"].items():
            frame[key] = value
    population["scope"] = "full_classification_population"
    return dict(
        per_gold=per_gold, distribution=distribution, stats=stats, population=population, rows=rows
    )


def count_summary(population):
    """Each observation is one gold label, not a numeric encoding of a response."""
    return population.groupby("model")[["incorrect", "incorrect_pct"]].agg(
        ["min", "max", "mean", "median"]
    )
