"""Guard against mixing strict failures, relaxed failures and missing support."""

import pandas as pd
import pytest

from affect_geometry.analysis.compat_exploration import analyze_source
from affect_geometry.classification.protocol import score


def fixture_source():
    cases = [
        ("joy extra words", "joy"),
        ("sadness extra words", "joy"),
        ("sadness", "sadness"),
        ("the emotion is sadness", "sadness"),
    ]
    rows = pd.DataFrame(
        [
            dict(
                sample_id=str(i),
                emotion=gold,
                raw_response=raw,
                status="completed",
                **score(raw, gold),
            )
            for i, (raw, gold) in enumerate(cases)
        ]
    )
    rows["compat_predicted_emotion"] = rows.compat_predicted_emotion.fillna("")
    support = pd.DataFrame(
        dict(
            emotion=["joy", "sadness", "pride"],
            correct=[200, 300, 10],
            incorrect=[100, 200, 90],
            sampled_per_subset=[1, 1, 0],
        )
    )
    return dict(rows=rows, support=support, provenance={"model": "fixture"})


def test_relaxed_and_formatting_categories():
    result = analyze_source(fixture_source())
    labels = result["per_gold"].set_index("gold")
    assert labels.loc["joy", "rescued_correct"] == 1
    assert labels.loc["joy", "wrong_label_with_format_issue"] == 1
    assert labels.loc["sadness", "unparseable"] == 1
    assert not labels.loc["pride", "available"]
    assert pd.isna(labels.loc["pride", "incorrect"])
    dist = result["distribution"]
    assert dist.groupby("gold")["count"].sum().to_dict() == {"joy": 1, "sadness": 1}
    assert result["stats"].set_index("gold").loc["joy", "mean"] == pytest.approx(1 / 27)
    assert result["population"].incorrect.sum() == 390


def test_saved_metric_mismatch_is_rejected():
    source = fixture_source()
    source["rows"].loc[0, "compat_correct"] = False
    with pytest.raises(ValueError, match="correctness disagrees"):
        analyze_source(source)


def test_duplicate_sample_is_rejected():
    source = fixture_source()
    source["rows"].loc[1, "sample_id"] = "0"
    with pytest.raises(ValueError, match="duplicate"):
        analyze_source(source)
