import numpy as np
import pytest

from affect_geometry.m03.metrics import (
    GroupBootstrap,
    class_counts,
    intervals,
    neutral_transitions,
    paired,
    ranks_and_margins,
    summarize,
)
from affect_geometry.m03.rules import (
    BETA_GRID,
    LENSES,
    arms,
    contrast,
    dynamic,
    layer_plan,
    log_prob,
    nearest,
)


def test_predeclared_layer_mapping_and_complete_coverage():
    plan = layer_plan(
        dict(logit=list(range(42)), tuned=list(range(41)), jacobian=list(range(41))), 42
    )
    assert plan["headline_layer"] == 30  # 31/42 and 32/42 tie: choose the lower.
    assert plan["dynamic_layers"] == [26, 30, 35, 39]
    assert plan["upper_layer"] == 40
    assert plan["missing_layers"]["tuned"] == [41]
    assert nearest([1, 3], 0.75, 4) == 1


@pytest.mark.parametrize("bad", [[], [2, 1], [0, 0], [-1, 1], [0, 42]])
def test_invalid_coverage_is_rejected(bad):
    with pytest.raises(ValueError):
        layer_plan(dict(logit=list(range(42)), tuned=bad, jacobian=[0, 1]), 42)


def test_beta_zero_and_score_offset_invariance():
    rng = np.random.default_rng(9)
    final, early = rng.normal(size=(2, 5, 28))
    np.testing.assert_array_equal(contrast(final, early, 0), log_prob(final))
    np.testing.assert_allclose(
        contrast(final, early, 0.5), contrast(final + 100, early - 200, 0.5), atol=1e-12
    )
    f = np.full((1, 28), -100.0)
    e = f.copy()
    f[0, :2], e[0, :2] = [-0.9, -1.1], [-0.7, -2.3]
    assert np.argmax(e) == np.argmax(f) == 0
    assert np.argmax(contrast(f, e, 0.5)) == 1


def test_dynamic_cosine_selection_and_distance_beta():
    final = np.full((2, 28), -20.0)
    final[:, 0] = 0
    opposite = np.roll(final, 1, axis=-1)
    chosen = np.vstack([opposite[0], final[1]])
    panel = {5: chosen, 8: np.vstack([final[0], opposite[1]])}
    score, layer, beta = dynamic(final, panel, 10)
    assert layer.tolist() == [5, 8]
    np.testing.assert_allclose(beta, [0.5, 2])
    np.testing.assert_allclose(score[0], contrast(final[:1], chosen[:1], 0.5)[0])
    _, tied_layers, _ = dynamic(final, {8: final, 5: final}, 10)
    assert tied_layers.tolist() == [5, 5]


def test_all_arms_primary_six_mask_and_within_lens():
    plan = layer_plan(dict(logit=[0, 1, 2], tuned=[0, 1], jacobian=[0, 1]), 3)
    rng = np.random.default_rng(12)
    scores = {
        (r, layer): rng.normal(size=(4, 28))
        for r, layers in plan["coverage"].items()
        for layer in layers
    }
    scores[("native", 2)] = rng.normal(size=(4, 28)) * 5
    result = arms(scores, plan)
    assert sum(a.headline and a.kind != "baseline" for a in result) == 6
    assert sum(a.kind == "direct" for a in result) == 7
    assert sum(a.kind == "contrast" for a in result) == 6 * len(BETA_GRID)
    assert sum(a.kind == "within" for a in result) == 3
    assert sum(a.kind == "dynamic" for a in result) == 3
    for arm in result:
        if arm.kind == "masked":
            p = np.exp(log_prob(scores[("native", 2)]))
            np.testing.assert_array_equal(np.isfinite(arm.scores), p >= 0.1 * p.max(-1)[:, None])
        if arm.kind == "contrast" and arm.beta == 0:
            np.testing.assert_allclose(arm.scores, log_prob(scores[("native", 2)]))
    assert {a.lens for a in result if a.kind == "within"} == set(LENSES)


def test_stable_ties_ranks_and_margins():
    scores = np.zeros((3, 28))
    ranks, margin, top = ranks_and_margins(scores, np.array([0, 1, 27]))
    assert ranks.tolist() == [1, 2, 28]
    assert np.argmax(scores, axis=-1).tolist() == [0, 0, 0]
    np.testing.assert_array_equal(margin, 0)
    np.testing.assert_array_equal(top, 0)
    with pytest.raises(ValueError, match="finite"):
        log_prob(np.full((2, 28), np.nan))


def test_fixed_28_class_macro_f1_and_invalid_false_negatives():
    gold = np.array([0, 0, 1])
    prediction = np.array([0, -1, 1])
    result = summarize(gold, prediction)
    assert result["accuracy"] == 2 / 3
    assert result["macro_f1"] == pytest.approx((2 / 3 + 1) / 28)
    assert class_counts(gold, prediction)["recall"][0] == 0.5


def test_repairs_damage_identity_and_undefined_denominators():
    gold = np.array([0, 1, 2, 3, 4])
    baseline = np.array([0, -1, 1, 3, 2])
    prediction = np.array([1, 1, 2, 3, 0])
    values = paired(gold, prediction, baseline)
    assert values["repairs"] == 2 and values["damages"] == 1
    assert values["invalid_repairs"] == 1
    assert values["wrong_to_different_wrong"] == 1
    assert values["delta_accuracy_pp"] == 20
    assert paired(gold, gold, gold)["repair_rate"] is None
    assert paired(gold, gold, gold + 10)["damage_rate"] is None


def test_bootstrap_samples_duplicate_text_groups_and_is_paired():
    gold = np.array([0, 0, 1])
    prediction = np.array([0, 0, -1])
    bootstrap = GroupBootstrap(gold, ["same", "same", "other"], 100, 42)
    result = bootstrap.distribution(prediction)
    assert bootstrap.n_groups == 2
    assert set(np.round(result["accuracy"], 8)) <= {0, round(2 / 3, 8), 1}
    np.testing.assert_array_equal(
        result["accuracy"],
        GroupBootstrap(gold, ["same", "same", "other"], 100, 42).distribution(prediction)[
            "accuracy"
        ],
    )
    ci = intervals(result, {"native": result})
    assert ci["delta_accuracy_native_pp_ci_low"] == 0
    assert ci["delta_macro_f1_native_pp_ci_high"] == 0


def test_invalid_outputs_are_not_counted_as_non_neutral_emotions():
    values = neutral_transitions(
        np.array([27, 0, 27, 1, 2]), np.array([27, -1, 27, 1, -1]), np.array([-1, 27, 2, 27, 3])
    )
    assert values == dict(
        neutral_to_emotion=dict(n=1, repairs=1, damages=0),
        emotion_to_neutral=dict(n=1, repairs=1, damages=0),
    )
