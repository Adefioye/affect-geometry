"""Scientific checks for the additional three-axis evaluation."""

import numpy as np

from affect_geometry.analysis.projected_audit import _axis_metrics


def test_three_axis_similarity_and_loo_recover_known_geometry():
    rng = np.random.default_rng(1701)
    source = rng.normal(size=(28, 3))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    target = 2.7 * source @ rotation + np.array([5, -2, 3])
    rows, fitted, predicted, skill = _axis_metrics(
        source, target, ["valence", "arousal", "dominance"]
    )
    np.testing.assert_allclose(fitted, target, atol=1e-12)
    np.testing.assert_allclose(predicted, target, atol=1e-12)
    assert np.isclose(skill, 1)
    assert [r["axis"] for r in rows] == ["valence", "arousal", "dominance"]
    assert all(np.isclose(row["loo_skill"], 1) for row in rows)
    assert np.isclose(sum(row["target_variance_share"] for row in rows), 1)


def test_shared_scale_distinguishes_correlation_from_axis_recovery():
    # Orthogonal axes with unequal target scales: a joint fit cannot match both.
    angles = np.linspace(0, 2 * np.pi, 28, endpoint=False)
    source = np.column_stack([np.cos(angles), np.sin(angles)])
    target = source * [10, 0.1]
    rows, _, _, _ = _axis_metrics(source, target, ["valence", "arousal"])
    assert rows[1]["fitted_pearson"] > 0.999
    assert rows[1]["fitted_r2"] < 0
    assert rows[1]["loo_skill"] < 0
