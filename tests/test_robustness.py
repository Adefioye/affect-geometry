import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import trustworthiness

from affect_geometry.analysis.quality import neighbor_order, trustworthiness_continuity
from affect_geometry.analysis.robustness import (
    bh_adjust,
    leave_one_emotion_out,
    permutation_procrustes,
    procrustes_scores,
)
from affect_geometry.embeddings.isomap import geodesic_distances


def test_zero_distance_ties_exclude_self_and_match_sklearn():
    x = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    d = squareform(pdist(x))
    assert not (neighbor_order(d) == np.arange(len(d))[:, None]).any()
    y = np.random.default_rng(3).normal(size=(len(d), 2))
    trust, _ = trustworthiness_continuity(d, y, 2)
    assert 0 <= trust <= 1
    assert trustworthiness_continuity(d, x, 2) == (1.0, 1.0)
    # sklearn does not specify a stable tie order; compare on untied inputs.
    untied = squareform(pdist(np.random.default_rng(9).normal(size=(6, 4))))
    trust, _ = trustworthiness_continuity(untied, y, 2)
    assert np.isclose(trust, trustworthiness(untied, y, n_neighbors=2, metric="precomputed"))
    geo, k = geodesic_distances(d, 2)
    assert np.isfinite(geo).all()
    assert k >= 2


def test_similarity_validation_recovers_known_out_of_sample_transform():
    x = np.random.default_rng(3).normal(size=(12, 2))
    y = x @ np.array([[0.0, 1.0], [1.0, 0.0]]) * 2 + 7
    metrics, predictions, _ = leave_one_emotion_out(x, y)
    assert metrics["loo_joint_skill"] > 0.999999
    assert np.allclose(predictions, y)
    assert np.allclose(procrustes_scores(x[None], y), 1.0)


def test_shared_permutation_max_statistic_and_bh():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(3, 10, 2))
    target = x[0].copy()
    perms = np.stack([rng.permutation(10) for _ in range(99)])
    scores, p, pmax, null = permutation_procrustes(x, target, perms)
    assert np.isclose(scores[0], 1)
    assert null.shape == (3, 99)
    assert (pmax >= p).all()
    assert np.allclose(bh_adjust(np.array([0.01, 0.04, 0.2])), [0.03, 0.06, 0.2])
