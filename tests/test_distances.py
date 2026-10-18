import numpy as np
import pytest

from affect_geometry.distances.metrics import DISTANCE_FUNCTIONS


@pytest.fixture
def groups() -> list[np.ndarray]:
    rng = np.random.default_rng(4)
    return [rng.normal(index * 2.0, 0.5, size=(24, 5)) for index in range(3)]


@pytest.mark.parametrize("name", sorted(DISTANCE_FUNCTIONS))
def test_rdm_invariants(name: str, groups: list[np.ndarray]) -> None:
    raw, distance = DISTANCE_FUNCTIONS[name](
        groups,
        seed=5,
        folds=4,
        projections=32,
        bandwidth_multipliers=[0.5, 1.0, 2.0],
    )
    assert raw.shape == (3, 3)
    assert np.allclose(raw, raw.T)
    assert np.allclose(np.diag(raw), 0)
    assert np.allclose(distance, distance.T)
    assert np.all(distance >= 0)
    assert np.allclose(np.diag(distance), 0)


def test_distribution_metrics_detect_variance_without_mean_shift() -> None:
    rng = np.random.default_rng(7)
    groups = [rng.normal(0, 0.2, (150, 3)), rng.normal(0, 2.0, (150, 3))]
    for name in ("energy", "rbf_mmd", "sliced_wasserstein"):
        _, distance = DISTANCE_FUNCTIONS[name](groups, seed=2, projections=64)
        assert distance[0, 1] > 0
