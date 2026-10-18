import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from affect_geometry.analysis.quality import metric_stress, trustworthiness_continuity
from affect_geometry.analysis.va import va_recovery
from affect_geometry.embeddings.isomap import isomap
from affect_geometry.embeddings.mds import classical_mds


def test_classical_mds_recovers_euclidean_geometry() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    distances = squareform(pdist(points))
    result = classical_mds(distances, 2)
    assert metric_stress(distances, result.coordinates) < 1e-10
    trust, continuity = trustworthiness_continuity(distances, result.coordinates, 1)
    assert trust == 1.0
    assert continuity == 1.0


def test_isomap_connects_sparse_curve() -> None:
    angle = np.linspace(0, np.pi, 12)
    points = np.column_stack([np.cos(angle), np.sin(angle)])
    result = isomap(squareform(pdist(points)), 2, neighbors=2)
    assert np.isfinite(result.geodesic_distances).all()
    assert result.neighbors_used >= 2


def test_va_recovery_is_invariant_to_similarity_transform() -> None:
    target = np.array([[-1.0, -0.5], [-0.2, 1.0], [0.8, 0.6], [1.0, -0.8]])
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    source = target @ rotation * 3.0 + np.array([8.0, -2.0])
    reference = pd.DataFrame(
        {"emotion": list("abcd"), "valence": target[:, 0], "arousal": target[:, 1]}
    )
    metrics, aligned = va_recovery(list("abcd"), source, reference)
    assert metrics["va_procrustes_r2"] > 0.999999
    assert np.allclose(aligned[["aligned_x", "aligned_y"]], target)
