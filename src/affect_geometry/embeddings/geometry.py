"""Strict EXP3 decomposition contracts, separate from legacy permissive defaults."""

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.csgraph import connected_components


class UnsupportedGeometry(ValueError):
    """A scientific cell is unsupported; the orchestrator must persist the reason."""


@dataclass
class Decomposition:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    weights: np.ndarray
    reference: np.ndarray
    arrays: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def coordinates(self, dimension: int, tolerance=1e-10):
        if dimension not in (1, 2):
            raise ValueError("EXP3 supports only 1D and 2D")
        order = np.asarray(self.metadata["mode_order"], dtype=int)
        if len(order) < dimension or np.any(self.weights[order[:dimension]] <= tolerance):
            raise UnsupportedGeometry("insufficient_positive_rank_or_diffusion_energy")
        chosen = order[:dimension]
        return self.eigenvectors[:, chosen] * np.sqrt(self.weights[chosen])


def validate_distances(distances):
    d = np.asarray(distances, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or len(d) < 3:
        raise UnsupportedGeometry("distance_matrix_needs_at_least_three_nodes")
    if not np.isfinite(d).all() or np.any(d < -1e-10):
        raise UnsupportedGeometry("invalid_distances")
    if not np.allclose(d, d.T) or not np.allclose(np.diag(d), 0):
        raise ValueError("Distances must be symmetric with zero diagonal")
    return np.maximum(d, 0)


def mds_decomposition(distances, tolerance=1e-10):
    d = validate_distances(distances)
    j = np.eye(len(d)) - np.ones_like(d) / len(d)
    b = -0.5 * j @ (d * d) @ j
    values, vectors = np.linalg.eigh((b + b.T) / 2)
    order = np.argsort(-values, kind="stable")
    values, vectors = values[order], vectors[:, order]
    return Decomposition(
        values,
        vectors,
        np.maximum(values, 0),
        d,
        {"gram": b},
        {"mode_order": np.flatnonzero(values > tolerance).tolist()},
    )


def neighborhood(distances, neighbors, ids):
    d = validate_distances(distances)
    if not 1 <= neighbors < len(d):
        raise UnsupportedGeometry("invalid_neighbor_count")
    if len(set(ids)) != len(d):
        raise ValueError("Unique emotion IDs required for stable ties")
    support = np.zeros_like(d, dtype=bool)
    for i in range(len(d)):
        candidates = [j for j in range(len(d)) if j != i]
        nearest = sorted(candidates, key=lambda j: (d[i, j], ids[j]))[:neighbors]
        support[i, nearest] = True
    support |= support.T
    components = connected_components(support.astype(int), directed=False)[0]
    if components != 1:
        raise UnsupportedGeometry(f"disconnected_graph:{components}")
    return support


def isomap_decomposition(distances, neighbors, ids, tolerance=1e-10):
    d = validate_distances(distances)
    support = neighborhood(d, neighbors, ids)
    adjacency = np.full_like(d, np.inf)
    adjacency[support] = d[support]
    np.fill_diagonal(adjacency, 0)
    # Floyd-Warshall preserves real zero-weight edges; dense scipy graphs drop them.
    geodesic = adjacency.copy()
    for k in range(len(d)):
        geodesic = np.minimum(geodesic, geodesic[:, k, None] + geodesic[None, k, :])
    result = mds_decomposition(geodesic, tolerance)
    laplacian = np.diag(support.sum(axis=1)) - support.astype(float)
    result.arrays.update(
        support=support, geodesics=geodesic, laplacian_eigenvalues=np.linalg.eigvalsh(laplacian)
    )
    density = float(support.sum() / (len(d) * (len(d) - 1)))
    result.metadata.update(
        neighbors=neighbors,
        graph_density=density,
        components=1,
        nearly_complete=density >= 0.9,
        algebraic_connectivity=float(np.linalg.eigvalsh(laplacian)[1]),
    )
    return result
