from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

from affect_geometry.analysis.quality import neighbor_order
from affect_geometry.embeddings.mds import MDSResult, classical_mds


@dataclass(frozen=True)
class IsomapResult:
    coordinates: np.ndarray
    eigenvalues: np.ndarray
    geodesic_distances: np.ndarray
    neighbors_used: int


def geodesic_distances(distances: np.ndarray, neighbors: int) -> tuple[np.ndarray, int]:
    """Build a union-symmetric kNN graph, increasing k until it is connected."""
    distances = np.asarray(distances, dtype=np.float64)
    n = len(distances)
    if n < 2:
        return distances.copy(), 0
    ordered_neighbors = neighbor_order(distances)
    for k in range(min(max(1, neighbors), n - 1), n):
        adjacency = np.full((n, n), np.inf)
        np.fill_diagonal(adjacency, 0.0)
        for i in range(n):
            nearest = ordered_neighbors[i, :k]
            for j in nearest:
                weight = max(float(distances[i, j]), np.finfo(float).eps)
                adjacency[i, j] = min(adjacency[i, j], weight)
                adjacency[j, i] = min(adjacency[j, i], weight)
        graph = csr_matrix(np.where(np.isfinite(adjacency), adjacency, 0.0))
        components, _ = connected_components(graph, directed=False)
        if components == 1:
            return shortest_path(graph, directed=False), k
    raise RuntimeError("Could not construct a connected Isomap graph")


def isomap(distances: np.ndarray, dimensions: int, neighbors: int = 5) -> IsomapResult:
    geodesic, used = geodesic_distances(distances, neighbors)
    result: MDSResult = classical_mds(geodesic, dimensions)
    return IsomapResult(result.coordinates, result.eigenvalues, geodesic, used)
