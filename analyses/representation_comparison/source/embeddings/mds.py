from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MDSResult:
    coordinates: np.ndarray
    eigenvalues: np.ndarray


def classical_mds(distances: np.ndarray, dimensions: int) -> MDSResult:
    """Classical metric MDS from a symmetric dissimilarity matrix."""
    distances = np.asarray(distances, dtype=np.float64)
    n = len(distances)
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (distances**2) @ centering
    gram = (gram + gram.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    positive = np.maximum(eigenvalues[:dimensions], 0.0)
    coordinates = eigenvectors[:, :dimensions] * np.sqrt(positive)
    if coordinates.shape[1] < dimensions:
        coordinates = np.pad(coordinates, ((0, 0), (0, dimensions - coordinates.shape[1])))
    return MDSResult(coordinates, eigenvalues)
