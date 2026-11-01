"""Locally scaled diffusion maps with stationary-inner-product normalized modes."""

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform

from .geometry import Decomposition, UnsupportedGeometry, neighborhood, validate_distances


def diffusion_maps(distances, neighbors, scale_rank, time, ids, alpha=1.0, tolerance=1e-10):
    d = validate_distances(distances)
    if type(time) is not int or time < 1 or alpha not in (0.0, 0.5, 1.0):
        raise ValueError("Invalid diffusion time or alpha")
    if not 1 <= scale_rank < len(d):
        raise UnsupportedGeometry("invalid_local_scale_rank")
    support = neighborhood(d, neighbors, ids)
    masked = d.copy()
    np.fill_diagonal(masked, np.inf)
    scales = np.sort(masked, axis=1)[:, scale_rank - 1]
    if np.any(scales <= tolerance):
        raise UnsupportedGeometry("zero_local_scale")
    w = np.exp(-(d**2) / np.outer(scales, scales)) * support
    np.fill_diagonal(w, 1.0)
    if connected_components((w > 0).astype(int), directed=False)[0] != 1:
        raise UnsupportedGeometry("disconnected_affinity_after_numerical_underflow")
    q = w.sum(axis=1)
    kernel = w / np.outer(q, q) ** alpha
    degrees = kernel.sum(axis=1)
    stationary = degrees / degrees.sum()
    conjugate = kernel / np.sqrt(np.outer(degrees, degrees))
    values, vectors = np.linalg.eigh(conjugate)
    order = np.argsort(-values, kind="stable")
    values, vectors = values[order], vectors[:, order]
    if not np.isclose(values[0], 1.0):
        raise UnsupportedGeometry("missing_stationary_mode")
    psi = vectors / np.sqrt(stationary[:, None])
    weights = np.abs(values) ** (2 * time)
    weights[0] = 0.0
    mode_order = sorted(range(1, len(d)), key=lambda i: (-weights[i], i))
    full = psi[:, 1:] * (values[1:] ** time)
    reference = squareform(pdist(full))
    # Preserve signed lambda**t coordinates, including odd-time negative modes.
    signed_psi = psi * np.where(values**time < 0, -1.0, 1.0)
    return Decomposition(
        values,
        signed_psi,
        weights,
        reference,
        dict(
            affinity=w,
            kernel=kernel,
            degrees=degrees,
            density=q,
            transition=kernel / degrees[:, None],
            stationary=stationary,
            markov_eigenvectors=psi,
            local_scales=scales,
            support=support,
        ),
        dict(
            mode_order=mode_order,
            time=time,
            alpha=alpha,
            neighbors=neighbors,
            scale_rank=scale_rank,
            mode_policy="descending_abs_lambda_power_then_eigenvalue_order",
            self_loop=1.0,
            graph_density=float(support.sum() / (len(d) * (len(d) - 1))),
        ),
    )
