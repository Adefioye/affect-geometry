"""Common/native fidelity, graph ratios and basic spectral diagnostics."""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

from affect_geometry.analysis.quality import trustworthiness_continuity, upper_triangle


def fidelity(reference, coordinates, neighborhoods):
    original = upper_triangle(reference)
    embedded = pdist(coordinates)
    denom = float(original @ original)
    scale = (
        float(original @ embedded / (embedded @ embedded)) if embedded @ embedded > 0 else np.nan
    )
    stress = np.sqrt(np.sum((original - scale * embedded) ** 2) / denom) if denom > 0 else np.nan
    disparities = IsotonicRegression().fit_transform(original, embedded)
    nonmetric = (
        np.sqrt(np.sum((embedded - disparities) ** 2) / np.sum(embedded**2))
        if np.sum(embedded**2) > 0
        else np.nan
    )
    result = dict(
        stress=float(stress),
        stress_scale=scale,
        nonmetric_stress=float(nonmetric),
        shepard_spearman=float(spearmanr(original, embedded).statistic),
    )
    for k in neighborhoods:
        if not 1 <= k < len(reference) / 2:
            result.update({f"trustworthiness_k{k}": np.nan, f"continuity_k{k}": np.nan})
            continue
        trust, continuity = trustworthiness_continuity(reference, coordinates, k)
        result.update({f"trustworthiness_k{k}": trust, f"continuity_k{k}": continuity})
    i, j = np.triu_indices(len(reference), 1)
    return result, pd.DataFrame(
        dict(i=i, j=j, reference_distance=original, embedded_distance=embedded)
    )


def spectrum(values, weights=None):
    values = np.asarray(values)
    weights = np.maximum(values, 0) if weights is None else np.asarray(weights)
    positive = np.sort(weights[weights > 1e-10])[::-1]
    total = positive.sum()
    result = dict(
        participation_ratio=float(total**2 / (positive @ positive)) if total else np.nan,
        negative_mass=float(np.abs(values[values < 0]).sum()),
        positive_mass=float(np.maximum(values, 0).sum()),
    )
    for threshold in (0.9, 0.95):
        result[f"dimensions_{int(100 * threshold)}"] = (
            int(np.searchsorted(np.cumsum(positive) / total, threshold) + 1) if total else np.nan
        )
    for dim in (1, 2):
        result[f"retained_energy_{dim}d"] = float(positive[:dim].sum() / total) if total else np.nan
    return result


def spectral_table(decomposition, tolerance=1e-8):
    values = decomposition.eigenvalues
    gaps = np.r_[np.abs(np.diff(values)), np.nan]
    weights = decomposition.weights
    order = np.argsort(-weights, kind="stable")
    cumulative = np.zeros(len(weights))
    if weights.sum() > 0:
        cumulative[order] = np.cumsum(weights[order]) / weights.sum()
    return pd.DataFrame(
        dict(
            mode=np.arange(len(values)),
            eigenvalue=values,
            weight=weights,
            eigengap=gaps,
            near_tie=gaps <= tolerance,
            cumulative_mass=cumulative,
        )
    )


def graph_diagnostics(direct, geodesics, labels):
    i, j = np.triu_indices(len(direct), 1)
    denominator = direct[i, j]
    ratio = np.divide(
        geodesics[i, j], denominator, out=np.full(len(i), np.nan), where=denominator > 1e-10
    )
    return pd.DataFrame(
        dict(
            emotion_i=np.asarray(labels)[i],
            emotion_j=np.asarray(labels)[j],
            direct=denominator,
            geodesic=geodesics[i, j],
            ratio=ratio,
        )
    )


def triangle_violations(d):
    from itertools import combinations

    count, total, maximum = 0, 0, 0.0
    for a, b, c in combinations(range(len(d)), 3):
        for i, j, k in ((a, b, c), (b, a, c), (c, a, b)):
            excess = d[j, k] - d[j, i] - d[i, k]
            count += excess > 1e-10
            total += 1
            maximum = max(maximum, excess)
    return dict(
        triangle_violation_fraction=count / total if total else np.nan,
        maximum_triangle_excess=float(maximum),
    )


def subspace_stability(decomposition, recompute, distances, tolerance=1e-8):
    """Near-tied eigenspace stability under three recorded tiny symmetric perturbations."""
    values = decomposition.eigenvalues
    groups, start = [], 0
    for i in range(1, len(values) + 1):
        if i == len(values) or abs(values[i] - values[i - 1]) > tolerance:
            if i - start > 1:
                groups.append(list(range(start, i)))
            start = i
    rows = []
    rng = np.random.default_rng(0)
    for repeat in range(3):
        noise = rng.normal(size=distances.shape)
        noise = (noise + noise.T) / 2
        perturbed = np.maximum(distances + 1e-7 * noise, 0)
        np.fill_diagonal(perturbed, 0)
        if not groups:
            break
        try:
            changed = recompute(perturbed)
            for group in groups:
                a = np.linalg.qr(decomposition.eigenvectors[:, group])[0]
                b = np.linalg.qr(changed.eigenvectors[:, group])[0]
                cosines = np.linalg.svd(a.T @ b, compute_uv=False)
                rows.append(
                    dict(
                        modes="|".join(map(str, group)),
                        repeat=repeat,
                        minimum_principal_cosine=float(cosines.min()),
                        projector_distance=float(np.linalg.norm(a @ a.T - b @ b.T)),
                        perturbation_scale=1e-7,
                        seed=0,
                        status="ok",
                        reason="",
                    )
                )
        except ValueError as exc:
            rows.append(
                dict(
                    modes="all_tied_groups",
                    repeat=repeat,
                    minimum_principal_cosine=np.nan,
                    projector_distance=np.nan,
                    perturbation_scale=1e-7,
                    seed=0,
                    status="unsupported",
                    reason=str(exc),
                )
            )
    return pd.DataFrame(rows)


def layerwise_metric_means(metrics, layers):
    """Fixed-setting, equal-layer summaries; missing cells invalidate an aggregate."""
    keys = ["cohort", "method", "dimension", "setting", "reference", "metric"]
    chosen = metrics.loc[
        metrics.metric.str.startswith("trustworthiness_")
        | metrics.metric.eq("geodesic_direct_ratio_mean")
    ]
    records = []
    for key, group in chosen.groupby(keys):
        valid = (
            len(group) == len(layers)
            and set(group.layer) == set(layers)
            and np.isfinite(group.value.to_numpy(dtype=float)).all()
        )
        records.append(
            dict(zip(keys, key, strict=True))
            | dict(
                mean_value=float(group.value.mean()) if valid else np.nan,
                layer_count=len(layers),
                completed_layers=int(group.value.notna().sum()),
                status="ok" if valid else "unsupported",
                reason="" if valid else "incomplete_layer_support",
            )
        )
    return pd.DataFrame(records)
