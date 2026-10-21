"""Descriptive diagnostics for a completed representation comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from affect_geometry.analysis.va import align_similarity


def comparison_tables(
    layerwise: pd.DataFrame, coordinates: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    scores = layerwise.pivot(index="layer", columns="variant", values="va_procrustes_r2")
    summary = []
    phases = []
    for variant, series in scores.items():
        summary.append(
            {
                "variant": variant,
                "mean_r2": series.mean(),
                "median_r2": series.median(),
                "min_r2": series.min(),
                "max_r2": series.max(),
                "peak_layer": int(series.idxmax()),
                "final_r2": series.iloc[-1],
                "winning_layers": int((scores.idxmax(axis=1) == variant).sum()),
            }
        )
        for phase, start, end in [
            ("Early (0–8)", 0, 8),
            ("Middle (9–18)", 9, 18),
            ("Late (19–27)", 19, 27),
        ]:
            values = series.loc[start:end]
            phases.append(
                {
                    "variant": variant,
                    "phase": phase,
                    "start_layer": start,
                    "end_layer": end,
                    "layers": len(values),
                    "mean_r2": values.mean(),
                }
            )
    contrasts = []
    for name, candidate, baseline in [
        ("LT: centering − raw", "LT-Centered", "LT-Raw"),
        ("LT: projection − centering", "LT-NeutralProjected", "LT-Centered"),
        ("Mean: centering − raw", "Mean-Centered", "Mean-Raw"),
        ("Mean: projection − centering", "Mean-NeutralProjected", "Mean-Centered"),
        ("Mean: projection − raw", "Mean-NeutralProjected", "Mean-Raw"),
        ("LT − Mean: raw", "LT-Raw", "Mean-Raw"),
        ("LT − Mean: centered", "LT-Centered", "Mean-Centered"),
        ("LT − Mean: projected", "LT-NeutralProjected", "Mean-NeutralProjected"),
    ]:
        for layer, delta in (scores[candidate] - scores[baseline]).items():
            contrasts.append(
                {
                    "contrast": name,
                    "candidate": candidate,
                    "baseline": baseline,
                    "layer": layer,
                    "delta_r2": delta,
                }
            )
    contrasts = pd.DataFrame(contrasts)
    contrast_summary = (
        contrasts.groupby(["contrast", "candidate", "baseline"], sort=False)
        .agg(
            mean_delta=("delta_r2", "mean"),
            median_delta=("delta_r2", "median"),
            min_delta=("delta_r2", "min"),
            max_delta=("delta_r2", "max"),
            positive_layers=("delta_r2", lambda x: int((x > 0).sum())),
        )
        .reset_index()
    )
    axis_scores, errors = [], []
    for (variant, layer), frame in coordinates.groupby(["variant", "layer"]):
        record = {"variant": variant, "layer": layer}
        non_neutral = frame.loc[frame.emotion != "neutral"]
        _, record["non_neutral_refit_r2"] = align_similarity(
            non_neutral[["coordinate_1", "coordinate_2"]].to_numpy(),
            non_neutral[["valence", "arousal"]].to_numpy(),
        )
        for axis in ("valence", "arousal"):
            target, aligned = frame[axis].to_numpy(), frame[f"aligned_{axis}"].to_numpy()
            sse = np.sum((target - aligned) ** 2)
            record[f"{axis}_r2"] = 1 - sse / np.sum((target - target.mean()) ** 2)
            record[f"{axis}_r"] = np.corrcoef(target, aligned)[0, 1]
        axis_scores.append(record)
        current = frame.copy()
        current["squared_error"] = (frame.valence - frame.aligned_valence) ** 2 + (
            frame.arousal - frame.aligned_arousal
        ) ** 2
        current["error_distance"] = np.sqrt(current.squared_error)
        current["sse_share"] = current.squared_error / current.squared_error.sum()
        errors.append(current)
    return {
        "summary": pd.DataFrame(summary).sort_values("mean_r2", ascending=False),
        "phases": pd.DataFrame(phases),
        "contrasts": contrasts,
        "contrast_summary": contrast_summary,
        "axis_scores": pd.DataFrame(axis_scores),
        "emotion_errors": pd.concat(errors, ignore_index=True),
    }


def geometry_diagnostics(
    vectors: np.ndarray, distances: np.ndarray, eigenvalues: np.ndarray, components: np.ndarray
) -> dict[str, float]:
    """Measure MDS spectral concentration and energy in the removed neutral subspace."""
    positive = np.maximum(eigenvalues, 0)
    pair_distances = distances[np.triu_indices(len(distances), 1)]
    return {
        "mds_top1_positive_share": float(positive[0] / positive.sum()),
        "mds_top2_positive_share": float(positive[:2].sum() / positive.sum()),
        "mds_negative_absolute_share": float(
            -eigenvalues[eigenvalues < 0].sum() / np.abs(eigenvalues).sum()
        ),
        "neutral_subspace_energy_share": float(
            np.sum((vectors @ components.T) ** 2) / np.sum(vectors**2)
        ),
        "median_cosine_distance": float(np.median(pair_distances)),
        "distance_q25": float(np.quantile(pair_distances, 0.25)),
        "distance_q75": float(np.quantile(pair_distances, 0.75)),
        "mean_vector_norm": float(np.linalg.norm(vectors, axis=1).mean()),
    }
