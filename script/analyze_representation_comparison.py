#!/usr/bin/env python3
"""Audit completed comparison artifacts and render supplementary report figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from affect_geometry.analysis.representation_audit import comparison_tables, geometry_diagnostics
from affect_geometry.analysis.representation_comparison import cosine_rdm
from affect_geometry.analysis.va import align_similarity
from affect_geometry.logging import configure_file_logging, get_logger, progress
from affect_geometry.reporting.representation_audit import plot_representation_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/representation_comparison"))
    args = parser.parse_args()
    root = args.root
    provenance = json.loads((root / "analyses/provenance.json").read_text())
    configure_file_logging(
        f"logs/qwen--qwen3-0.6b__representation-audit-{provenance['mean_run_id']}",
        "analyze_representation_comparison",
        root=root,
    )
    logger = get_logger()
    tables_dir = root / "analyses/report"
    plot_data = root / "plot_data/report"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plot_data.mkdir(parents=True, exist_ok=True)
    layerwise = pd.read_csv(root / "analyses/layerwise_procrustes.csv")
    coordinates = pd.read_csv(root / "analyses/aligned_coordinates.csv")
    assert len(layerwise) == 6 * provenance["layers"]
    assert not layerwise.duplicated(["variant", "layer"]).any()
    assert len(coordinates) == len(layerwise) * 28
    tables = comparison_tables(layerwise, coordinates)
    context = {
        k: provenance[k]
        for k in [
            "model",
            "last_token_run_id",
            "mean_run_id",
            "samples",
            "hidden_size",
            "mean_model_revision",
        ]
    }
    context.update(
        distance="cosine",
        embedding="classical_mds",
        embedding_dimensions=2,
        va_reference=provenance["va_reference"],
        neutral_variance_threshold=0.5,
        last_token_model_revision="unknown",
    )

    def write_plot(frame, name):
        frame.assign(**context).to_csv(plot_data / f"{name}.csv", index=False)

    diagnostics, neutral, pairs = [], [], []
    max_rdm_error = max_r2_error = 0.0
    for row in progress(
        layerwise.itertuples(index=False),
        total=len(layerwise),
        description="Auditing saved representation geometries",
        unit="variant-layer",
    ):
        variant, layer = row.variant, row.layer
        vectors_file = np.load(root / f"representations/{variant}/layer_{layer:03d}.npz")
        vectors, labels = vectors_file["vectors"], vectors_file["labels"]
        distances = np.load(root / f"rdms/{variant}/layer_{layer:03d}_cosine.npz")["distance"]
        eigenvalues = np.load(root / f"embeddings/{variant}/layer_{layer:03d}_eigenvalues.npz")[
            "eigenvalues"
        ]
        checkpoint = np.load(root / f"checkpoints/neutral_pca/{row.pooling}/layer_{layer:03d}.npz")
        assert vectors.shape == (28, provenance["hidden_size"])
        assert np.isfinite(vectors).all() and np.isfinite(distances).all()
        current = coordinates[(coordinates.variant == variant) & (coordinates.layer == layer)]
        assert current.emotion.tolist() == labels.tolist()
        _, score = align_similarity(
            current[["coordinate_1", "coordinate_2"]].to_numpy(),
            current[["valence", "arousal"]].to_numpy(),
        )
        max_r2_error = max(max_r2_error, abs(score - row.va_procrustes_r2))
        max_rdm_error = max(max_rdm_error, float(np.max(np.abs(cosine_rdm(vectors) - distances))))
        diagnostics.append(
            {
                "variant": variant,
                "layer": layer,
                "pooling": row.pooling,
                **geometry_diagnostics(vectors, distances, eigenvalues, checkpoint["components"]),
            }
        )
        if row.transformation == "neutral_projected":
            neutral.append(
                {
                    "pooling": row.pooling,
                    "layer": layer,
                    "neutral_components_fitted": row.neutral_components_fitted,
                    "neutral_variance_explained": row.neutral_variance_explained,
                    "neutral_pc1_variance_share": float(checkpoint["explained_variance_ratio"][0]),
                    "neutral_samples": int(checkpoint["neutral_samples"]),
                }
            )
        for i, j in zip(*np.triu_indices(len(labels), 1), strict=True):
            pairs.append(
                {
                    "variant": variant,
                    "layer": layer,
                    "emotion_a": labels[i],
                    "emotion_b": labels[j],
                    "cosine_distance": distances[i, j],
                }
            )
    assert max_rdm_error < 1e-10 and max_r2_error < 1e-10
    tables["geometry"] = pd.DataFrame(diagnostics)
    tables["neutral"] = pd.DataFrame(neutral)
    source = Path(provenance["last_token_run"]) / "dataset/samples.csv"
    mean_source = Path(provenance["mean_run"]) / "dataset/samples.csv"
    assert source.read_bytes() == mean_source.read_bytes()
    samples = pd.read_csv(source)
    counts = (
        samples.groupby(["split", "emotion"], as_index=False)
        .size()
        .rename(columns={"size": "sample_count"})
    )
    tables["sample_counts"] = counts
    tables["layerwise_scores"] = layerwise.pivot(
        index="layer", columns="variant", values="va_procrustes_r2"
    ).reset_index()
    peaks = tables["summary"][["variant", "peak_layer"]].rename(columns={"peak_layer": "layer"})
    best_coordinates = coordinates.merge(peaks, on=["variant", "layer"])
    best_errors = tables["emotion_errors"].merge(peaks, on=["variant", "layer"])
    best_axes = tables["axis_scores"].merge(peaks, on=["variant", "layer"])
    tables["best_layer_axes"] = best_axes
    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)
    write_plot(layerwise, "scores")
    write_plot(tables["summary"], "summary")
    for name in ["contrasts", "geometry", "neutral", "sample_counts"]:
        write_plot(tables[name], name)
    write_plot(best_coordinates, "selected_coordinates")
    write_plot(best_errors, "selected_errors")
    write_plot(pd.DataFrame(pairs), "pairwise_distances")
    plot_representation_audit(plot_data, root / "plots/report")
    summary = pd.read_csv(root / "analyses/representation_summary.csv")
    reconstructed = tables["summary"].set_index("variant").mean_r2
    assert np.allclose(summary.mean_va_procrustes_r2, summary.variant.map(reconstructed))
    validation = {
        "variant_layers": len(layerwise),
        "coordinate_rows": len(coordinates),
        "samples_identical_bytes": True,
        "samples_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "max_rdm_reconstruction_error": max_rdm_error,
        "max_r2_reconstruction_error": max_r2_error,
        "summary_matches": True,
        "context": context,
    }
    (tables_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    logger.info("audit.complete variant_layers=%d output=%s", len(layerwise), tables_dir)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
