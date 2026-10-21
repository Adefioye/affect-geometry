"""Post-run audit and robustness analysis. Raw run artifacts are read-only inputs."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

from affect_geometry.analysis.agreement import embedding_procrustes_r2, neighborhood_overlap
from affect_geometry.analysis.intrinsic import (
    eigenvalue_summary,
    elbow_dimension,
    residual_variance,
)
from affect_geometry.analysis.quality import embedding_quality, upper_triangle
from affect_geometry.analysis.robustness import (
    apply_similarity,
    bh_adjust,
    fit_similarity,
    leave_one_emotion_out,
    partial_rank_correlation,
    permutation_procrustes,
    procrustes_scores,
    rsa_permutations,
)
from affect_geometry.embeddings.isomap import geodesic_distances
from affect_geometry.embeddings.mds import classical_mds
from affect_geometry.logging import get_logger, progress


def compute_report_tables(run: Path, destination: Path, *, permutations=4999, seed=1701):
    manifest = json.loads((run / "manifest.json").read_text())
    config = manifest["config"]
    model = config["model"]["name"]
    pool = config["activations"]["poolings"][0]
    context = dict(
        model=model,
        run_id=config["run_id"],
        pooling=pool,
        split=config["rdm"]["split"],
        seed=seed,
        pca_components=config["rdm"]["pca_components"],
        distribution_cap=config["rdm"]["max_samples_per_emotion"],
        quality_neighbors=5,
    )
    destination.mkdir(parents=True, exist_ok=True)
    tables = {}

    def save(name, frame):
        frame = pd.DataFrame(frame).copy()
        for key, value in context.items():
            if key not in frame:
                frame[key] = value
        frame.to_csv(destination / f"{name}.csv", index=False)
        tables[name] = frame
        return frame

    samples = pd.read_csv(run / "dataset/samples.csv")
    samples["word_count"] = samples.text.fillna("").str.split().str.len()
    counts = samples.groupby("emotion", as_index=False).agg(
        sample_count=("sample_id", "size"),
        mean_words=("word_count", "mean"),
        median_words=("word_count", "median"),
    )
    capped = pd.read_csv(run / "rdms/distribution_sample.csv").groupby("emotion").size()
    counts["distribution_sample_count"] = counts.emotion.map(capped)
    counts = save("class_balance", counts)
    save("split_balance", samples.groupby(["split", "emotion"], as_index=False).size())
    reference_path = Path(config["analysis"]["va_reference"])
    reference = pd.read_csv(reference_path)
    labels = sorted(samples.emotion.unique())
    target = reference.set_index("emotion").loc[labels, ["valence", "arousal"]].to_numpy()
    save("reference", reference)
    freq = np.log(counts.set_index("emotion").loc[labels, "sample_count"].to_numpy())
    length = counts.set_index("emotion").loc[labels, "mean_words"].to_numpy()
    nuisance = np.column_stack(
        [
            upper_triangle(abs(freq[:, None] - freq[None, :])),
            upper_triangle(freq[:, None] + freq[None, :]),
            upper_triangle(abs(length[:, None] - length[None, :])),
        ]
    )
    mats, coords, sources, va_keys = {}, {}, [], []
    quality, va, predictions, alignments, dimensions, spectra, residuals = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    pairs, audit, graph_audit, sweep, sensitivity, shepard, comparison = [], [], [], [], [], [], []
    input_hashes = {}
    cosine_metric = []
    paths = sorted((run / "rdms" / pool).glob("layer_*/*.npz"))
    for path in progress(paths, description="Auditing all emotion geometries", unit="RDM"):
        layer = int(path.parent.name.split("_")[1])
        distance = path.stem
        key = dict(layer=layer, distance=distance)
        input_hashes[str(path.relative_to(run))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as z:
            if z["labels"].astype(str).tolist() != labels:
                raise ValueError(f"Emotion ordering mismatch: {path}")
            direct, raw = z["distance"].copy(), z["raw"].copy()
        if not np.isfinite(direct).all() or not np.allclose(direct, direct.T):
            raise ValueError(f"Invalid RDM: {path}")
        mats[layer, distance] = direct
        if distance == "cosine":
            for transform, transformed in [
                ("cosine dissimilarity", direct),
                ("unit-sphere chord", np.sqrt(2 * np.maximum(direct, 0))),
            ]:
                for method in ["mds", "isomap"]:
                    graph = (
                        transformed if method == "mds" else geodesic_distances(transformed, 5)[0]
                    )
                    result = classical_mds(graph, 2)
                    cosine_metric.append(
                        {
                            **key,
                            "method": method,
                            "transform": transform,
                            "va_r2": procrustes_scores(result.coordinates[None], target)[0],
                            **embedding_quality(transformed, result.coordinates, 5),
                            **eigenvalue_summary(result.eigenvalues),
                        }
                    )
        geo, used = geodesic_distances(direct, 5)
        original_geo = np.load(
            run / "embeddings" / pool / path.parent.name / distance / "geodesic_distances.npz"
        )["distances"]
        original_c = pd.read_csv(
            run / "embeddings" / pool / path.parent.name / distance / "coordinates.csv"
        )
        mds, iso = classical_mds(direct, len(labels) - 1), classical_mds(geo, len(labels) - 1)
        self_neighbors = int(
            (np.argsort(direct, axis=1)[:, 1:6] == np.arange(len(labels))[:, None]).any(1).sum()
        )
        graph_audit.append(
            {
                **key,
                "old_self_neighbor_rows": self_neighbors,
                "max_geodesic_change": float(abs(geo - original_geo).max()),
                "neighbors_used": used,
            }
        )
        tri = np.triu_indices(len(labels), 1)
        d = direct[tri]
        triangles = np.array(
            [
                [direct[i, j], direct[i, k], direct[j, k]]
                for i, j, k in combinations(range(len(labels)), 3)
            ]
        )
        audit.append(
            {
                **key,
                "raw_negative_pairs": int((raw[tri] < 0).sum()),
                "zero_distance_pairs": int((d == 0).sum()),
                "pairs": len(d),
                "triangle_violation_fraction": float(
                    (2 * triangles.max(1) > triangles.sum(1) + 1e-10).mean()
                ),
                "self_neighbor_rows_before_fix": self_neighbors,
                "shepard_pair_count": len(d),
            }
        )
        for i, j in zip(*tri):
            pairs.append(
                {
                    **key,
                    "emotion_i": labels[i],
                    "emotion_j": labels[j],
                    "direct_distance": direct[i, j],
                    "geodesic_distance": geo[i, j],
                    "ratio": geo[i, j] / direct[i, j] if direct[i, j] > 0 else np.nan,
                    "raw_dissimilarity": raw[i, j],
                }
            )
        for method, result in [("mds", mds), ("isomap", iso)]:
            x = result.coordinates[:, :2]
            coords[layer, distance, method] = x
            va_keys.append({**key, "method": method})
            sources.append(x)
            fitted = apply_similarity(x, fit_similarity(x, target))
            metrics, loo, baseline = leave_one_emotion_out(x, target)
            _, neutral_score = (
                None,
                procrustes_scores(
                    x[np.array(labels) != "neutral"][None], target[np.array(labels) != "neutral"]
                )[0],
            )
            axis_metrics = {
                f"{axis}_pearson": pearsonr(fitted[:, j], target[:, j]).statistic
                for j, axis in enumerate(["valence", "arousal"])
            }
            metrics.update(axis_metrics)
            metrics["r2_without_neutral"] = neutral_score
            metrics["method"] = method
            metrics.update(key)
            va.append(metrics)
            # Leave-one-emotion-out refits, preserving the geometry of all 28 nodes.
            for i, emotion in enumerate(labels):
                alignments.append(
                    {
                        **key,
                        "method": method,
                        "emotion": emotion,
                        "embedding_x": x[i, 0],
                        "embedding_y": x[i, 1],
                        "aligned_valence": fitted[i, 0],
                        "aligned_arousal": fitted[i, 1],
                        "valence": target[i, 0],
                        "arousal": target[i, 1],
                        "squared_error": float(((fitted[i] - target[i]) ** 2).sum()),
                    }
                )
                predictions.append(
                    {
                        **key,
                        "method": method,
                        "emotion": emotion,
                        "true_valence": target[i, 0],
                        "true_arousal": target[i, 1],
                        "predicted_valence": loo[i, 0],
                        "predicted_arousal": loo[i, 1],
                        "baseline_valence": baseline[i, 0],
                        "baseline_arousal": baseline[i, 1],
                    }
                )
            for dim in [1, 2, 3]:
                met = embedding_quality(direct, result.coordinates[:, :dim], 5)
                quality.append({**key, "method": method, "dimensions": dim, **met})
                # Affine decoding tests coordinate utility rather than preservation of shape.
                cv, _, _ = leave_one_emotion_out(result.coordinates[:, :dim], target, affine=True)
                comparison.append({**key, "method": method, "dimensions": dim, **cv})
            for component, value in enumerate(result.eigenvalues, 1):
                positive = np.maximum(result.eigenvalues, 0)
                spectra.append(
                    {
                        **key,
                        "method": method,
                        "component": component,
                        "eigenvalue": value,
                        "positive_mass": max(value, 0) / positive.sum(),
                        "cumulative_positive_mass": positive[:component].sum() / positive.sum(),
                    }
                )
            residual = [residual_variance(geo, iso.coordinates[:, :d]) for d in range(1, 11)]
            dimensions.append(
                {
                    **key,
                    "method": method,
                    **eigenvalue_summary(result.eigenvalues),
                    "isomap_residual_elbow": elbow_dimension(list(range(1, 11)), residual)
                    if method == "isomap"
                    else np.nan,
                }
            )
            if method == "isomap":
                residuals.extend(
                    {**key, "dimensions": d, "residual_variance": r}
                    for d, r in enumerate(residual, 1)
                )
            for k in [3, 5, 8, 10]:
                sensitivity.append(
                    {**key, "method": method, "neighbors": k, **embedding_quality(direct, x, k)}
                )
            old = (
                original_c.loc[(original_c.method == method) & (original_c.dimensions == 2)]
                .set_index("emotion")
                .loc[labels, ["coordinate_1", "coordinate_2"]]
                .to_numpy()
            )
            graph_audit[-1][f"{method}_old_new_procrustes_r2"] = embedding_procrustes_r2(old, x)
            graph_audit[-1][f"{method}_old_new_va_r2_difference"] = float(
                procrustes_scores(x[None], target)[0] - procrustes_scores(old[None], target)[0]
            )
            if layer in [0, 7, 14, 21, 27]:
                embedded = squareform(pdist(x))
                shepard.extend(
                    {
                        **key,
                        "method": method,
                        "emotion_i": labels[i],
                        "emotion_j": labels[j],
                        "direct_distance": direct[i, j],
                        "embedded_distance": embedded[i, j],
                    }
                    for i, j in zip(*tri)
                )
        for k in [3, 4, 5, 6, 8, 10, 12, 20, 27]:
            gd, ku = geodesic_distances(direct, k)
            gx = classical_mds(gd, 2).coordinates
            sweep.append(
                {
                    **key,
                    "requested_neighbors": k,
                    "neighbors_used": ku,
                    "va_r2": procrustes_scores(gx[None], target)[0],
                    "residual_variance": residual_variance(gd, gx),
                    "geodesic_ratio_median": np.nanmedian(
                        np.divide(gd[tri], d, out=np.full_like(d, np.nan), where=d > 0)
                    ),
                    **embedding_quality(direct, gx, 5),
                }
            )
    rng = np.random.default_rng(seed)
    permutations_idx = np.stack([rng.permutation(len(labels)) for _ in range(permutations)])
    get_logger().info("analysis.permutations tests=%d permutations=%d", len(sources), permutations)
    with_progress = progress(
        [0], description="Testing V-A recovery with shared permutations", unit="family"
    )
    for _ in with_progress:
        obs, p, pmax, null = permutation_procrustes(np.stack(sources), target, permutations_idx)
    va = pd.DataFrame(va)
    va["dimensions"] = 2
    va["va_procrustes_r2"], va["permutation_p"], va["maxT_p"] = obs, p, pmax
    va["bh_q"] = bh_adjust(p)
    va["null_mean"] = null.mean(1)
    va["null_p95"] = np.quantile(null, 0.95, axis=1)
    va["permutations"] = permutations
    save("va_recovery", va)
    save(
        "va_null_maxima", pd.DataFrame({"permutation": range(permutations), "max_r2": null.max(0)})
    )
    save("va_predictions", pd.DataFrame(predictions).assign(dimensions=2))
    save("va_alignment", pd.DataFrame(alignments).assign(dimensions=2))
    save("quality", quality)
    save("affine_decoding", comparison)
    save("intrinsic", dimensions)
    save("spectrum", spectra)
    save("residual_variance", residuals)
    save("geodesic_pairs", pairs)
    save("metric_audit", audit)
    save("graph_audit", graph_audit)
    save("neighbor_sweep", sweep)
    save("quality_k_sensitivity", sensitivity)
    save("shepard_pairs", shepard)
    save("cosine_metric_sensitivity", cosine_metric)

    agreement, neighborhood, cross = [], [], []
    distances = sorted({d for _, d in mats})
    layers = sorted({layer for layer, _ in mats})
    for layer in progress(
        layers, description="Comparing distances and neighborhoods", unit="layer"
    ):
        for a, b in combinations(distances, 2):
            overlap, changes = neighborhood_overlap(labels, mats[layer, a], mats[layer, b], 5)
            changes["layer"], changes["distance_i"], changes["distance_j"] = layer, a, b
            neighborhood.extend(changes.to_dict("records"))
            for method in ["mds", "isomap"]:
                agreement.append(
                    dict(
                        layer=layer,
                        distance_i=a,
                        distance_j=b,
                        method=method,
                        spearman=spearmanr(
                            upper_triangle(mats[layer, a]), upper_triangle(mats[layer, b])
                        ).statistic,
                        procrustes_r2=embedding_procrustes_r2(
                            coords[layer, a, method], coords[layer, b, method]
                        ),
                        knn_overlap=overlap,
                    )
                )
    for distance in distances:
        for method in ["mds", "isomap"]:
            for a, b in combinations(layers, 2):
                cross.append(
                    dict(
                        distance=distance,
                        method=method,
                        layer_i=a,
                        layer_j=b,
                        procrustes_r2=embedding_procrustes_r2(
                            coords[a, distance, method], coords[b, distance, method]
                        ),
                        rdm_spearman=spearmanr(
                            upper_triangle(mats[a, distance]), upper_triangle(mats[b, distance])
                        ).statistic,
                    )
                )
    save("distance_agreement", pd.DataFrame(agreement).assign(dimensions=2))
    save("neighborhood_change", neighborhood)
    save("cross_layer", pd.DataFrame(cross).assign(dimensions=2))
    rsa = []
    matrix_keys = sorted(mats)
    matrix_array = np.stack([mats[key] for key in matrix_keys])
    tri = np.triu_indices(len(labels), 1)
    for axis, values in progress(
        [("valence", target[:, :1]), ("arousal", target[:, 1:]), ("joint", target)],
        description="Testing reference RDM agreement",
        unit="reference",
    ):
        r, p = rsa_permutations(matrix_array, values, permutations_idx)
        ref_vector = pdist(values)
        for i, (layer, distance) in enumerate(matrix_keys):
            controls = nuisance
            if axis != "joint":
                other = pdist(target[:, 1:] if axis == "valence" else target[:, :1])
                controls = np.column_stack([other, nuisance])
            rsa.append(
                dict(
                    layer=layer,
                    distance=distance,
                    reference_axis=axis,
                    spearman=r[i],
                    permutation_p=p[i],
                    partial_spearman=partial_rank_correlation(
                        matrix_array[i][tri], ref_vector, controls
                    ),
                    permutations=permutations,
                )
            )
    rsa = pd.DataFrame(rsa)
    rsa["bh_q"] = bh_adjust(rsa.permutation_p.to_numpy())
    save("reference_rsa", rsa)
    # All-label selection is made explicit in the chosen examples table.
    selection = []
    for (distance, method), group in va.groupby(["distance", "method"]):
        best = group.loc[group.va_procrustes_r2.idxmax()]
        selection.append(
            dict(
                distance=distance,
                method=method,
                layer=int(best.layer),
                criterion="maximum in-sample V-A R2 across 28 layers",
            )
        )
    save("selected_examples", selection)
    provenance = dict(
        source_run=str(run),
        context=context,
        manifest=manifest,
        permutations=permutations,
        input_rdm_sha256=input_hashes,
        reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        samples=len(samples),
        duplicate_text_rows=int(samples.text.duplicated().sum()),
        distinct_source_ids=int(samples.source_id.nunique()),
        note=(
            "All source artifacts preserved. Isomap graphs rebuilt with explicit self exclusion. "
            "Stable ties use alphabetical label order."
        ),
    )
    (destination.parent / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return tables
