"""Extended audit of a frozen neutral-projected run, with portable report tables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr

from affect_geometry.analysis.comprehensive import compute_report_tables
from affect_geometry.analysis.neutral_projected import emotion_group
from affect_geometry.analysis.quality import neighbor_order, upper_triangle
from affect_geometry.analysis.robustness import (
    apply_similarity,
    bh_adjust,
    fit_similarity,
    permutation_procrustes,
    procrustes_scores,
)
from affect_geometry.embeddings.mds import classical_mds
from affect_geometry.logging import get_logger, progress


def _axis_metrics(x, target, axes):
    fitted = apply_similarity(x, fit_similarity(x, target))
    predictions = np.empty_like(target)
    baseline = np.empty_like(target)
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        predictions[i] = apply_similarity(x[i : i + 1], fit_similarity(x[keep], target[keep]))[0]
        baseline[i] = target[keep].mean(0)
    errors = (fitted - target) ** 2
    loo_errors = (predictions - target) ** 2
    total = ((target - target.mean(0)) ** 2).sum(0)
    base_errors = ((baseline - target) ** 2).sum(0)
    rows = []
    for j, axis in enumerate(axes):
        rows.append(
            dict(
                axis=axis,
                target_variance_share=total[j] / total.sum(),
                fitted_r2=1 - errors[:, j].sum() / total[j],
                fitted_pearson=pearsonr(fitted[:, j], target[:, j]).statistic,
                loo_skill=1 - loo_errors[:, j].sum() / base_errors[j],
                loo_pearson=pearsonr(predictions[:, j], target[:, j]).statistic,
            )
        )
    return rows, fitted, predictions, 1 - loo_errors.sum() / base_errors.sum()


def extended_projected_tables(
    run: Path,
    destination: Path,
    *,
    permutations=4999,
    eigengap_permutations=999,
    repetitions=200,
    seed=1701,
    reuse_base=False,
):
    """Audit saved outputs; sample checks condition on the original fitted projection."""
    destination.mkdir(parents=True, exist_ok=True)
    if not reuse_base:
        compute_report_tables(run, destination, permutations=permutations, seed=seed)
    manifest = json.loads((run / "manifest.json").read_text())
    context = dict(
        model=manifest["config"]["model"]["name"],
        run_id=manifest["config"]["run_id"],
        source_run=str(run),
        pooling="last_token",
        split="all",
        analysis_seed=seed,
        variant="LT-NeutralProjected",
        neutral_variance_threshold=0.5,
        post_projection_pca=False,
        activation_dimensions=1024,
        distribution_cap=manifest["config"]["rdm"]["max_samples_per_emotion"],
        quality_neighbors=5,
    )
    # Extend inherited tables with explicit representation context.
    for path in destination.glob("*.csv"):
        frame = pd.read_csv(path)
        for key, value in context.items():
            if key not in frame:
                frame[key] = value
        frame.to_csv(path, index=False)

    def save(name, rows):
        frame = pd.DataFrame(rows).copy()
        for key, value in context.items():
            if key not in frame:
                frame[key] = value
        frame.to_csv(destination / f"{name}.csv", index=False)
        return frame

    reference = pd.read_csv(run / "references/goemotions_nrc_vad.csv").set_index("emotion")
    labels = sorted(reference.index)
    samples = pd.read_csv(run / "dataset/samples.csv")
    coords = pd.read_csv(run / "plot_data/coordinates.csv")
    va = pd.read_csv(destination / "va_recovery.csv")
    lookup_va = va.set_index(["layer", "distance", "method"])
    rng = np.random.default_rng(seed)
    permutation_indices = np.stack([rng.permutation(len(labels)) for _ in range(permutations)])
    axes_rows, errors, vad_sources, vad_keys, curvature, influence = [], [], [], [], [], []
    for key, group in progress(
        list(coords.groupby(["layer", "distance", "method", "dimensions"])),
        description="Auditing axes, label leverage and curvature",
        unit="embedding",
    ):
        layer, distance, method, dim = key
        if dim not in [2, 3]:
            continue
        group = group.set_index("emotion").loc[labels]
        x = group[[f"coordinate_{i}" for i in range(1, dim + 1)]].to_numpy(float)
        axes = ["valence", "arousal", "dominance"][:dim]
        target = reference.loc[labels, axes].to_numpy(float)
        rowkey = dict(layer=layer, distance=distance, method=method, dimensions=dim)
        rows, fitted, predicted, loo_skill = _axis_metrics(x, target, axes)
        axes_rows.extend({**rowkey, **row} for row in rows)
        full_score = float(procrustes_scores(x[None], target)[0])
        for i, emotion in enumerate(labels):
            err = dict(
                **rowkey,
                emotion=emotion,
                emotion_group=emotion_group(emotion),
                squared_error=((fitted[i] - target[i]) ** 2).sum(),
                loo_squared_error=((predicted[i] - target[i]) ** 2).sum(),
            )
            for j, axis in enumerate(axes):
                err.update(
                    {
                        axis: target[i, j],
                        f"aligned_{axis}": fitted[i, j],
                        f"predicted_{axis}": predicted[i, j],
                    }
                )
            errors.append(err)
            keep = np.arange(len(labels)) != i
            score = procrustes_scores(x[keep][None], target[keep])[0]
            influence.append(
                dict(
                    **rowkey,
                    emotion=emotion,
                    full_r2=full_score,
                    refit_without_emotion_r2=score,
                    change=score - full_score,
                )
            )
        if dim == 3:
            vad_sources.append(x)
            vad_keys.append(dict(**rowkey, loo_joint_skill=loo_skill))
        else:
            # Fixed eigenvector convention; descriptive, not rotation invariant.
            xs = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
            ys = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()
            models = []
            for degree in [1, 2]:
                design = np.vander(xs, degree + 1, increasing=True)
                beta = np.linalg.lstsq(design, ys, rcond=None)[0]
                residual = ys - design @ beta
                hat = np.einsum("ij,ji->i", design, np.linalg.pinv(design))
                press = np.sum((residual / (1 - hat)) ** 2)
                models.append((1 - (residual**2).sum() / (ys**2).sum(), press))
            curvature.append(
                dict(
                    **rowkey,
                    linear_r2=models[0][0],
                    quadratic_r2=models[1][0],
                    quadratic_loo_skill=1 - models[1][1] / models[0][1],
                    orientation="unaligned first and second eigenvectors",
                )
            )
    vad_observed, vad_p, vad_max, vad_null = permutation_procrustes(
        np.stack(vad_sources),
        reference.loc[labels, ["valence", "arousal", "dominance"]].to_numpy(),
        permutation_indices,
    )
    vad = pd.DataFrame(vad_keys)
    vad["vad_procrustes_r2"], vad["permutation_p"], vad["maxT_p"] = vad_observed, vad_p, vad_max
    vad["bh_q"], vad["permutations"] = bh_adjust(vad_p), permutations
    vad["null_mean"], vad["null_p95"] = vad_null.mean(1), np.quantile(vad_null, 0.95, axis=1)
    vad = save("vad_audit", vad)
    save("vad_null_maxima", dict(permutation=np.arange(permutations), max_r2=vad_null.max(0)))
    all_p = np.r_[va.permutation_p, vad.permutation_p]
    va["joint_va_vad_bh_q"] = bh_adjust(all_p)[: len(va)]
    vad["joint_va_vad_bh_q"] = bh_adjust(all_p)[len(va) :]
    save("va_recovery", va)
    save("vad_audit", vad)
    axis_frame = save("axis_diagnostics", axes_rows)
    save("emotion_residuals", errors)
    save("label_influence", influence)
    save("curvature", curvature)

    eigenrows, group_rows, pairrows, neutralrows = [], [], [], []
    mats = {}
    tri = np.triu_indices(len(labels), 1)
    signed = np.array([emotion_group(e) for e in labels])
    valid = np.flatnonzero(np.isin(signed, ["positive", "negative"]))
    for path in progress(
        sorted((run / "rdms/last_token").glob("layer_*/*.npz")),
        description="Testing eigengaps and categorical organization",
        unit="RDM",
    ):
        layer, distance = int(path.parent.name.split("_")[1]), path.stem
        with np.load(path) as z:
            assert z["labels"].astype(str).tolist() == labels
            matrix = z["distance"].copy()
        mats[layer, distance] = matrix
        values = classical_mds(matrix, len(labels) - 1).eigenvalues
        null_ratios = []
        batch = np.zeros((eigengap_permutations, len(labels), len(labels)))
        for b in range(eigengap_permutations):
            batch[b][tri] = rng.permutation(matrix[tri])
        batch += batch.transpose(0, 2, 1)
        center = np.eye(len(labels)) - np.ones((len(labels), len(labels))) / len(labels)
        gram = -0.5 * center @ (batch**2) @ center
        null_eigen = np.linalg.eigvalsh(gram)[:, ::-1]
        for k in range(1, 6):
            observed = values[k - 1] / values[k] if values[k] > 1e-12 else np.nan
            null_ratios = np.divide(
                null_eigen[:, k - 1],
                null_eigen[:, k],
                out=np.full(eigengap_permutations, np.inf),
                where=null_eigen[:, k] > 1e-12,
            )
            eigenrows.append(
                dict(
                    layer=layer,
                    distance=distance,
                    gap_rank=k,
                    eigenvalue_ratio=observed,
                    null_p95=np.quantile(null_ratios, 0.95),
                    permutation_p=(1 + (null_ratios >= observed).sum())
                    / (eigengap_permutations + 1),
                    permutations=eigengap_permutations,
                    null="symmetric off-diagonal value shuffle; MDS recomputed",
                )
            )
        sub = matrix[np.ix_(valid, valid)]
        same = signed[valid, None] == signed[valid][None, :]
        a, b = np.triu_indices(len(valid), 1)
        between, within = sub[a, b][~same[a, b]], sub[a, b][same[a, b]]
        order = neighbor_order(sub)[:, :5]
        purity = (signed[valid][order] == signed[valid, None]).mean()
        group_rows.append(
            dict(
                layer=layer,
                distance=distance,
                positive_negative_5nn_purity=purity,
                between_within_ratio=between.mean() / within.mean(),
                within_mean=within.mean(),
                between_mean=between.mean(),
                labels_used=len(valid),
                positive_labels=12,
                negative_labels=11,
            )
        )
        best_layer = int(
            lookup_va.xs((distance, "mds"), level=("distance", "method")).va_procrustes_r2.idxmax()
        )
        if layer == best_layer:
            order_pairs = np.argsort(matrix[tri])
            for kind, indexes in [
                ("nearest", order_pairs[:5]),
                ("farthest", order_pairs[-5:][::-1]),
            ]:
                for rank, p in enumerate(indexes, 1):
                    i, j = tri[0][p], tri[1][p]
                    pairrows.append(
                        dict(
                            layer=layer,
                            distance=distance,
                            kind=kind,
                            rank=rank,
                            emotion_i=labels[i],
                            emotion_j=labels[j],
                            value=matrix[i, j],
                            selection="best 2D MDS VA layer",
                        )
                    )
        neutral = labels.index("neutral")
        for rank, j in enumerate(neighbor_order(matrix)[neutral], 1):
            neutralrows.append(
                dict(
                    layer=layer,
                    distance=distance,
                    rank=rank,
                    emotion=labels[j],
                    value=matrix[neutral, j],
                )
            )
    gaps = pd.DataFrame(eigenrows)
    gaps["bh_q"] = bh_adjust(gaps.permutation_p.to_numpy())
    save("eigengaps", gaps)
    save("taxonomy_structure", group_rows)
    save("selected_pairs", pairrows)
    save("neutral_neighbors", neutralrows)

    indices = [np.flatnonzero(samples.emotion.to_numpy() == label) for label in labels]
    unique = ~samples.text.duplicated(keep="first").to_numpy()
    balance, splits, centroid_checks = [], [], []
    for layer in progress(
        [0, 7, 14, 17, 21, 27],
        description="Checking frozen-projection sample sensitivity",
        unit="layer",
    ):
        source = np.load(run / f"activations/last_token/layer_{layer:03d}.npy", mmap_mode="r")
        groups = [np.asarray(source[ix], dtype=np.float64) for ix in indices]
        centers = np.stack([g.mean(0) for g in groups])
        for distance in ["cosine", "euclidean"]:
            matrix = cdist(centers, centers, metric=distance)
            np.fill_diagonal(matrix, 0)
            centroid_checks.append(
                dict(
                    layer=layer,
                    distance=distance,
                    max_abs_rdm_error=np.max(abs(matrix - mats[layer, distance])),
                )
            )
        target = reference.loc[labels, ["valence", "arousal"]].to_numpy()
        for repeat in range(repetitions):
            cent = np.stack([g[rng.choice(len(g), 47, replace=False)].mean(0) for g in groups])
            for distance in ["cosine", "euclidean"]:
                matrix = cdist(cent, cent, metric=distance)
                np.fill_diagonal(matrix, 0)
                balance.append(
                    dict(
                        layer=layer,
                        distance=distance,
                        repeat=repeat,
                        n_per_emotion=47,
                        va_r2=procrustes_scores(classical_mds(matrix, 2).coordinates[None], target)[
                            0
                        ],
                        rdm_spearman=spearmanr(
                            upper_triangle(matrix), upper_triangle(mats[layer, distance])
                        ).statistic,
                        projection_fit="frozen all-split projection",
                    )
                )
        splitmat = {}
        for split in ["train", "validation", "test"]:
            ix = [i[(samples.iloc[i].split.to_numpy() == split) & unique[i]] for i in indices]
            cent = np.stack([np.asarray(source[i], dtype=float).mean(0) for i in ix])
            for distance in ["cosine", "euclidean"]:
                matrix = cdist(cent, cent, metric=distance)
                np.fill_diagonal(matrix, 0)
                splitmat[split, distance] = matrix
                splits.append(
                    dict(
                        layer=layer,
                        distance=distance,
                        split=split,
                        samples=sum(map(len, ix)),
                        smallest_class=min(map(len, ix)),
                        va_r2=procrustes_scores(classical_mds(matrix, 2).coordinates[None], target)[
                            0
                        ],
                        rdm_spearman_to_train=spearmanr(
                            upper_triangle(matrix), upper_triangle(splitmat["train", distance])
                        ).statistic,
                        projection_fit="frozen all-split projection",
                        deduplication="exact text globally; first occurrence retained",
                    )
                )
    save("balanced_draws", balance)
    save("frozen_split_sensitivity", splits)
    save("centroid_reconstruction", centroid_checks)
    save("projection", pd.read_csv(run / "analyses/neutral_projection.csv"))
    save("exploration", pd.read_csv(run / "plot_data/exploration_coordinates.csv"))

    for name, frame, score in [
        ("va_summary", va, "va_procrustes_r2"),
        ("vad_summary", vad, "vad_procrustes_r2"),
    ]:
        rows = []
        for (distance, method), group in frame.groupby(["distance", "method"]):
            peak = group.loc[group[score].idxmax()]
            row = dict(
                distance=distance,
                method=method,
                mean_r2=group[score].mean(),
                median_r2=group[score].median(),
                peak_r2=peak[score],
                peak_layer=int(peak.layer),
                final_r2=group.loc[group.layer == 27, score].iloc[0],
                mean_loo_skill=group.loo_joint_skill.mean(),
                peak_loo_skill=peak.loo_joint_skill,
                significant_bh=int((group.bh_q < 0.05).sum()),
                significant_maxT=int((group.maxT_p < 0.05).sum()),
                peak_maxT_p=peak.maxT_p,
            )
            for phase, lo, hi in [("early", 0, 8), ("middle", 9, 18), ("late", 19, 27)]:
                row[f"{phase}_r2"] = group.loc[group.layer.between(lo, hi), score].mean()
            rows.append(row)
        save(name, pd.DataFrame(rows).sort_values("mean_r2", ascending=False))
    save(
        "reference_variance",
        [
            dict(
                axis=a,
                mean=reference[a].mean(),
                variance=reference[a].var(ddof=0),
                va_share=reference[a].var(ddof=0)
                / reference[["valence", "arousal"]].var(ddof=0).sum(),
                vad_share=reference[a].var(ddof=0)
                / reference[["valence", "arousal", "dominance"]].var(ddof=0).sum(),
            )
            for a in ["valence", "arousal", "dominance"]
        ],
    )
    # Compare every recomputed recovery result with the frozen source summaries.
    validation = {}
    for name, new, score in [("va", va, "va_procrustes_r2"), ("vad", vad, "vad_procrustes_r2")]:
        old = pd.read_csv(run / f"analyses/{name}_recovery.csv")
        joined = new.merge(old, on=["layer", "distance", "method"], suffixes=("_new", "_old"))
        validation[f"{name}_max_score_error"] = float(
            abs(joined[score + "_new"] - joined[score + "_old"]).max()
        )
        assert validation[f"{name}_max_score_error"] < 1e-8
    graph = pd.read_csv(destination / "graph_audit.csv")
    validation["max_geodesic_error"] = float(graph.max_geodesic_change.max())
    validation["geometry_count"] = len(mats)
    validation["va_tests"] = len(va)
    validation["vad_tests"] = len(vad)
    validation["axis_rows"] = len(axis_frame)
    validation["centroid_max_error"] = float(pd.DataFrame(centroid_checks).max_abs_rdm_error.max())
    validation["finite_recovery"] = bool(np.isfinite(vad_observed).all())
    validation["passed"] = (
        validation["max_geodesic_error"] < 1e-8 and validation["centroid_max_error"] < 1e-3
    )
    (destination.parent / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    assert validation["passed"], validation
    inventory = []
    for path in sorted(run.rglob("*")):
        if not path.is_file() or "report" in path.relative_to(run).parts:
            continue
        inventory.append(
            dict(
                path=str(path.relative_to(run)),
                bytes=path.stat().st_size,
                suffix=path.suffix,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()
                if path.stat().st_size < 2_000_000
                else "large file; not hashed",
            )
        )
    save("source_inventory", inventory)
    provenance_path = destination.parent / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["extended_audit"] = dict(
        context=context,
        permutations=permutations,
        eigengap_permutations=eigengap_permutations,
        repetitions=repetitions,
        reference_path=str(run / "references/goemotions_nrc_vad.csv"),
        note="Split and balanced checks condition on the all-split neutral projection.",
    )
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    get_logger().info("projected_audit.complete validation=%s", validation)
