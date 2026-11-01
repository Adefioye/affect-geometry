"""RQ3 orchestration: saved source → cohorts → geometry → evaluation → inference."""

import importlib.metadata
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from affect_geometry.analysis.cohorts import balanced_cohorts
from affect_geometry.analysis.geometry_comparisons import (
    COHORTS,
    METHODS,
    aggregate_scores,
    contrasts,
    emergence,
    selected_cka,
)
from affect_geometry.analysis.geometry_inference import (
    corrected_pvalues,
    infer,
    permutation_schedule,
)
from affect_geometry.analysis.geometry_preprocessing import preprocess
from affect_geometry.analysis.geometry_validation import evaluate, evaluate_layers
from affect_geometry.analysis.nonlinearity import (
    fidelity,
    graph_diagnostics,
    layerwise_metric_means,
    spectral_table,
    spectrum,
    subspace_stability,
    triangle_violations,
)
from affect_geometry.analysis.topology import persistence
from affect_geometry.classification.protocol import EMOTIONS
from affect_geometry.embeddings.diffusion_maps import diffusion_maps
from affect_geometry.embeddings.geometry import (
    Decomposition,
    UnsupportedGeometry,
    isomap_decomposition,
    mds_decomposition,
)
from affect_geometry.logging import progress
from affect_geometry.persistence.correctness_geometry import GeometryStore, checksum, fingerprint


def settings(config):
    yield "mds", "classical", {}
    for k in config["isomap_neighbors"]:
        yield "isomap", f"k{k}", dict(neighbors=k)
    for k, rank, t, alpha in itertools.product(
        config["diffusion_neighbors"],
        config["diffusion_scale_ranks"],
        config["diffusion_times"],
        config["diffusion_alphas"],
    ):
        yield (
            "diffusion_maps",
            f"k{k}-s{rank}-t{t}-a{alpha:g}",
            dict(neighbors=k, scale_rank=rank, time=t, alpha=alpha),
        )


def context_frame(frame, context):
    result = frame.copy()
    for key, value in context.items():
        if key not in result:
            result[key] = value
    if "status" not in result:
        result["status"] = "ok"
    else:
        result["status"] = result["status"].fillna("ok")
    if "reason" not in result:
        result["reason"] = ""
    else:
        result["reason"] = result["reason"].fillna("")
    if "value" in result:
        invalid = ~np.isfinite(pd.to_numeric(result["value"], errors="coerce"))
        result.loc[invalid, "status"] = "unsupported"
        result.loc[invalid & result.reason.eq(""), "reason"] = "undefined_metric"
    return result


def _decompose(method, d, params, labels, tolerance):
    if method == "mds":
        return mds_decomposition(d, tolerance)
    if method == "isomap":
        return isomap_decomposition(d, ids=labels, tolerance=tolerance, **params)
    return diffusion_maps(d, ids=labels, tolerance=tolerance, **params)


def run_geometry(
    config,
    slug,
    samples,
    arrays,
    source_manifest,
    source_audit,
    activation_metadata,
    target,
    nrc_metadata,
    root,
    prepared=None,
    skip_permutations=False,
):
    """Injectable scientific runner; arrays may be memory maps in production."""
    started = time.monotonic()
    from affect_geometry.experiments.rq3_budget import RunBudget

    budget = RunBudget(config["budgets"])
    cohort = (
        prepared.cohorts
        if prepared is not None
        else balanced_cohorts(samples, EMOTIONS, config["seed"])
    )
    labels = cohort.labels
    if target.shape != (len(labels), 2):
        raise ValueError("NRC target rows must match the saved supported emotion order")
    layers = list(range(len(arrays))) if config["layers"] is None else config["layers"]
    if (
        not layers
        or len(set(layers)) != len(layers)
        or any(layer < 0 or layer >= len(arrays) for layer in layers)
    ):
        raise ValueError("Invalid prespecified layer panel")
    layers = sorted(layers)
    code_root = Path(__file__).parents[1]
    code_hash = fingerprint(
        {str(p.relative_to(code_root)): checksum(p) for p in sorted(code_root.rglob("*.py"))}
    )
    manifest = dict(
        config=config,
        model=slug,
        source=source_manifest,
        source_audit=source_audit,
        activations=activation_metadata,
        nrc=nrc_metadata,
        emotion_order=labels,
        layers=layers,
        layer_weights=[1 / len(layers)] * len(layers),
        code_hash=code_hash,
        versions={
            p: importlib.metadata.version(p) for p in ("numpy", "scipy", "pandas", "scikit-learn")
        },
        inference_label=(
            "descriptive_permutations_skipped"
            if skip_permutations
            else "non_inferential_smoke"
            if config["smoke"]
            else "alignment_against_chance"
        ),
        evaluation="transductive_representation_held_out_rating",
        graph_policy="union; distance ties broken by emotion ID; reject disconnected",
        full_population=source_audit.get("full_population", False),
    )
    store = GeometryStore(root, manifest)
    if store.valid("run", store.manifest):
        return store.root
    base = dict(
        model=slug,
        run_id=config["run_id"],
        source_split="pooled_original_splits",
        representation_level="emotion_centroid",
        preprocessing="cohort_neutral_pca_0.5",
        distance="cosine",
        emotion_set_id=fingerprint(labels),
        emotion_order="|".join(labels),
        K=len(labels),
        draw_id=config["seed"],
        input_condition="classification_prompt_before_generation",
        fold_policy="nested_LOEO",
        inference_label=manifest["inference_label"],
    )
    store.csv("cohorts/samples.csv", cohort.samples)
    store.csv("cohorts/support.csv", context_frame(cohort.support, base))
    store.csv("cohorts/unresolved.csv", cohort.unresolved)
    store.csv("cohorts/counts.csv", context_frame(cohort.counts, base))
    store.json("activations/references.json", activation_metadata)
    store.json(
        "validation/fold_schedule.json",
        [
            dict(held_out=i, emotion=e, training=[j for j in range(len(labels)) if j != i])
            for i, e in enumerate(labels)
        ],
    )
    store.csv(
        "validation/targets.csv",
        pd.DataFrame(dict(emotion=labels, valence=target[:, 0], arousal=target[:, 1])),
    )
    row_index = {sample: i for i, sample in enumerate(samples.sample_id)}
    candidates = {dim: {} for dim in (1, 2)}
    coordinates, evaluations, score_rows, metric_rows, coordinate_rows = {}, {}, [], [], []
    spectrum_rows, pair_rows, topology_rows, betti_rows, ratio_rows, failure_rows = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    preprocessing = {}
    for c in progress(COHORTS, description=f"{slug}: cohort geometry", unit="cohort"):
        selected = cohort.samples.loc[cohort.samples.cohort.eq(c)].sort_values("sample_id")
        rows = np.array([row_index[s] for s in selected.sample_id], dtype=int)
        for layer in progress(layers, description=f"{slug}/{c}: block geometry", unit="layer"):
            budget.check()
            ctx = dict(
                base, cohort=c, layer=layer, depth=(layer + 1) / len(arrays), n_samples=len(rows)
            )
            failure = None
            prepath = f"preprocessing/{c}/layer_{layer:03d}.npz"
            preprov = dict(
                manifest=fingerprint(store.manifest),
                cohort=c,
                layer=layer,
                sample_ids=selected.sample_id.tolist(),
            )
            try:
                stage = f"preprocessing/{c}-{layer}"
                if store.valid(stage, preprov):
                    with np.load(store.root / prepath) as saved:
                        prep = dict(saved)
                else:
                    prep = (
                        prepared.preprocessing(c, layer)
                        if prepared is not None
                        else preprocess(
                            np.asarray(arrays[layer][rows]), selected, labels, config["tolerance"]
                        )
                    )
                    path = store.arrays(prepath, **prep)
                    meta = store.json(prepath.replace(".npz", ".json"), preprov)
                    store.complete(stage, preprov, [path, meta])
                preprocessing[(c, layer)] = prep
                original = mds_decomposition(prep["D"], config["tolerance"])
                diagnostics = dict(
                    spectrum(original.eigenvalues),
                    **triangle_violations(prep["D"]),
                    neutral_components_removed=len(prep["components"]),
                )
                for metric, value in diagnostics.items():
                    metric_rows.append(
                        dict(
                            ctx,
                            method="shared",
                            dimension=0,
                            target_set="none",
                            setting="original",
                            reference="cosine",
                            metric=metric,
                            value=value,
                        )
                    )
                # Within-emotion scatter after the cohort-specific projection.
                if prepared is not None:
                    projected = prepared.projected(c, layer)
                else:
                    centered = np.asarray(arrays[layer][rows], float) - prep["grand_mean"]
                    projected = centered - (centered @ prep["components"].T) @ prep["components"]
                for i, emotion in enumerate(labels):
                    residual = projected[selected.emotion.to_numpy() == emotion] - prep["R"][i]
                    metric_rows.append(
                        dict(
                            ctx,
                            method="shared",
                            dimension=0,
                            target_set="none",
                            setting="original",
                            reference="projected_samples",
                            emotion=emotion,
                            metric="within_emotion_scatter",
                            value=float(np.mean(np.sum(residual**2, axis=1))),
                        )
                    )
                diagrams, betti = persistence(
                    prep["D"], config["filtration_bound"], config["betti_grid_size"]
                )
                topology_rows.extend(
                    context_frame(
                        diagrams,
                        dict(ctx, method="shared", filtration_bound=config["filtration_bound"]),
                    ).to_dict("records")
                )
                betti_rows.extend(
                    context_frame(
                        betti,
                        dict(ctx, method="shared", filtration_bound=config["filtration_bound"]),
                    ).to_dict("records")
                )
            except UnsupportedGeometry as exc:
                failure = str(exc)
                failure_rows.append(
                    dict(
                        ctx,
                        method="shared",
                        stage="preprocessing",
                        reason=failure,
                        status="unsupported",
                    )
                )
            for method in METHODS:
                for dim in (1, 2):
                    candidates[dim][(c, method, layer)] = {}
            for method, setting, params in settings(config):
                decomposition = None
                setting_failure = failure
                geompath = f"geometry/{c}/layer_{layer:03d}/{method}/{setting}"
                if failure is None:
                    try:
                        stage = f"geometry/{c}-{layer}-{method}-{setting}"
                        prov = dict(preprov, method=method, settings=params)
                        if store.valid(stage, prov):
                            with np.load(
                                store.root / f"{geompath}/decomposition/arrays.npz"
                            ) as saved:
                                content = dict(saved)
                            meta = json.loads(
                                (store.root / f"{geompath}/decomposition/metadata.json").read_text()
                            )
                            decomposition = Decomposition(
                                content.pop("eigenvalues"),
                                content.pop("eigenvectors"),
                                content.pop("weights"),
                                content.pop("reference"),
                                content,
                                meta,
                            )
                        else:
                            decomposition = _decompose(
                                method, prep["D"], params, labels, config["tolerance"]
                            )
                            p = store.arrays(
                                f"{geompath}/decomposition/arrays.npz",
                                eigenvalues=decomposition.eigenvalues,
                                eigenvectors=decomposition.eigenvectors,
                                weights=decomposition.weights,
                                reference=decomposition.reference,
                                **decomposition.arrays,
                            )
                            q = store.json(
                                f"{geompath}/decomposition/metadata.json", decomposition.metadata
                            )
                            store.complete(stage, prov, [p, q])
                        spectral = spectral_table(decomposition)
                        stability = subspace_stability(
                            decomposition,
                            lambda d: _decompose(method, d, params, labels, config["tolerance"]),
                            prep["D"],
                        )
                        store.csv(
                            f"{geompath}/decomposition/subspace_stability.csv",
                            context_frame(stability, dict(ctx, method=method, setting=setting)),
                        )
                        spectrum_rows.extend(
                            context_frame(
                                spectral, dict(ctx, method=method, setting=setting)
                            ).to_dict("records")
                        )
                        if method == "isomap":
                            ratios = graph_diagnostics(prep["D"], decomposition.reference, labels)
                            ratio_rows.extend(
                                context_frame(
                                    ratios, dict(ctx, method=method, setting=setting)
                                ).to_dict("records")
                            )
                            metric_rows.append(
                                dict(
                                    ctx,
                                    method="graph",
                                    dimension=0,
                                    target_set="none",
                                    setting=setting,
                                    reference="cosine",
                                    metric="geodesic_direct_ratio_mean",
                                    value=ratios.ratio.mean(),
                                )
                            )
                        if method == "diffusion_maps":
                            summaries = spectrum(decomposition.eigenvalues, decomposition.weights)
                            i, j = np.triu_indices(len(labels), 1)
                            summaries["diffusion_direct_spearman"] = float(
                                spearmanr(decomposition.reference[i, j], prep["D"][i, j]).statistic
                            )
                            for metric, value in summaries.items():
                                metric_rows.append(
                                    dict(
                                        ctx,
                                        method=method,
                                        dimension=0,
                                        target_set="none",
                                        setting=setting,
                                        reference="diffusion_full_spectrum",
                                        metric=metric,
                                        value=value,
                                    )
                                )
                        for metric in (
                            "graph_density",
                            "algebraic_connectivity",
                            "components",
                            "nearly_complete",
                        ):
                            if metric in decomposition.metadata:
                                metric_rows.append(
                                    dict(
                                        ctx,
                                        method=method,
                                        dimension=0,
                                        target_set="none",
                                        setting=setting,
                                        reference="graph",
                                        metric=metric,
                                        value=decomposition.metadata[metric],
                                    )
                                )
                    except UnsupportedGeometry as exc:
                        setting_failure = str(exc)
                for dim in (1, 2):
                    dctx = dict(
                        ctx,
                        method=method,
                        dimension=dim,
                        setting=setting,
                        target_set="valence" if dim == 1 else "VA",
                    )
                    try:
                        if setting_failure:
                            raise UnsupportedGeometry(setting_failure)
                        x = decomposition.coordinates(dim, config["tolerance"])
                        # Primary selection never mixes optional alpha sensitivities with alpha=1.
                        if method != "diffusion_maps" or params["alpha"] == 1.0:
                            candidates[dim][(c, method, layer)][setting] = x
                        store.arrays(f"{geompath}/{dim}d/coordinates.npz", native=x)
                        store.json(
                            f"{geompath}/{dim}d/metadata.json",
                            dict(
                                dctx,
                                decomposition=f"{geompath}/decomposition/arrays.npz",
                                decomposition_sha256=checksum(
                                    store.root / f"{geompath}/decomposition/arrays.npz"
                                ),
                                preprocessing=prepath,
                                preprocessing_sha256=checksum(store.root / prepath),
                                reference_ids=selected.sample_id.tolist(),
                                status="ok",
                            ),
                        )
                        for refname, reference in [
                            ("cosine", prep["D"]),
                            ("native", decomposition.reference),
                        ]:
                            metrics, pairs = fidelity(reference, x, config["neighborhoods"])
                            pairs["emotion_i"] = pairs.i.map(dict(enumerate(labels)))
                            pairs["emotion_j"] = pairs.j.map(dict(enumerate(labels)))
                            pair_rows.extend(
                                context_frame(pairs, dict(dctx, reference=refname)).to_dict(
                                    "records"
                                )
                            )
                            for metric, value in metrics.items():
                                metric_rows.append(
                                    dict(
                                        dctx,
                                        reference=refname,
                                        metric=metric,
                                        value=value,
                                        status="ok" if np.isfinite(value) else "unsupported",
                                        reason=""
                                        if np.isfinite(value)
                                        else "undefined_or_invalid_neighborhood",
                                    )
                                )
                        for i, emotion in enumerate(labels):
                            coordinate_rows.append(
                                dict(
                                    dctx,
                                    emotion=emotion,
                                    x=x[i, 0],
                                    y=x[i, 1] if dim == 2 else np.nan,
                                    sample_count=int(prep["sample_counts"][i]),
                                    coordinate_type="native",
                                    status="ok",
                                    reason="",
                                )
                            )
                    except UnsupportedGeometry as exc:
                        record = dict(dctx, status="unsupported", reason=str(exc))
                        failure_rows.append(record)
                        coordinate_rows.append(
                            dict(record, emotion="", x=np.nan, y=np.nan, coordinate_type="native")
                        )
                        store.json(f"{geompath}/{dim}d/metadata.json", record)
            for method in METHODS:
                for dim in (1, 2):
                    key = (c, method, layer, dim)
                    sctx = dict(
                        ctx,
                        method=method,
                        dimension=dim,
                        target_set="valence" if dim == 1 else "VA",
                    )
                    try:
                        result = evaluate(
                            candidates[dim][key[:3]], target[:, :dim], selection=config["selection"]
                        )
                        evaluations[key] = result
                        setting = result["descriptive_setting"]
                        coordinates[key] = candidates[dim][key[:3]][setting]
                        score_rows.append(
                            dict(
                                sctx,
                                cv_r2=result["cv_r2"],
                                full_r2=result["full_r2"],
                                valence_pearson=result["pearson"][0],
                                arousal_pearson=result["pearson"][1] if dim == 2 else np.nan,
                                setting=setting,
                                status="ok",
                                reason="",
                            )
                        )
                        path = f"validation/{c}/{method}/layer_{layer:03d}/{dim}d"
                        store.json(f"{path}/folds.json", result["folds"])
                        store.json(
                            f"{path}/full_transform.json",
                            dict(setting=setting, **result["full_transform"]),
                        )
                        table = pd.DataFrame(
                            dict(
                                emotion=labels,
                                observed_valence=target[:, 0],
                                predicted_valence=result["predictions"][:, 0],
                                error=result["errors"],
                                training_mean_valence=result["training_means"][:, 0],
                            )
                        )
                        if dim == 2:
                            table["observed_arousal"] = target[:, 1]
                            table["predicted_arousal"] = result["predictions"][:, 1]
                            table["training_mean_arousal"] = result["training_means"][:, 1]
                        store.csv(f"{path}/predictions.csv", context_frame(table, sctx))
                        for i, emotion in enumerate(labels):
                            coordinate_rows.append(
                                dict(
                                    sctx,
                                    setting=setting,
                                    emotion=emotion,
                                    x=result["aligned"][i, 0],
                                    y=result["aligned"][i, 1] if dim == 2 else np.nan,
                                    coordinate_type="full_data_aligned",
                                    status="ok",
                                    reason="",
                                )
                            )
                    except UnsupportedGeometry as exc:
                        score_rows.append(
                            dict(
                                sctx,
                                cv_r2=np.nan,
                                full_r2=np.nan,
                                valence_pearson=np.nan,
                                arousal_pearson=np.nan,
                                setting="",
                                status="unsupported",
                                reason=str(exc),
                            )
                        )
    scores = pd.DataFrame(score_rows)
    scores["raw_p"], scores["corrected_p"] = np.nan, np.nan
    aggregates = aggregate_scores(scores, layers)
    aggregates["raw_p"], aggregates["corrected_p"] = np.nan, np.nan
    for table in (scores, aggregates):
        table["inference_status"] = (
            "skipped"
            if skip_permutations
            else "non_inferential_smoke"
            if config["smoke"]
            else "pending"
        )
        table["inference_reason"] = "user_requested_skip" if skip_permutations else ""
    schedule = (
        np.empty((0, len(labels)), dtype=int)
        if skip_permutations
        else permutation_schedule(
            len(labels), config["permutations"], config["permutation_seed"], config["smoke"]
        )
    )
    store.arrays("inference/permutation_schedule.npz", indices=schedule)
    for dim in (1, 2):
        if skip_permutations:
            store.json(
                f"inference/{dim}d/status.json",
                dict(
                    status="skipped",
                    reason="user_requested_skip",
                    completed=0,
                    requested=config["permutations"],
                ),
            )
            continue
        family_scores = scores.loc[scores.dimension.eq(dim)]
        # A failed member invalidates the whole family, never silently drops out.
        if not family_scores.status.eq("ok").all():
            for table in (scores, aggregates):
                table.loc[table.dimension.eq(dim), "inference_status"] = "unsupported"
                table.loc[table.dimension.eq(dim), "inference_reason"] = "failed_family_member"
            store.json(
                f"inference/{dim}d/status.json",
                dict(
                    status="unsupported",
                    reason="failed_family_member",
                    expected_members=[list(k) for k in sorted(candidates[dim])],
                    completed=0,
                ),
            )
            continue
        inference = infer(
            candidates[dim],
            target[:, :dim],
            layers,
            schedule,
            config["selection"],
            store=store,
            prefix=f"inference/{dim}d",
            budget=budget,
        )
        store.arrays(
            f"inference/{dim}d/nulls.npz",
            layerwise=inference["layerwise_null"],
            aggregate=inference["aggregate_null"],
        )
        store.json(f"inference/{dim}d/selections.json", inference["selections"])
        store.json(
            f"inference/{dim}d/families.json",
            dict(
                layerwise=inference["keys"],
                aggregate=inference["groups"],
                seed=config["permutation_seed"],
                completed=len(schedule),
                failures=inference["failures"],
                scope="within_model_and_target",
                layers=layers,
                label=manifest["inference_label"],
            ),
        )
        if config["smoke"]:
            continue  # Never present inferential p-values from a smoke schedule.
        for table in (scores, aggregates):
            table.loc[table.dimension.eq(dim), "inference_status"] = (
                "unsupported" if inference["failures"] else "ok"
            )
            table.loc[table.dimension.eq(dim), "inference_reason"] = (
                "failed_permutation_family_member" if inference["failures"] else ""
            )
        indices = [
            scores.index[
                scores.cohort.eq(c)
                & scores.method.eq(m)
                & scores.layer.eq(layer)
                & scores.dimension.eq(dim)
            ][0]
            for c, m, layer in inference["keys"]
        ]
        raw, adjusted = corrected_pvalues(scores.loc[indices, "cv_r2"], inference["layerwise_null"])
        scores.loc[indices, "raw_p"], scores.loc[indices, "corrected_p"] = raw, adjusted
        indices = [
            aggregates.index[
                aggregates.cohort.eq(c) & aggregates.method.eq(m) & aggregates.dimension.eq(dim)
            ][0]
            for c, m in inference["groups"]
        ]
        raw, adjusted = corrected_pvalues(
            aggregates.loc[indices, "A_M"], inference["aggregate_null"]
        )
        aggregates.loc[indices, "raw_p"], aggregates.loc[indices, "corrected_p"] = raw, adjusted
    # Secondary selected-pipeline prediction, distinct from primary mean-layer A_M.
    selected_rows = []
    for dim in (1, 2):
        for c in COHORTS:
            for method in METHODS:
                budget.check()
                context = dict(
                    cohort=c,
                    method=method,
                    dimension=dim,
                    target_set="valence" if dim == 1 else "VA",
                    analysis="nested_layer_and_setting_selection",
                    inference_status="descriptive_no_permutation_test",
                    layer_set="|".join(map(str, layers)),
                )
                try:
                    result = evaluate_layers(
                        {layer: candidates[dim].get((c, method, layer), {}) for layer in layers},
                        target[:, :dim],
                        layers,
                    )
                    path = f"validation/selected_layers/{c}/{method}/{dim}d"
                    store.json(f"{path}/folds.json", result["folds"])
                    store.json(f"{path}/candidate_lookup.json", result["candidate_lookup"])
                    rows = []
                    for i, emotion in enumerate(labels):
                        for axis in range(dim):
                            rows.append(
                                dict(
                                    emotion=emotion,
                                    axis=("valence", "arousal")[axis],
                                    observed=target[i, axis],
                                    predicted=result["predictions"][i, axis],
                                    training_mean=result["training_means"][i, axis],
                                    selected_layer=result["folds"][i]["selected_layer"],
                                    selected_setting=result["folds"][i]["selected_setting"],
                                )
                            )
                    store.csv(
                        f"{path}/predictions.csv",
                        context_frame(pd.DataFrame(rows), dict(base, **context)),
                    )
                    selected_rows.append(
                        dict(context, cv_r2=result["cv_r2"], status="ok", reason="")
                    )
                except UnsupportedGeometry as exc:
                    selected_rows.append(
                        dict(context, cv_r2=np.nan, status="unsupported", reason=str(exc))
                    )
    differences, errors = contrasts(scores, evaluations)
    if len(errors):
        errors["emotion"] = errors.emotion_index.map(dict(enumerate(labels)))
    rdm_contrasts = []
    for layer in layers:
        for a, b in itertools.combinations(COHORTS, 2):
            if (a, layer) in preprocessing and (b, layer) in preprocessing:
                pa, pb = preprocessing[a, layer], preprocessing[b, layer]
                for i, j in itertools.combinations(range(len(labels)), 2):
                    rdm_contrasts.append(
                        dict(
                            layer=layer,
                            cohort_left=a,
                            cohort_right=b,
                            emotion_i=labels[i],
                            emotion_j=labels[j],
                            distance_left=pa["D"][i, j],
                            distance_right=pb["D"][i, j],
                            delta_distance=pa["D"][i, j] - pb["D"][i, j],
                        )
                    )
    for record in metric_rows:
        record["value"] = float(record["value"])
    tables = dict(
        selected_layer_scores=pd.DataFrame(selected_rows),
        scores=scores,
        aggregates=aggregates,
        cka=selected_cka(scores, coordinates),
        contrasts=differences,
        paired_errors=errors,
        metrics=pd.DataFrame(metric_rows),
        fidelity_aggregates=layerwise_metric_means(pd.DataFrame(metric_rows), layers)
        if metric_rows
        else pd.DataFrame(),
        coordinates=pd.DataFrame(coordinate_rows),
        spectra=pd.DataFrame(spectrum_rows),
        pairwise_distances=pd.DataFrame(pair_rows),
        persistence=pd.DataFrame(topology_rows),
        betti=pd.DataFrame(betti_rows),
        graph_ratios=pd.DataFrame(ratio_rows),
        failures=pd.DataFrame(failure_rows),
        rdm_contrasts=pd.DataFrame(rdm_contrasts),
    )
    for name, frame in tables.items():
        if not len(frame) and not len(frame.columns):
            frame = pd.DataFrame(columns=["status", "reason"])
        if "cohort" in frame and "n_samples" not in frame:
            frame["n_samples"] = frame.cohort.map(cohort.samples.groupby("cohort").size())
        if "layer" in frame and "depth" not in frame:
            frame["depth"] = (frame.layer + 1) / len(arrays)
        if "dimension" in frame and "target_set" not in frame:
            frame["target_set"] = frame.dimension.map({0: "none", 1: "valence", 2: "VA"})
        store.csv(f"metrics/{name}.csv", context_frame(frame, base))
    store.json("metrics/emergence.json", emergence(scores, **config["emergence"]))
    store.json(
        "metrics/timing.json",
        dict(
            elapsed_seconds=time.monotonic() - started,
            layers=layers,
            bounded_pilot=config["smoke"],
            peak_rss=_peak_rss(),
        ),
    )
    from affect_geometry.reporting.correctness_geometry import render

    render(store.root)
    store.finish()
    return store.root


def _peak_rss():
    import resource
    import sys

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(value=value, unit="bytes" if sys.platform == "darwin" else "KiB")
