"""Common-emotion comparison on each model's unchanged primary balanced draw."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from affect_geometry.analysis.geometry_comparisons import COHORTS, METHODS, aggregate_scores
from affect_geometry.analysis.geometry_validation import evaluate
from affect_geometry.embeddings.geometry import UnsupportedGeometry
from affect_geometry.persistence.correctness_geometry import GeometryStore, checksum, fingerprint


def compare_common_support(roots, destination):
    # Import locally to avoid coupling low-level calculations to orchestration at import time.
    from affect_geometry.experiments.correctness_geometry import _decompose, settings

    roots = [Path(root) for root in roots]
    manifests = [json.loads((root / "manifest.json").read_text()) for root in roots]
    labels = (
        sorted(set.intersection(*(set(m["emotion_order"]) for m in manifests))) if manifests else []
    )
    store = GeometryStore(
        Path(destination) / "common_support",
        dict(
            sources={str(root): checksum(root / "manifest.json") for root in roots},
            emotion_order=labels,
            sample_policy="unchanged_model_specific_balanced_draws",
            comparison="descriptive_common_emotion_refit",
            preprocessing="recenter_common_non_neutral_centroids_then_saved_cohort_neutral_projection",
        ),
    )
    if "neutral" not in labels or len(labels) < 5:
        store.json("status.json", dict(status="unsupported", reason="insufficient_common_emotions"))
        return
    records, aggregates, sample_sets = [], [], []
    for root, manifest in zip(roots, manifests, strict=True):
        model = manifest["model"]
        config = manifest["config"]
        order = [manifest["emotion_order"].index(label) for label in labels]
        targets = (
            pd.read_csv(root / "validation/targets.csv")
            .set_index("emotion")
            .loc[labels, ["valence", "arousal"]]
            .to_numpy()
        )
        samples = pd.read_csv(root / "cohorts/samples.csv")
        samples = samples.loc[samples.emotion.isin(labels)]
        store.csv(f"{model}/samples.csv", samples)
        sample_sets.append(
            dict(
                model=model,
                sample_ids=sorted(samples.sample_id.unique()),
                source_emotions=manifest["emotion_order"],
                common_emotions=labels,
            )
        )
        model_records = []
        for cohort in COHORTS:
            for layer in manifest["layers"]:
                cells = {dim: {method: {} for method in METHODS} for dim in (1, 2)}
                prepath = root / f"preprocessing/{cohort}/layer_{layer:03d}.npz"
                try:
                    if not prepath.exists():
                        raise UnsupportedGeometry("missing_primary_preprocessing")
                    with np.load(prepath) as pre:
                        raw = pre["raw_centroids"][order]
                        centered = raw - raw[
                            [i for i, e in enumerate(labels) if e != "neutral"]
                        ].mean(0)
                        projected = centered - (centered @ pre["components"].T) @ pre["components"]
                    norms = np.linalg.norm(projected, axis=1)
                    if np.any(norms <= config["tolerance"]):
                        raise UnsupportedGeometry("near_zero_common_projection")
                    u = projected / norms[:, None]
                    d = np.clip(1 - u @ u.T, 0, 2)
                    np.fill_diagonal(d, 0)
                    store.arrays(
                        f"{model}/{cohort}/{layer}/preprocessing.npz", R=projected, U=u, D=d
                    )
                    for method, setting, params in settings(config):
                        if method == "diffusion_maps" and params["alpha"] != 1.0:
                            continue
                        try:
                            decomposition = _decompose(
                                method, d, params, labels, config["tolerance"]
                            )
                            for dim in (1, 2):
                                try:
                                    cells[dim][method][setting] = decomposition.coordinates(
                                        dim, config["tolerance"]
                                    )
                                except UnsupportedGeometry as exc:
                                    store.json(
                                        f"{model}/{cohort}/{layer}/{method}/{setting}/{dim}d-failure.json",
                                        dict(status="unsupported", reason=str(exc)),
                                    )
                        except UnsupportedGeometry as exc:
                            store.json(
                                f"{model}/{cohort}/{layer}/{method}/{setting}/failure.json",
                                dict(status="unsupported", reason=str(exc)),
                            )
                except UnsupportedGeometry as exc:
                    store.json(
                        f"{model}/{cohort}/{layer}/failure.json",
                        dict(status="unsupported", reason=str(exc)),
                    )
                for dim in (1, 2):
                    for method in METHODS:
                        context = dict(
                            model=model,
                            cohort=cohort,
                            layer=layer,
                            dimension=dim,
                            depth=(layer + 1)
                            / manifest["activations"]
                            .get("metadata", {})
                            .get("n_layers", max(manifest["layers"]) + 1),
                            method=method,
                            emotion_order="|".join(labels),
                            emotion_set_id=fingerprint(labels),
                            K=len(labels),
                            source_run=str(root),
                            target_set="valence" if dim == 1 else "VA",
                        )
                        try:
                            result = evaluate(
                                cells[dim][method], targets[:, :dim], selection=config["selection"]
                            )
                            record = dict(
                                context,
                                cv_r2=result["cv_r2"],
                                full_r2=result["full_r2"],
                                status="ok",
                                reason="",
                            )
                            store.json(
                                f"{model}/{cohort}/{layer}/{method}/{dim}d/evaluation.json", result
                            )
                        except UnsupportedGeometry as exc:
                            record = dict(
                                context,
                                cv_r2=np.nan,
                                full_r2=np.nan,
                                status="unsupported",
                                reason=str(exc),
                            )
                        model_records.append(record)
        records.extend(model_records)
        summary = aggregate_scores(pd.DataFrame(model_records), manifest["layers"])
        summary["model"], summary["emotion_order"], summary["K"] = (
            model,
            "|".join(labels),
            len(labels),
        )
        aggregates.extend(summary.to_dict("records"))
    store.csv("scores.csv", pd.DataFrame(records))
    store.csv("aggregates.csv", pd.DataFrame(aggregates))
    store.json("sample_sets.json", sample_sets)
    store.finish()
