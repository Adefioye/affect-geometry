"""Two-stage centroid-PC1 selection and frozen OOD directional patching."""

import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from affect_geometry.activations.exp3_bundle import COHORT_NAMES
from affect_geometry.activations.valence_source import open_source, validate_source
from affect_geometry.analysis.valence_direction import fit_centroid_pc1
from affect_geometry.config import ModelConfig
from affect_geometry.data.sentiment_transfer import load_candidates, make_pairs
from affect_geometry.embeddings.geometry import UnsupportedGeometry
from affect_geometry.evaluation.directional_patching import evaluate_pairs, summarize
from affect_geometry.logging import configure_file_logging, get_logger, progress
from affect_geometry.models.activation_patching import LastTokenPatcher
from affect_geometry.models.huggingface import load_causal_lm
from affect_geometry.persistence.correctness_geometry import checksum

MODELS = {
    "gemma2": "google/gemma-2-9b",
    "apertus": "swiss-ai/Apertus-8B-Instruct-2509",
    "olmo2": "allenai/OLMo-2-1124-7B-Instruct",
}
DATASETS = {"sst", "imdb", "dynasent_r1", "dynasent_r2"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    root = path.parent.parent
    cfg["project_root"] = str(root)
    for key in ("output_dir", "nrc_path"):
        cfg[key] = str((root / cfg[key]).resolve())
    if not Path(cfg["output_dir"]).is_relative_to(root / "outputs"):
        raise ValueError("Run output must be inside project outputs/")
    if set(cfg["models"]) != set(MODELS):
        raise ValueError("Configure exactly the three EXP3 models")
    if {d["name"] for d in cfg["datasets"]} != DATASETS or len(cfg["datasets"]) != 4:
        raise ValueError("Expected SST, IMDb, DynaSent R1 and R2")
    if cfg.get("random_directions", 5) < 1 or cfg.get("max_tokens", 128) < 1:
        raise ValueError("Positive random control count and token budget required")
    for model in cfg["models"].values():
        validate_source(model)
        if model.get("bundle"):
            model["bundle"] = str((root / model["bundle"]).resolve())
    for spec in cfg["datasets"]:
        if spec.get("path"):
            spec["path"] = str((root / spec["path"]).resolve())
    return cfg


def _settings_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def _manifest(cfg, phase):
    versions = {}
    for package in ("numpy", "pandas", "scipy", "torch", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return dict(
        phase=phase,
        status="running",
        config=cfg,
        config_sha256=_settings_hash(cfg),
        versions=versions,
    )


def select_directions(cfg):
    """Reuse checksummed EXP3 projections; select max |Pearson(PC1, NRC V)|."""
    destination = Path(cfg["output_dir"]) / "directions"
    destination.mkdir(parents=True, exist_ok=False)
    manifest = _manifest(cfg, "selection")
    write_json(destination / "manifest.json", manifest)
    references = pd.read_csv(cfg["nrc_path"])
    if references.emotion.duplicated().any():
        raise ValueError("Duplicate NRC emotion rows")
    references = references.set_index("emotion")
    selected = []
    sources = {}
    try:
        for model, settings in cfg["models"].items():
            configure_file_logging(
                Path(cfg["project_root"]) / "logs" / f"{model}-valence-transfer",
                "select",
                model=model,
            )
            bundle, remote = open_source(settings, model, cfg["project_root"])
            if bundle.manifest["model_id"] != MODELS[model]:
                raise ValueError("Unexpected EXP3 model ID")
            if bundle.manifest.get("projected_states_are_unit_normalized") is not False:
                raise ValueError("Require unnormalized projected states")
            if bundle.manifest["prompt_metadata"].get("layer_indexing") != "zero_based_post_block":
                raise ValueError("Require zero-based post-block activations")
            labels = bundle.manifest["emotion_order"]
            if "neutral" not in labels:
                raise ValueError("Neutral must be included")
            ratings = references.loc[labels, "valence"].to_numpy(float)
            revision = bundle.manifest["source"].get("model_revision")
            if not revision or revision in {"main", "default"}:
                raise ValueError("Source bundle must record a pinned model revision")
            sources[model] = dict(
                bundle=str(bundle.root),
                manifest_sha256=checksum(bundle.root / "manifest.json"),
                model_id=MODELS[model],
                model_revision=revision,
            )
            if remote is not None:
                sources[model].update(
                    activation_repo_id=remote["repo_id"],
                    activation_revision=remote["revision"],
                    activation_path_in_repo=remote["path_in_repo"],
                )
            for cohort in COHORT_NAMES:
                out = destination / model / cohort
                out.mkdir(parents=True)
                samples = bundle.cohorts.samples.loc[bundle.cohorts.samples.cohort.eq(cohort)]
                rows, scores, best = [], [], None
                for layer in progress(
                    range(bundle.manifest["n_layers"]),
                    description=f"{model}/{cohort} PC1",
                    unit="layer",
                ):
                    try:
                        fit = fit_centroid_pc1(
                            bundle.projected(cohort, layer),
                            samples.emotion.to_numpy(),
                            labels,
                            ratings,
                        )
                    except UnsupportedGeometry as error:
                        rows.append(
                            dict(
                                model=model,
                                cohort=cohort,
                                layer=layer,
                                status="unsupported",
                                reason=str(error),
                            )
                        )
                        continue
                    row = dict(
                        model=model,
                        cohort=cohort,
                        layer=layer,
                        status="ok",
                        pearson_r=fit.pearson_r,
                        pearson_p_descriptive=fit.pearson_p,
                        spearman_r=fit.spearman_r,
                        explained_variance=fit.explained_variance,
                        relative_eigengap=fit.relative_eigengap,
                        n_emotions=len(labels),
                    )
                    rows.append(row)
                    for emotion, score, rating, count in zip(
                        labels, fit.scores, ratings, fit.counts
                    ):
                        scores.append(
                            dict(
                                model=model,
                                cohort=cohort,
                                layer=layer,
                                emotion=emotion,
                                pc1_score=score,
                                valence=rating,
                                n_samples=int(count),
                            )
                        )
                    np.savez(
                        out / f"layer_{layer:03d}.npz",
                        direction=fit.vector,
                        center=fit.center,
                        scores=fit.scores,
                        valence=ratings,
                        emotion_order=np.asarray(labels),
                        counts=fit.counts,
                    )
                    if best is None or fit.pearson_r > best["pearson_r"]:
                        best = row
                pd.DataFrame(rows).to_csv(out / "layer_scores.csv", index=False)
                pd.DataFrame(scores).to_csv(out / "emotion_scores.csv", index=False)
                if best is None:
                    raise ValueError(f"No supported layers for {model}/{cohort}")
                artifact = out / f"layer_{best['layer']:03d}.npz"
                selected.append(
                    {
                        **best,
                        **sources[model],
                        "artifact": str(artifact.relative_to(destination)),
                        "artifact_sha256": checksum(artifact),
                    }
                )
                get_logger().info(
                    "selected model=%s cohort=%s layer=%s r=%s",
                    model,
                    cohort,
                    best["layer"],
                    best["pearson_r"],
                )
        pd.DataFrame(selected).to_csv(destination / "selection.csv", index=False)
        manifest.update(
            status="completed",
            sources=sources,
            nrc_sha256=checksum(Path(cfg["nrc_path"])),
            selection_sha256=checksum(destination / "selection.csv"),
            selection_interpretation="descriptive maximum over all source layers",
        )
        write_json(destination / "manifest.json", manifest)
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        write_json(destination / "manifest.json", manifest)
        raise
    return destination


def evaluate_frozen(cfg, *, model_loader=load_causal_lm, candidate_loader=load_candidates):
    """Evaluate all nine frozen directions; never refit/select with OOD outcomes."""
    source = Path(cfg["output_dir"]) / "directions"
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] != "completed" or manifest["config_sha256"] != _settings_hash(cfg):
        raise ValueError("Require completed selection with identical frozen configuration")
    if checksum(source / "selection.csv") != manifest["selection_sha256"]:
        raise ValueError("Selection table changed after freezing")
    selection = pd.read_csv(source / "selection.csv")
    if len(selection) != 9 or selection.duplicated(["model", "cohort"]).any():
        raise ValueError("Expected exactly nine frozen model/cohort directions")
    destination = Path(cfg["output_dir"]) / "evaluation"
    destination.mkdir(parents=True, exist_ok=False)
    run = _manifest(cfg, "evaluation")
    write_json(destination / "manifest.json", run)
    try:
        for model_name, settings in cfg["models"].items():
            configure_file_logging(
                Path(cfg["project_root"]) / "logs" / f"{model_name}-valence-transfer",
                "evaluate",
                model=model_name,
            )
            selected = selection.loc[selection.model.eq(model_name)]
            if set(selected.cohort) != set(COHORT_NAMES):
                raise ValueError("Missing frozen cohort")
            identity = manifest["sources"][model_name]
            config = ModelConfig(
                name=identity["model_id"],
                revision=identity["model_revision"],
                device=settings.get("device", "auto"),
                dtype=settings.get("dtype", "auto"),
            )
            model, tokenizer, device = model_loader(config)
            resolved = getattr(model.config, "_commit_hash", None)
            if resolved is not None and resolved != config.revision:
                raise ValueError("Loaded model revision differs from EXP3")
            patcher = LastTokenPatcher(model, tokenizer, device, cfg["max_tokens"])
            for spec in cfg["datasets"]:
                out = destination / model_name / spec["name"]
                out.mkdir(parents=True)
                candidates = candidate_loader(spec)
                pairs, audit = make_pairs(
                    candidates,
                    tokenizer,
                    seed=cfg["seed"],
                    max_tokens=cfg["max_tokens"],
                    max_pairs=cfg.get("max_pairs"),
                )
                audit.to_csv(out / "candidates.csv", index=False)
                pairs.to_csv(out / "pairs.csv", index=False)
                write_json(
                    out / "provenance.json",
                    dict(
                        dataset=spec,
                        candidate_sha256=checksum(out / "candidates.csv"),
                        pairs_sha256=checksum(out / "pairs.csv"),
                        model_id=config.name,
                        model_revision=config.revision,
                        answer_ids=patcher.answer_ids,
                        add_special_tokens=True,
                        patch_position="final",
                        layer_indexing="zero_based_post_block",
                    ),
                )
                for row in selected.to_dict("records"):
                    path = (source / row["artifact"]).resolve()
                    if (
                        not path.is_relative_to(source.resolve())
                        or checksum(path) != row["artifact_sha256"]
                    ):
                        raise ValueError("Direction checkpoint changed after selection")
                    with np.load(path) as artifact:
                        vector = artifact["direction"]
                    if vector.shape != (model.config.hidden_size,) or not np.isclose(
                        np.linalg.norm(vector), 1
                    ):
                        raise ValueError("Wrong direction width or norm")
                    results = evaluate_pairs(
                        patcher,
                        pairs,
                        vector,
                        int(row["layer"]),
                        random_count=cfg["random_directions"],
                        seed=cfg["seed"],
                    )
                    results.insert(0, "cohort", row["cohort"])
                    results.insert(0, "dataset", spec["name"])
                    results.insert(0, "model", model_name)
                    results.to_csv(out / f"{row['cohort']}_cases.csv", index=False)
                    metrics = summarize(results)
                    metrics.insert(0, "layer", int(row["layer"]))
                    metrics.insert(0, "cohort", row["cohort"])
                    metrics.insert(0, "dataset", spec["name"])
                    metrics.insert(0, "model", model_name)
                    metrics.to_csv(out / f"{row['cohort']}_metrics.csv", index=False)
            del patcher, model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        run["status"] = "completed"
        write_json(destination / "manifest.json", run)
    except Exception as error:
        run.update(status="failed", error=str(error))
        write_json(destination / "manifest.json", run)
        raise
    return destination
