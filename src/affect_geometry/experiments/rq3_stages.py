"""Two-stage RQ3 configuration and orchestration; no model loading during analysis."""

import json
import math
from pathlib import Path

from affect_geometry.activations.classification_prompt import (
    extract_prompt_states,
    open_prompt_states,
)
from affect_geometry.activations.rq3_bundle import PreparedActivations, export_cohort_bundle
from affect_geometry.analysis.cohorts import balanced_cohorts
from affect_geometry.classification.protocol import EMOTIONS, MODELS
from affect_geometry.classification.storage import atomic_json
from affect_geometry.data.classification_source import read_classification, read_targets
from affect_geometry.experiments.rq3_budget import RunBudget
from affect_geometry.experiments.rq3_config import PANEL, validate_config
from affect_geometry.logging import configure_file_logging, get_logger, progress
from affect_geometry.persistence.correctness_geometry import checksum
from affect_geometry.persistence.rq3_hub import download_bundle, publish_stage, repo_path

ANALYSIS_KEYS = {
    "run_id",
    "model",
    "activation_source",
    "publish",
    "nrc_path",
    "nrc_provenance",
    "permutation_seed",
    "permutations",
    "smoke",
    "layers",
    "isomap_neighbors",
    "diffusion_neighbors",
    "diffusion_scale_ranks",
    "diffusion_times",
    "diffusion_alphas",
    "selection",
    "neighborhoods",
    "filtration_bound",
    "betti_grid_size",
    "tolerance",
    "settings_frozen",
    "pilot_verified",
    "emergence",
    "budgets",
}


def _name(value):
    if not isinstance(value, str) or not value or Path(value).name != value or value in (".", ".."):
        raise ValueError("Run identifiers must be single filesystem-safe names")


def _publish_settings(settings):
    if set(settings) != {"enabled", "repo_id", "private"}:
        raise ValueError("Publish settings require enabled, repo_id, private")
    if type(settings["enabled"]) is not bool or type(settings["private"]) is not bool:
        raise ValueError("Publish enabled/private must be boolean")
    if not isinstance(settings["repo_id"], str) or settings["repo_id"].count("/") != 1:
        raise ValueError("Publish repo_id must be username/dataset or {username}/dataset")


def load_extraction_config(path):
    config = json.loads(Path(path).read_text())
    if set(config) != {
        "run_id",
        "seed",
        "allow_partial_source",
        "models",
        "extraction",
        "projection",
        "budgets",
    }:
        raise ValueError("Unexpected extraction configuration fields")
    _name(config["run_id"])
    if set(config["models"]) != set(PANEL):
        raise ValueError("Extraction config must include all three RQ3 models")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("Sampling seed must be a nonnegative integer")
    if type(config["allow_partial_source"]) is not bool:
        raise ValueError("allow_partial_source must be boolean")
    for slug, model in config["models"].items():
        if set(model) != {"model_id", "classification_run", "prompt_run", "publish"}:
            raise ValueError(f"Invalid extraction model fields: {slug}")
        if model["model_id"] != MODELS[slug][0]:
            raise ValueError("Extraction model ID mismatch")
        _publish_settings(model["publish"])
    if set(config["projection"]) != {"storage_dtype", "batch_size", "tolerance"}:
        raise ValueError("Projection fields must specify storage_dtype, batch_size, tolerance")
    if config["projection"]["storage_dtype"] not in ("float32", "float64"):
        raise ValueError("Projection storage must preserve at least float32 precision")
    if (
        type(config["projection"]["batch_size"]) is not int
        or config["projection"]["batch_size"] < 1
        or not math.isfinite(config["projection"]["tolerance"])
        or config["projection"]["tolerance"] <= 0
    ):
        raise ValueError("Invalid projection batch size or numerical tolerance")
    if set(config["extraction"]) != {"batch_size", "dtype", "device", "storage_dtype"}:
        raise ValueError("Invalid prompt extraction settings")
    if (
        type(config["extraction"]["batch_size"]) is not int
        or config["extraction"]["batch_size"] < 1
    ):
        raise ValueError("Extraction batch size must be positive")
    if config["extraction"]["dtype"] not in ("float32", "float16", "bfloat16"):
        raise ValueError("Invalid model precision")
    if config["extraction"]["storage_dtype"] not in ("float32", "float64"):
        raise ValueError("Save raw states in at least float32 precision")
    if set(config["budgets"]) != {"max_seconds", "max_rss_gib"} or any(
        not math.isfinite(value) or value <= 0 for value in config["budgets"].values()
    ):
        raise ValueError("Positive finite extraction budgets are required")
    return config


def analysis_geometry_config(config, seed=0):
    """Adapt one model's analysis-only settings to the shared scientific runner."""
    slug = config["model"]
    geometry = {
        k: v for k, v in config.items() if k not in ("model", "activation_source", "publish")
    }
    geometry.update(
        seed=seed,
        extraction={},
        models={
            slug: dict(
                model_id=MODELS[slug][0],
                classification_run="frozen_in_activation_bundle",
                activation_run=config["activation_source"]["local_path"],
            )
        },
    )
    return validate_config(geometry, require_panel=False)


def load_analysis_config(path):
    config = json.loads(Path(path).read_text())
    if set(config) != ANALYSIS_KEYS or config["model"] not in PANEL:
        raise ValueError("Expected one RQ3 model and analysis-only configuration fields")
    source = config["activation_source"]
    if set(source) != {"local_path", "repo_id", "path_in_repo", "revision"}:
        raise ValueError("Activation source requires local_path, repo_id, path_in_repo, revision")
    if not isinstance(source["local_path"], str) or not source["local_path"]:
        raise ValueError("Specify a local bundle path (also used before a Hub fallback)")
    repo_path(source["path_in_repo"])
    if not isinstance(source["revision"], str) or not source["revision"]:
        raise ValueError("Activation source revision is required")
    _publish_settings(config["publish"])
    if source["repo_id"] != config["publish"]["repo_id"]:
        raise ValueError("Activations and analysis must use the same model dataset repository")
    analysis_geometry_config(config)
    return config


def _output_path(project, path):
    path = (project / path).resolve()
    if not path.is_relative_to((project / "outputs").resolve()):
        raise ValueError("Generated artifacts must be under the project outputs directory")
    return path


def run_extraction(config, project_root, *, command="run", publish=True):
    project = Path(project_root).resolve()
    outputs, failures = [], []
    for slug, spec in progress(
        config["models"].items(), description="RQ3 extraction panel", unit="model"
    ):
        configure_file_logging(
            project / "logs" / f"{slug}-{config['run_id']}-extract", "rq3-extract"
        )
        root = project / "outputs/rq3" / slug / config["run_id"] / "activation_bundle"
        try:
            if command == "upload":
                bundle = PreparedActivations.open(root, expected_model=slug)
            else:
                rows, manifest, audit = read_classification(
                    project / spec["classification_run"],
                    spec["model_id"],
                    config["allow_partial_source"],
                )
                cohorts = balanced_cohorts(rows, EMOTIONS, config["seed"])
                if not cohorts.labels:
                    raise ValueError("No supported balanced emotion cohorts")
                selected = rows.loc[rows.sample_id.isin(cohorts.samples.sample_id)].sort_values(
                    "sample_id"
                )
                if command == "validate":
                    outputs.append(dict(model=slug, source=audit, labels=list(cohorts.labels)))
                    continue
                prompt_root = _output_path(project, spec["prompt_run"])
                budget = RunBudget(config["budgets"])
                # Reuse only a completely validated prompt cache. A partial extraction is rerun.
                if (prompt_root / "checkpoints/extraction.json").exists():
                    prompt_samples, paths, metadata, hashes = open_prompt_states(
                        prompt_root, selected, manifest
                    )
                    for key, value in config["extraction"].items():
                        if metadata.get(key) != value:
                            raise ValueError(f"Prompt cache extraction setting differs: {key}")
                else:
                    extract_prompt_states(selected, manifest, prompt_root, **config["extraction"])
                    prompt_samples, paths, metadata, hashes = open_prompt_states(
                        prompt_root, selected, manifest
                    )
                budget.check()
                bundle = export_cohort_bundle(
                    root,
                    model=slug,
                    run_id=config["run_id"],
                    cohorts=cohorts,
                    source_manifest=manifest,
                    source_audit=audit,
                    prompt_samples=prompt_samples,
                    layer_paths=paths,
                    prompt_metadata=metadata,
                    prompt_checksums=hashes,
                    seed=config["seed"],
                    projected_dtype=config["projection"]["storage_dtype"],
                    batch_size=config["projection"]["batch_size"],
                    tolerance=config["projection"]["tolerance"],
                    budget=budget,
                )
            receipt = None
            if publish and spec["publish"]["enabled"]:
                receipt = publish_stage(
                    root,
                    repo_id=spec["publish"]["repo_id"],
                    path_in_repo=f"activations/{config['run_id']}",
                    private=spec["publish"]["private"],
                )
                atomic_json(root.parent / "activation_upload.json", receipt)
            outputs.append(dict(model=slug, path=str(bundle.root), upload=receipt))
        except Exception as exc:
            get_logger().exception("rq3.extraction_failed model=%s", slug)
            failures.append(dict(model=slug, error_type=type(exc).__name__, reason=str(exc)))
    atomic_json(
        project / "outputs/rq3/extraction" / config["run_id"] / f"{command}-status.json",
        dict(models=outputs, failures=failures),
    )
    if failures:
        raise RuntimeError("; ".join(f"{x['model']}: {x['reason']}" for x in failures))
    return outputs


def run_analysis(config, project_root, *, publish=True):
    project = Path(project_root).resolve()
    slug = config["model"]
    configure_file_logging(project / "logs" / f"{slug}-{config['run_id']}-analysis", "rq3-analysis")
    source = config["activation_source"]
    local = (project / source["local_path"]).resolve()
    remote = None
    if local.exists():
        # A broken local bundle must fail validation, never be silently replaced with another draw.
        bundle = PreparedActivations.open(local, expected_model=slug)
        receipt_path = local.parent / "activation_upload.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("manifest_sha256") != checksum(local / "manifest.json"):
                raise ValueError("Local activation upload receipt belongs to another bundle")
            remote = {key: receipt[key] for key in ("repo_id", "revision", "path_in_repo")}
    else:
        local, remote = download_bundle(source, project / "outputs/rq3" / slug / "downloads")
        bundle = PreparedActivations.open(local, expected_model=slug)
    if bundle.manifest["model_id"] != MODELS[slug][0]:
        raise ValueError("Prepared model identifier mismatch")
    if not config["smoke"] and not bundle.manifest["source_audit"].get("full_population", False):
        raise ValueError("Full analysis requires a full-population RQ2 source in the bundle")
    geometry = analysis_geometry_config(config, seed=bundle.manifest["seed"])
    target, nrc = read_targets(
        project / config["nrc_path"], project / config["nrc_provenance"], bundle.cohorts.labels
    )
    samples = bundle.cohorts.samples.loc[bundle.cohorts.samples.cohort.eq("all")].sort_values(
        "sample_id"
    )
    reference = bundle.reference() | dict(hub=remote, analysis_configuration=config)
    from affect_geometry.experiments.correctness_geometry import run_geometry

    root = project / "outputs/rq3" / slug / config["run_id"]
    if local.is_relative_to(root) or root.is_relative_to(local):
        raise ValueError("Analysis output and activation bundle directories must not overlap")
    run_geometry(
        geometry,
        slug,
        samples,
        bundle,
        bundle.manifest["source"],
        bundle.manifest["source_audit"],
        reference,
        target,
        nrc,
        root,
        prepared=bundle,
    )
    receipt = None
    if publish and config["publish"]["enabled"]:
        receipt = publish_stage(
            root,
            repo_id=config["publish"]["repo_id"],
            path_in_repo=f"analyses/{config['run_id']}",
            private=config["publish"]["private"],
        )
        atomic_json(root.parent / f"{config['run_id']}-upload.json", receipt)
    return dict(path=str(root), upload=receipt)


def compare_analyses(roots, project_root, comparison_id, *, publish=True):
    """Optional final comparison after the three independent model jobs have completed."""
    import pandas as pd

    from affect_geometry.analysis.cross_model_geometry import compare_common_support
    from affect_geometry.persistence.correctness_geometry import GeometryStore
    from affect_geometry.reporting.correctness_geometry import compare_models

    _name(comparison_id)
    project = Path(project_root).resolve()
    roots = [(project / Path(root)).resolve() for root in roots]
    manifests = [json.loads((root / "manifest.json").read_text()) for root in roots]
    if len(roots) != 3 or {m["model"] for m in manifests} != set(PANEL):
        raise ValueError("Compare requires one completed run for each RQ3 model")
    for root, manifest in zip(roots, manifests, strict=True):
        store = GeometryStore(root, {k: v for k, v in manifest.items() if k != "schema_version"})
        if not store.valid("run", store.manifest):
            raise ValueError(f"Incomplete/modified analysis run: {root}")
    destination = project / "outputs/rq3/comparisons" / comparison_id
    compare_common_support(roots, destination)
    compare_models(roots, destination)
    info = json.loads((destination / "manifest.json").read_text())
    info.update(
        schema_version=1,
        model="model-comparison",
        source_checksums={str(root): checksum(root / "manifest.json") for root in roots},
    )
    atomic_json(destination / "manifest.json", info)
    comparison = GeometryStore(
        destination, {k: v for k, v in info.items() if k != "schema_version"}
    )
    for name, source in [
        ("model_support", destination / "model_support.csv"),
        ("common_support_aggregates", destination / "common_support/aggregates.csv"),
        ("common_support_scores", destination / "common_support/scores.csv"),
    ]:
        if source.exists():
            comparison.csv(f"plot_data/{name}.csv", pd.read_csv(source))
    comparison.finish()
    receipts = []
    if publish:
        for manifest in manifests:
            settings = manifest["activations"]["analysis_configuration"]["publish"]
            if settings["enabled"]:
                receipts.append(
                    publish_stage(
                        destination,
                        repo_id=settings["repo_id"],
                        path_in_repo=f"comparisons/{comparison_id}",
                        private=settings["private"],
                    )
                )
        atomic_json(destination.parent / f"{comparison_id}-uploads.json", receipts)
    return dict(path=str(destination), uploads=receipts)
