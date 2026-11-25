"""Validate and publish model-specific valence-transfer results."""

import hashlib
import json

import pandas as pd

from affect_geometry.experiments.valence_transfer import COHORT_NAMES, DATASETS, MODELS
from affect_geometry.persistence.correctness_geometry import checksum
from affect_geometry.persistence.exp3_hub import repo_path

REPOSITORIES = {
    "gemma2": "kokolamba/exp3-gemma2",
    "apertus": "kokolamba/exp3-apertus",
    "olmo2": "kokolamba/exp3-olmo2",
}


def _read_table(path, **identity):
    table = pd.read_csv(path)
    if table.empty or any(not table[key].eq(value).all() for key, value in identity.items()):
        raise ValueError(f"Missing or mismatched rows in {path}")
    return table


def plan_uploads(run_dir, models):
    """Return the exact files and model-specific metadata for each selected model."""
    directions = run_dir / "directions"
    selection = json.loads((directions / "manifest.json").read_text())
    evaluation = json.loads((run_dir / "evaluation/manifest.json").read_text())
    if selection["status"] != "completed":
        raise ValueError(f"PC1 selection is incomplete in {run_dir}")
    # Evaluation batch size can change after selection freezes the directions.
    selection_settings = selection["config"].copy()
    evaluation_settings = evaluation["config"].copy()
    for settings in (selection_settings, evaluation_settings):
        settings.pop("batch_size", None)
    if selection_settings != evaluation_settings:
        raise ValueError(f"Selection and evaluation settings differ in {run_dir}")
    selection_path = directions / "selection.csv"
    if checksum(selection_path) != selection["selection_sha256"]:
        raise ValueError(f"Selection table changed after freezing: {selection_path}")
    table = pd.read_csv(selection_path, dtype=str, keep_default_na=False)
    datasets = {spec["name"]: spec for spec in evaluation["config"]["datasets"]}
    if datasets.keys() != DATASETS:
        raise ValueError(f"Expected SST, IMDb, DynaSent R1 and R2 in {run_dir}")
    return {
        model: _model_files(run_dir, model, table, selection, evaluation, datasets)
        for model in models
    }


def _model_files(run_dir, model, table, selection, evaluation, datasets):
    source = selection["sources"][model]
    selected = table.loc[table.model.eq(model)]
    if (
        source["model_id"] != MODELS[model]
        or len(selected) != len(COHORT_NAMES)
        or set(selected.cohort) != set(COHORT_NAMES)
        or not selected.status.eq("ok").all()
        or not selected.model_id.eq(source["model_id"]).all()
        or not selected.model_revision.eq(source["model_revision"]).all()
    ):
        raise ValueError(f"Expected one frozen correct and incorrect PC1 direction for {model}")

    files = []
    layers = {}
    for row in selected.itertuples(index=False):
        cohort = row.cohort
        layer = int(row.layer)
        layers[cohort] = layer
        cohort_dir = run_dir / "directions" / model / cohort
        scores_path = cohort_dir / "layer_scores.csv"
        emotion_path = cohort_dir / "emotion_scores.csv"
        scores = _read_table(scores_path, model=model, cohort=cohort)
        emotion_scores = _read_table(emotion_path, model=model, cohort=cohort)
        supported = set(scores.loc[scores.status.eq("ok"), "layer"])
        if layer not in supported or set(emotion_scores.layer) != supported:
            raise ValueError(f"Inconsistent PC1 layers in {cohort_dir}")
        for supported_layer in sorted(supported):
            checkpoint = cohort_dir / f"layer_{supported_layer:03d}.npz"
            if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                raise ValueError(f"Missing PC1 checkpoint: {checkpoint}")
            files.append(checkpoint)
        frozen = cohort_dir / f"layer_{layer:03d}.npz"
        if (
            row.artifact != frozen.relative_to(run_dir / "directions").as_posix()
            or checksum(frozen) != row.artifact_sha256
        ):
            raise ValueError(f"Frozen PC1 checkpoint changed: {frozen}")
        files.extend([scores_path, emotion_path])

    config = evaluation["config"]
    controls = {"pc1", "no_op", "full"}
    controls.update(f"random_{i}" for i in range(config["random_directions"]))
    summaries = {
        (control, subset)
        for control in controls
        for subset in ("all_pairs", "both_baselines_correct")
    }
    for dataset, spec in datasets.items():
        dataset_dir = run_dir / "evaluation" / model / dataset
        for cohort, layer in layers.items():
            path = dataset_dir / f"{cohort}_metrics.csv"
            metrics = _read_table(path, model=model, dataset=dataset, cohort=cohort, layer=layer)
            keys = set(zip(metrics.control, metrics.subset))
            if keys != summaries or len(metrics) != len(summaries):
                raise ValueError(f"Incomplete OOD metrics in {path}")
            files.append(path)
        path = dataset_dir / "provenance.json"
        provenance = json.loads(path.read_text())
        if provenance["dataset"] != spec or any(
            provenance[key] != source[key] for key in ("model_id", "model_revision")
        ):
            raise ValueError(f"Mismatched OOD provenance in {path}")
        files.append(path)

    selection_csv = selected.drop(columns="bundle").to_csv(index=False).encode()
    manifest = {
        "model": model,
        "run_id": run_dir.name,
        "source": {key: value for key, value in source.items() if key != "bundle"},
        "selection": {
            "config_sha256": selection["config_sha256"],
            "nrc_sha256": selection["nrc_sha256"],
            "selection_sha256": hashlib.sha256(selection_csv).hexdigest(),
            "versions": selection["versions"],
        },
        "evaluation": {
            "config_sha256": evaluation["config_sha256"],
            "versions": evaluation["versions"],
            "batch_size": evaluation["batch_size"],
            **{
                key: config.get(key)
                for key in ("seed", "max_tokens", "max_pairs", "random_directions")
            },
        },
    }
    return {
        **{path.relative_to(run_dir).as_posix(): path for path in files},
        "directions/selection.csv": selection_csv,
        "manifest.json": (json.dumps(manifest, indent=2) + "\n").encode(),
    }


def publish_model(api, model, prefix, files):
    """Publish the entire model package in one dataset commit."""
    from huggingface_hub import CommitOperationAdd

    repository = REPOSITORIES[model]
    prefix = repo_path(prefix)
    operations = [
        CommitOperationAdd(f"{prefix}/{name}", content) for name, content in files.items()
    ]
    api.create_repo(repo_id=repository, repo_type="dataset", private=True, exist_ok=True)
    return api.create_commit(
        repo_id=repository,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Upload {model} valence transfer results to {prefix}",
    )
