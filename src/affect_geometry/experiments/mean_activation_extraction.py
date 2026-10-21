from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from affect_geometry.activations.extract import extract_activations
from affect_geometry.activations.source import ActivationSource
from affect_geometry.config import (
    ActivationConfig,
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    validate_run_id,
)
from affect_geometry.data.goemotions import load_samples
from affect_geometry.logging import get_logger
from affect_geometry.pipeline import initialize_run


@dataclass(frozen=True)
class MeanActivationExtractionResult:
    run_directory: Path
    model_name: str
    run_id: str
    n_samples: int
    n_layers: int
    hidden_size: int


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required reference artifact is missing: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_artifacts(reference_run: str | Path) -> tuple[Path, dict, dict]:
    root = Path(reference_run).expanduser().resolve()
    manifest = _read_json(root / "manifest.json")
    metadata = _read_json(root / "activations" / "metadata.json")
    samples_path = root / "dataset" / "samples.csv"
    if not samples_path.is_file():
        raise FileNotFoundError(f"Required reference artifact is missing: {samples_path}")
    return root, manifest, metadata


def mean_extraction_config(
    reference_run: str | Path,
    *,
    run_id: str,
    output_root: str | Path = "outputs",
    log_root: str | Path = "logs",
    device: str | None = None,
    batch_size: int | None = None,
    model_revision: str | None = None,
) -> ExperimentConfig:
    """Build a mean-only extraction config from an existing activation run."""

    _, manifest, metadata = _reference_artifacts(reference_run)
    try:
        recorded = manifest["config"]
        model_values = recorded["model"]
    except (KeyError, TypeError) as error:
        raise ValueError("Reference manifest does not contain config.model") from error

    resolved_revision = metadata.get("resolved_model_revision")
    configured_revision = model_values.get("revision")
    recorded_revision = resolved_revision or configured_revision
    if recorded_revision and model_revision and recorded_revision != model_revision:
        raise ValueError("Explicit model revision differs from the recorded reference revision")
    model_revision = recorded_revision or model_revision
    if not model_revision:
        raise ValueError(
            "The reference run does not record a resolved or configured model revision; "
            "supply an explicit model_revision to extract without a verified reference revision"
        )
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")

    model = ModelConfig(**model_values)
    model = replace(
        model,
        revision=str(model_revision),
        device=device or model.device,
        batch_size=batch_size if batch_size is not None else model.batch_size,
    )
    activation_values = recorded.get("activations", {})
    activations = ActivationConfig(
        poolings=["mean"],
        storage_dtype=activation_values.get("storage_dtype", "float16"),
    )
    dataset = DatasetConfig(**recorded.get("dataset", {}))
    return ExperimentConfig(
        run_id=validate_run_id(run_id),
        output_root=str(output_root),
        log_root=str(log_root),
        seed=int(recorded.get("seed", 17)),
        dataset=dataset,
        model=model,
        activations=activations,
    )


def extract_mean_pooled_activations(
    reference_run: str | Path,
    config: ExperimentConfig,
) -> MeanActivationExtractionResult:
    """Extract and validate mean-pooled activations using a reference run's sample order."""

    logger = get_logger()
    reference_root, reference_manifest, reference_metadata = _reference_artifacts(reference_run)
    if config.activations.poolings != ["mean"]:
        raise ValueError("Mean extraction requires activations.poolings to be exactly ['mean']")

    reference_model = reference_metadata.get("model")
    if reference_model != config.model.name:
        raise ValueError(
            f"Reference/config model mismatch: {reference_model!r} != {config.model.name!r}"
        )
    destination = config.run_dir.resolve()
    if destination == reference_root:
        raise ValueError("Mean extraction destination must differ from the reference run")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Mean extraction destination is not empty: {destination}")

    logger.info(
        "stage.start stage=mean_activation_extraction reference=%s destination=%s",
        reference_root,
        destination,
    )
    layout = initialize_run(config)
    for source in sorted((reference_root / "dataset").rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(reference_root / "dataset")
        target = layout.dataset / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    reference_samples_path = reference_root / "dataset" / "samples.csv"
    copied_samples_path = layout.dataset / "samples.csv"
    reference_samples_hash = _file_sha256(reference_samples_path)
    if _file_sha256(copied_samples_path) != reference_samples_hash:
        raise RuntimeError("Copied samples.csv does not exactly match the reference run")

    samples = load_samples(copied_samples_path)
    reference_revision = (
        reference_metadata.get("resolved_model_revision")
        or reference_manifest.get("config", {}).get("model", {}).get("revision")
    )
    if not reference_revision:
        logger.warning(
            "activations.reference_revision_unknown extraction_revision=%s "
            "reference_model_identity_verified=false",
            config.model.revision,
        )
    metadata = extract_activations(
        samples,
        layout.activations,
        config.model,
        config.activations,
    )
    metadata.update(
        {
            "reference_run_id": reference_manifest.get("config", {}).get(
                "run_id", reference_root.name
            ),
            "reference_samples_sha256": reference_samples_hash,
            "pooling": "mean",
            "reference_model_revision": reference_revision,
            "reference_model_identity_verified": reference_revision is not None,
        }
    )
    layout.write_json("activations/metadata.json", metadata)

    source = ActivationSource.open(layout.root, "mean")
    expected_shape = (
        int(reference_metadata["n_samples"]),
        int(reference_metadata["n_layers"]),
        int(reference_metadata["hidden_size"]),
    )
    actual_shape = (source.n_samples, source.n_layers, source.hidden_size)
    if actual_shape != expected_shape:
        raise ValueError(
            "Mean activation dimensions do not match the reference run: "
            f"{actual_shape} != {expected_shape}"
        )
    expected_revision = str(reference_revision or config.model.revision)
    if source.model_revision != expected_revision:
        raise ValueError(
            "Mean activation model revision does not match the reference run: "
            f"{source.model_revision!r} != {expected_revision!r}"
        )
    if _file_sha256(copied_samples_path) != reference_samples_hash:
        raise RuntimeError("Sample ordering changed during mean activation extraction")

    logger.info(
        "stage.complete stage=mean_activation_extraction samples=%d layers=%d hidden_size=%d",
        source.n_samples,
        source.n_layers,
        source.hidden_size,
    )
    return MeanActivationExtractionResult(
        run_directory=layout.root,
        model_name=source.model_name,
        run_id=config.run_id,
        n_samples=source.n_samples,
        n_layers=source.n_layers,
        hidden_size=source.hidden_size,
    )
