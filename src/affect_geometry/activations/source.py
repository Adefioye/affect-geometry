from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_SAMPLE_COLUMNS = frozenset({"sample_id", "split", "label_id", "emotion"})


@dataclass(frozen=True)
class ActivationSource:
    """Validated layerwise activations and their row-to-sample mapping."""

    root: Path
    pooling: str
    samples: pd.DataFrame
    layer_paths: tuple[Path, ...]
    metadata: dict[str, Any]
    manifest: dict[str, Any]

    @classmethod
    def open(cls, root: str | Path, pooling: str) -> ActivationSource:
        root = Path(root)
        samples_path = root / "dataset" / "samples.csv"
        metadata_path = root / "activations" / "metadata.json"
        manifest_path = root / "manifest.json"
        pooling_path = root / "activations" / pooling

        for path in (samples_path, metadata_path, manifest_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required activation artifact is missing: {path}")
        if not pooling_path.is_dir():
            raise FileNotFoundError(f"Activation pooling directory is missing: {pooling_path}")

        samples = pd.read_csv(samples_path, dtype={"sample_id": str, "source_id": str})
        missing_columns = sorted(REQUIRED_SAMPLE_COLUMNS - set(samples.columns))
        if missing_columns:
            raise ValueError(f"samples.csv is missing required columns: {missing_columns}")
        if samples["sample_id"].isna().any():
            raise ValueError("samples.csv contains missing sample_id values")
        duplicates = samples.loc[samples["sample_id"].duplicated(), "sample_id"].unique()
        if len(duplicates):
            preview = ", ".join(map(str, duplicates[:5]))
            raise ValueError(f"samples.csv contains duplicate sample_id values: {preview}")

        metadata = json.loads(metadata_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        layer_paths = tuple(sorted(pooling_path.glob("layer_*.npy")))
        if not layer_paths:
            raise FileNotFoundError(f"No layer activation files found under {pooling_path}")

        expected_layers = int(metadata["n_layers"])
        expected_samples = int(metadata["n_samples"])
        expected_hidden = int(metadata["hidden_size"])
        if len(samples) != expected_samples:
            raise ValueError(
                f"Sample manifest has {len(samples)} rows; metadata declares {expected_samples}"
            )
        if len(layer_paths) != expected_layers:
            raise ValueError(
                f"Pooling {pooling!r} has {len(layer_paths)} layers; "
                f"metadata declares {expected_layers}"
            )

        expected_names = [f"layer_{layer:03d}.npy" for layer in range(expected_layers)]
        actual_names = [path.name for path in layer_paths]
        if actual_names != expected_names:
            raise ValueError(
                f"Pooling {pooling!r} layer files are not a contiguous zero-based sequence"
            )

        configured_poolings = manifest.get("config", {}).get("activations", {}).get("poolings", [])
        if pooling not in configured_poolings:
            raise ValueError(
                f"Pooling {pooling!r} is not declared in the run manifest: {configured_poolings}"
            )

        expected_shape = (expected_samples, expected_hidden)
        for path in layer_paths:
            array = np.load(path, mmap_mode="r")
            if array.shape != expected_shape:
                raise ValueError(
                    f"Unexpected activation shape in {path}: {array.shape}; "
                    f"expected {expected_shape}"
                )
            if not np.issubdtype(array.dtype, np.floating):
                raise ValueError(f"Activation array is not floating point: {path} ({array.dtype})")

        return cls(
            root=root,
            pooling=pooling,
            samples=samples,
            layer_paths=layer_paths,
            metadata=metadata,
            manifest=manifest,
        )

    @property
    def n_layers(self) -> int:
        return int(self.metadata["n_layers"])

    @property
    def n_samples(self) -> int:
        return int(self.metadata["n_samples"])

    @property
    def hidden_size(self) -> int:
        return int(self.metadata["hidden_size"])

    @property
    def model_name(self) -> str:
        return str(self.metadata["model"])

    @property
    def model_revision(self) -> str | None:
        resolved = self.metadata.get("resolved_model_revision")
        if resolved:
            return str(resolved)
        configured = self.metadata.get("model_revision")
        return None if configured in {None, "default"} else str(configured)

    def load_layer(
        self,
        layer: int,
        *,
        row_order: np.ndarray | None = None,
        dtype: np.dtype | str | None = None,
    ) -> np.ndarray:
        if not 0 <= layer < self.n_layers:
            raise IndexError(f"Layer {layer} is outside [0, {self.n_layers})")
        values = np.load(self.layer_paths[layer], mmap_mode="r")
        if row_order is not None:
            row_order = np.asarray(row_order)
            if row_order.shape != (self.n_samples,):
                raise ValueError(
                    f"row_order must have shape {(self.n_samples,)}, got {row_order.shape}"
                )
            values = values[row_order]
        if dtype is not None:
            values = np.asarray(values, dtype=dtype)
        return values


@dataclass(frozen=True)
class ActivationAlignment:
    """Row ordering that aligns a candidate activation source to a reference source."""

    reference: ActivationSource
    candidate: ActivationSource
    candidate_row_order: np.ndarray

    @property
    def samples(self) -> pd.DataFrame:
        return self.reference.samples


def align_activation_sources(
    reference: ActivationSource,
    candidate: ActivationSource,
) -> ActivationAlignment:
    """Validate compatible runs and align ``candidate`` rows by stable sample ID."""

    if reference.n_layers != candidate.n_layers:
        raise ValueError(f"Layer count mismatch: {reference.n_layers} != {candidate.n_layers}")
    if reference.hidden_size != candidate.hidden_size:
        raise ValueError(
            f"Hidden-size mismatch: {reference.hidden_size} != {candidate.hidden_size}"
        )
    if reference.model_name != candidate.model_name:
        raise ValueError(f"Model mismatch: {reference.model_name!r} != {candidate.model_name!r}")
    if (
        reference.model_revision is not None
        and candidate.model_revision is not None
        and reference.model_revision != candidate.model_revision
    ):
        raise ValueError(
            f"Model-revision mismatch: {reference.model_revision!r} != {candidate.model_revision!r}"
        )

    reference_ids = reference.samples["sample_id"].astype(str)
    candidate_ids = candidate.samples["sample_id"].astype(str)
    reference_set, candidate_set = set(reference_ids), set(candidate_ids)
    missing = sorted(reference_set - candidate_set)
    extra = sorted(candidate_set - reference_set)
    if missing or extra:
        raise ValueError(
            "Activation sources contain different sample IDs "
            f"(missing={missing[:5]}, extra={extra[:5]})"
        )

    candidate_positions = pd.Series(
        np.arange(len(candidate.samples), dtype=np.int64),
        index=candidate_ids,
    )
    order = candidate_positions.loc[reference_ids].to_numpy(dtype=np.int64)
    aligned_candidate = candidate.samples.iloc[order].reset_index(drop=True)
    reference_samples = reference.samples.reset_index(drop=True)
    for column in ("emotion", "split", "label_id"):
        left = reference_samples[column].astype(str).to_numpy()
        right = aligned_candidate[column].astype(str).to_numpy()
        mismatch = np.flatnonzero(left != right)
        if len(mismatch):
            row = int(mismatch[0])
            raise ValueError(
                f"Sample metadata mismatch for {reference_ids.iloc[row]!r} in column "
                f"{column!r}: {left[row]!r} != {right[row]!r}"
            )

    return ActivationAlignment(
        reference=reference,
        candidate=candidate,
        candidate_row_order=order,
    )
