from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from affect_geometry.persistence.layout import safe_slug

AUTO_RUN_ID = "auto"


def timestamp_run_id(now: datetime | None = None) -> str:
    """Return a UTC, lexically sortable, filesystem-safe run identifier."""
    instant = now or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    return instant.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def validate_run_id(run_id: str) -> str:
    """Require a non-reserved run ID that is safe as one directory segment."""
    if (
        not run_id
        or run_id == AUTO_RUN_ID
        or Path(run_id).name != run_id
        or run_id in {".", ".."}
        or safe_slug(run_id) != run_id.lower()
    ):
        raise ValueError(
            "run_id must be a single filesystem-safe name; use a UTC timestamp such as "
            "20261018T153045Z"
        )
    return run_id


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "google-research-datasets/go_emotions"
    subset: str = "simplified"
    revision: str | None = None
    splits: list[str] = field(default_factory=lambda: ["train", "validation", "test"])
    include_neutral: bool = True


@dataclass(frozen=True)
class ModelConfig:
    name: str = "Qwen/Qwen3-0.6B"
    revision: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    batch_size: int = 8
    max_length: int = 256


@dataclass(frozen=True)
class ActivationConfig:
    poolings: list[str] = field(default_factory=lambda: ["last_token", "mean"])
    storage_dtype: str = "float16"


@dataclass(frozen=True)
class RDMConfig:
    split: str = "all"
    pca_components: int = 64
    max_samples_per_emotion: int = 256
    crossnobis_folds: int = 5
    mmd_bandwidth_multipliers: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    sliced_wasserstein_projections: int = 256
    distances: list[str] = field(
        default_factory=lambda: [
            "crossnobis",
            "rbf_mmd",
            "energy",
            "sliced_wasserstein",
            "cosine",
            "euclidean",
        ]
    )


@dataclass(frozen=True)
class EmbeddingConfig:
    dimensions: list[int] = field(default_factory=lambda: [1, 2, 3])
    intrinsic_max_dimensions: int = 10
    isomap_neighbors: int = 5
    isomap_neighbor_grid: list[int] = field(default_factory=lambda: [3, 4, 5, 6, 8, 10, 12])
    quality_neighbors: int = 5


@dataclass(frozen=True)
class AnalysisConfig:
    va_reference: str = "data/reference/goemotions_nrc_va.csv"
    va_permutations: int = 1000
    knn_neighbors: int = 5


@dataclass(frozen=True)
class ExperimentConfig:
    run_id: str = AUTO_RUN_ID
    output_root: str = "outputs"
    log_root: str = "logs"
    seed: int = 17
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    activations: ActivationConfig = field(default_factory=ActivationConfig)
    rdm: RDMConfig = field(default_factory=RDMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / safe_slug(self.model.name) / validate_run_id(self.run_id)

    @property
    def log_identifier(self) -> str:
        return f"{safe_slug(self.model.name)}__{validate_run_id(self.run_id)}"

    @property
    def log_dir(self) -> Path:
        return Path(self.log_root) / self.log_identifier

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(
    path: str | Path,
    *,
    run_id: str | None = None,
    generate_run_id: bool = False,
) -> ExperimentConfig:
    raw = json.loads(Path(path).read_text())
    requested_run_id = run_id if run_id is not None else raw.get("run_id", AUTO_RUN_ID)
    if requested_run_id == AUTO_RUN_ID:
        if not generate_run_id:
            raise ValueError(
                "This config uses run_id='auto'. Supply --run-id to a standalone stage so all "
                "stages target the same run, or use python -m run to generate one ID for the "
                "complete pipeline."
            )
        requested_run_id = timestamp_run_id()
    requested_run_id = validate_run_id(requested_run_id)
    return ExperimentConfig(
        run_id=requested_run_id,
        output_root=raw.get("output_root", "outputs"),
        log_root=raw.get("log_root", "logs"),
        seed=raw.get("seed", 17),
        dataset=DatasetConfig(**raw.get("dataset", {})),
        model=ModelConfig(**raw.get("model", {})),
        activations=ActivationConfig(**raw.get("activations", {})),
        rdm=RDMConfig(**raw.get("rdm", {})),
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        analysis=AnalysisConfig(**raw.get("analysis", {})),
    )
