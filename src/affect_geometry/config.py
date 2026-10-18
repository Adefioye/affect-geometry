from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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
    run_id: str = "qwen3_0.6b_goemotions"
    output_root: str = "outputs"
    seed: int = 17
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    activations: ActivationConfig = field(default_factory=ActivationConfig)
    rdm: RDMConfig = field(default_factory=RDMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / self.run_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    raw = json.loads(Path(path).read_text())
    return ExperimentConfig(
        run_id=raw.get("run_id", "qwen3_0.6b_goemotions"),
        output_root=raw.get("output_root", "outputs"),
        seed=raw.get("seed", 17),
        dataset=DatasetConfig(**raw.get("dataset", {})),
        model=ModelConfig(**raw.get("model", {})),
        activations=ActivationConfig(**raw.get("activations", {})),
        rdm=RDMConfig(**raw.get("rdm", {})),
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        analysis=AnalysisConfig(**raw.get("analysis", {})),
    )
