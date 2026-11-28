"""Strict, serializable EXP5 settings; immutable revisions come from audited EXP2."""

import json
from dataclasses import asdict, dataclass, fields
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class NeuronTransferConfig:
    classification_dir: str
    run_id: str
    model_id: str = "allenai/OLMo-2-1124-7B-Instruct"
    output_root: str = "outputs/neuron-transfer"
    log_root: str = "logs"
    seed: int = 42
    discovery_per_emotion: int = 50
    selection_per_emotion: int = 10
    smoke_size: int = 32
    top_k: int = 5
    minimum_count: int = 5
    null_replicates: int = 20
    null_sweeps: int = 20
    edge_threshold: float = 8.0
    resolutions: tuple = (8.0, 32.0, 128.0)
    minimum_size: int = 2
    maximum_size: int = 128
    candidate_budget: int = 100
    community_budget: int = 20
    minimum_effect_examples: int = 5
    maximum_overlap: float = 0.80
    donor_attempts: int = 5
    duplicate_threshold: float = 0.90
    alpha: float = 0.25
    score_batch_size: int = 32
    label_prefix: str = " "
    stop_token_id: int | None = None
    dtype: str = "bfloat16"
    device: str = "auto"
    bootstrap_replicates: int = 2000
    allow_partial_source: bool = False
    test_previously_inspected: bool = False

    def __post_init__(self):
        if not self.run_id or Path(self.run_id).name != self.run_id or self.run_id in {".", ".."}:
            raise ValueError("Use an explicit, path-safe run_id")
        for key in (
            "seed",
            "discovery_per_emotion",
            "selection_per_emotion",
            "smoke_size",
            "top_k",
            "minimum_count",
            "null_replicates",
            "null_sweeps",
            "minimum_size",
            "maximum_size",
            "candidate_budget",
            "community_budget",
            "minimum_effect_examples",
            "donor_attempts",
            "score_batch_size",
            "bootstrap_replicates",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < (key != "seed"):
                raise ValueError(f"Invalid positive integer: {key}")
        if self.community_budget > 20 or self.candidate_budget > 100:
            raise ValueError("EXP5 budgets are at most 20 communities and 100 candidates")
        if self.discovery_per_emotion > 50 or self.selection_per_emotion > 10:
            raise ValueError("EXP5 caps are 50 discovery and 10 selection examples per label")
        if self.smoke_size > 32 or self.donor_attempts > 5:
            raise ValueError("EXP5 caps are 32 smoke comments and five donor attempts")
        if self.selection_per_emotion > self.discovery_per_emotion:
            raise ValueError("Selection panel must fit inside discovery")
        if self.null_replicates < 2 or self.bootstrap_replicates < 2:
            raise ValueError("At least two null/bootstrap replicates are required")
        if not 2 <= self.minimum_size <= self.maximum_size <= 128:
            raise ValueError("Invalid community size range")
        if self.alpha != 0.25:
            raise ValueError("The first EXP5 protocol fixes transfer strength to 0.25")
        if not 0 < self.maximum_overlap <= 1 or not 0 < self.duplicate_threshold <= 1:
            raise ValueError("Invalid overlap threshold")
        if (
            not isfinite(self.edge_threshold)
            or self.edge_threshold <= 0
            or not self.resolutions
            or any(not isfinite(v) or v <= 0 for v in self.resolutions)
        ):
            raise ValueError("Graph thresholds/resolutions must be positive")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported model dtype")

    @property
    def root(self):
        return Path(self.output_root) / self.model_id.replace("/", "--") / self.run_id

    def to_dict(self):
        return asdict(self)

    @classmethod
    def read(cls, path):
        payload = json.loads(Path(path).read_text())
        unknown = set(payload) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown EXP5 configuration keys: {sorted(unknown)}")
        if "resolutions" in payload:
            payload["resolutions"] = tuple(payload["resolutions"])
        return cls(**payload)
