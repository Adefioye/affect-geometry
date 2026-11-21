"""Strict, serializable experiment settings; no annotation-based selection options."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Config:
    model_id: str = "google/gemma-2-9b"
    model_revision: str = "33c193028431c2fde6c6e51f29e6f17b60cbfac6"
    dataset_id: str = "google-research-datasets/go_emotions"
    dataset_revision: str = "add492243ff905527e67aeb8b80c082af02207c3"
    baseline_root: str = (
        "outputs/exp2_classification/gemma2/full/hf_snapshot/"
        "cbe0b8e3dd8df548fac272246f70a291a2c74e1a"
    )
    baseline_repo: str = "kokolamba/rq2-goemotions-single-label-gemma2"
    baseline_revision: str = "cbe0b8e3dd8df548fac272246f70a291a2c74e1a"
    tuned_repo: str = "chutommy/tuned-lenses"
    tuned_revision: str = "5667bafe74f3e55e9f1be1f9b1a2f699a8383fdd"
    tuned_folder: str = "tuned-lens--gemma-2-9b--fineweb-edu"
    tuned_corpus: str = "HuggingFaceFW/fineweb-edu"
    tuned_indexing: str = "hf_hidden_states"
    tuned_softcap: bool = False
    jacobian_repo: str = "neuronpedia/jacobian-lens"
    jacobian_revision: str = "d2527c3cdbaa60dd4b65cbca1604921f044e989a"
    jacobian_folder: str = "gemma-2-9b/jlens/Salesforce-wikitext"
    jacobian_filename: str = "gemma-2-9b_jacobian_lens.pt"
    jacobian_corpus: str = "Salesforce/wikitext:wikitext-103-raw-v1:train"
    jacobian_target_layer: int = 41
    n_blocks: int = 42
    dtype: str = "bfloat16"
    device: str = "cuda"
    lens_device: str = "model"
    prefix_batch_size: int = 16
    projection_batch_size: int = 16
    seed: int = 42
    label_prefix: str = " "
    stop_token_id: int = 1
    max_new_tokens: int = 10
    pilot_size: int = 64
    diagnostic_size: int = 8
    retain_diagnostic_distributions: bool = True
    reference_size: int = 2048
    calibrated: bool = False
    length_normalized: bool = False
    bootstrap_replicates: int = 2000
    native_atol: float = 0.25
    native_rtol: float = 0.02
    score_atol: float = 0.25
    alternate_menu: bool = True
    alternate_prompt: bool = False
    test_history: str = "exploratory; prior EXP2 outcomes were available"
    output_root: str = "outputs/m03_lens_comparison"

    def __post_init__(self) -> None:
        for name in (
            "calibrated",
            "length_normalized",
            "tuned_softcap",
            "alternate_menu",
            "alternate_prompt",
            "retain_diagnostic_distributions",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a JSON boolean")
        for name in (
            "model_revision",
            "dataset_revision",
            "tuned_revision",
            "jacobian_revision",
            "baseline_revision",
        ):
            if not re.fullmatch(r"[0-9a-f]{40}", getattr(self, name)):
                raise ValueError(f"{name} must be an immutable 40-character commit")
        for name in (
            "n_blocks",
            "prefix_batch_size",
            "projection_batch_size",
            "pilot_size",
            "reference_size",
            "bootstrap_replicates",
            "max_new_tokens",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.diagnostic_size < 0 or not 0 < self.jacobian_target_layer < self.n_blocks:
            raise ValueError("Invalid diagnostic size or Jacobian target layer")
        if self.tuned_indexing not in {"hf_hidden_states", "post_block"}:
            raise ValueError("Declare hf_hidden_states or post_block tuned indexing")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported dtype")
        if self.label_prefix not in {"", " "} or self.stop_token_id < 0:
            raise ValueError("Invalid canonical continuation settings")
        if any(getattr(self, k) < 0 for k in ("native_atol", "native_rtol", "score_atol")):
            raise ValueError("Numerical tolerances must be nonnegative")
        if not self.test_history.strip():
            raise ValueError("Record the actual history of test inspection")
        output = Path(self.output_root)
        if (
            not output.parts
            or output.is_absolute()
            or output.parts[0] != "outputs"
            or ".." in output.parts
        ):
            raise ValueError("Generated artifacts must live under outputs/")
        if self.reference_size > 2048:
            raise ValueError("The declared unlabeled reference is limited to 2048 comments")
        if self.lens_device not in {"model", "cpu"}:
            raise ValueError("lens_device must be model or cpu")

    @classmethod
    def load(cls, path: str | Path) -> Config:
        value = json.loads(Path(path).read_text())
        extra = set(value) - {f.name for f in fields(cls)}
        if extra:
            raise ValueError(
                f"Unknown settings (validation tuning is unsupported): {sorted(extra)}"
            )
        return cls(**value)

    def to_dict(self) -> dict:
        return asdict(self)

    def root(self, run_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
            raise ValueError("Use a simple, nonempty run ID without path separators")
        return Path(self.output_root) / self.model_id.replace("/", "--") / run_id
