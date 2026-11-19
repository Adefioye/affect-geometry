"""Pinned model/artifact loading, separated from extraction and analysis."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from affect_geometry.classification.environment import load_hf_token
from affect_geometry.m03.configuration import Config
from affect_geometry.m03.readouts import Readouts
from affect_geometry.m03.storage import file_hash


def load(config: Config, assets: Path):
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Use a CUDA host for Gemma-2-9B; "
            "the offline smoke test runs on CPU."
        )
    started = time.perf_counter()
    token = load_hf_token()
    assets.mkdir(parents=True, exist_ok=True)
    common = dict(revision=config.model_revision, token=token, cache_dir=assets)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, **common)
    if tokenizer.eos_token_id != config.stop_token_id:
        raise ValueError("Declared terminal event differs from the model's EOS token")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=getattr(torch, config.dtype),
        device_map="auto" if config.device == "auto" else None,
        attn_implementation="eager",
        **common,
    )
    if config.device != "auto":
        model.to(config.device)
    model.eval().requires_grad_(False)

    def fetch(repo, revision, filename):
        return Path(
            hf_hub_download(repo, filename, revision=revision, cache_dir=assets, token=token)
        )

    tuned_config = fetch(
        config.tuned_repo, config.tuned_revision, config.tuned_folder + "/config.json"
    )
    tuned_params = fetch(
        config.tuned_repo, config.tuned_revision, config.tuned_folder + "/params.pt"
    )
    jac_config = fetch(
        config.jacobian_repo, config.jacobian_revision, config.jacobian_folder + "/config.yaml"
    )
    jac_params = fetch(
        config.jacobian_repo,
        config.jacobian_revision,
        config.jacobian_folder + "/" + config.jacobian_filename,
    )
    readouts = Readouts(
        model,
        config,
        json.loads(tuned_config.read_text()),
        torch.load(tuned_params, map_location="cpu", weights_only=True),
        torch.load(jac_params, map_location="cpu", weights_only=True),
    )
    metadata = dict(
        **readouts.metadata,
        artifact_sha256={
            "tuned_config": file_hash(tuned_config),
            "tuned_params": file_hash(tuned_params),
            "jacobian_config": file_hash(jac_config),
            "jacobian_params": file_hash(jac_params),
        },
        jacobian_fit_config=jac_config.read_text(),
    )
    return model, tokenizer, readouts, metadata, time.perf_counter() - started
