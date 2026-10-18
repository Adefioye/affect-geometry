from __future__ import annotations

from typing import Any

import torch

from affect_geometry.config import ModelConfig


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested != "auto":
        return getattr(torch, requested)
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def load_causal_lm(config: ModelConfig) -> tuple[Any, Any, torch.device]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(config.name, revision=config.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.name,
        revision=config.revision,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def decoder_layers(model: Any) -> list[torch.nn.Module]:
    """Resolve common HF decoder paths, including Qwen3's ``model.layers``."""
    candidates = (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    )
    for parent_name, layers_name in candidates:
        parent = getattr(model, parent_name, None)
        layers = getattr(parent, layers_name, None) if parent is not None else None
        if layers is not None:
            return list(layers)
    raise ValueError("Could not locate decoder layers on this Hugging Face model")
