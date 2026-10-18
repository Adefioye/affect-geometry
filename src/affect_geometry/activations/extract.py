from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from affect_geometry.activations.store import ActivationStore
from affect_geometry.config import ActivationConfig, ModelConfig
from affect_geometry.logging import get_logger, progress
from affect_geometry.models.huggingface import decoder_layers, load_causal_lm


def _batches(frame: pd.DataFrame, size: int) -> Iterator[pd.DataFrame]:
    for start in range(0, len(frame), size):
        yield frame.iloc[start : start + size]


def _hidden_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, (tuple, list)) else output


def pool_hidden(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    content = attention_mask.bool() & ~special_tokens_mask.bool()
    empty = ~content.any(dim=1)
    if empty.any():
        content[empty] = attention_mask[empty].bool()
    if pooling == "mean":
        weights = content.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
    if pooling == "last_token":
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        indices = positions.masked_fill(~content, -1).max(dim=1).values
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]
    raise ValueError(f"Unknown pooling: {pooling}")


def extract_activations(
    samples: pd.DataFrame,
    destination: Path,
    model_config: ModelConfig,
    activation_config: ActivationConfig,
) -> dict[str, int | str]:
    """Capture the post-block residual stream from every decoder layer."""
    logger = get_logger()
    logger.info("activations.model_loading model=%s", model_config.name)
    model, tokenizer, device = load_causal_lm(model_config)
    layers = decoder_layers(model)
    hidden_size = int(model.config.hidden_size)
    store = ActivationStore(
        destination,
        len(samples),
        len(layers),
        hidden_size,
        activation_config.poolings,
        activation_config.storage_dtype,
    )
    logger.info(
        "activations.start samples=%d layers=%d hidden_size=%d device=%s poolings=%s",
        len(samples),
        len(layers),
        hidden_size,
        device,
        ",".join(activation_config.poolings),
    )
    captured: list[torch.Tensor | None] = [None] * len(layers)
    handles = []
    for index, layer in enumerate(layers):

        def hook(_module: Any, _inputs: Any, output: Any, i: int = index) -> None:
            captured[i] = _hidden_tensor(output)

        handles.append(layer.register_forward_hook(hook))
    try:
        offset = 0
        backbone = model.get_decoder() if hasattr(model, "get_decoder") else model
        with torch.inference_mode():
            batches = _batches(samples, model_config.batch_size)
            total_batches = (len(samples) + model_config.batch_size - 1) // model_config.batch_size
            for batch in progress(
                batches,
                description=f"Extracting {len(layers)} Qwen layers",
                total=total_batches,
                unit="batch",
            ):
                encoded = tokenizer(
                    batch["text"].astype(str).tolist(),
                    padding=True,
                    truncation=True,
                    max_length=model_config.max_length,
                    return_tensors="pt",
                    return_special_tokens_mask=True,
                )
                special = encoded.pop("special_tokens_mask").to(device)
                encoded = {key: value.to(device) for key, value in encoded.items()}
                # Run the decoder directly: the LM-head logits are unnecessary and expensive.
                backbone(**encoded, use_cache=False, return_dict=True)
                for layer_index, hidden in enumerate(captured):
                    if hidden is None:
                        raise RuntimeError(f"Layer hook {layer_index} did not fire")
                    for pooling in activation_config.poolings:
                        pooled = pool_hidden(hidden, encoded["attention_mask"], special, pooling)
                        values = pooled.float().cpu().numpy()
                        store.write(pooling, layer_index, offset, values)
                    captured[layer_index] = None
                offset += len(batch)
        store.flush()
        logger.info("activations.complete samples=%d destination=%s", len(samples), destination)
    finally:
        for handle in handles:
            handle.remove()
    return {
        "n_samples": len(samples),
        "n_layers": len(layers),
        "hidden_size": hidden_size,
        "model": model_config.name,
        "model_revision": model_config.revision or "default",
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "device": str(device),
    }
