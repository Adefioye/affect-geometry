"""Frozen readouts over raw post-block states, with explicit artifact indexing.

Tuned: h + Wh + b, then the artifact's final norm and output head (no softcap).
Jacobian: h J.T, then native final norm/head/softcap, matching jlens.HFLensModel.
No additional output-head copies or translator fitting are needed.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch.nn import functional as F

from affect_geometry.m03.configuration import Config
from affect_geometry.m03.rules import layer_plan


def unembedding_hash(weight: torch.Tensor) -> str:
    """Match tuned_lens.utils.tensor_hash without copying a multi-GB head.

    NumPy's default array_str shows only the edge items for large matrices.
    A summarized 7x7 matrix has the same displayed edges and ellipses.
    """
    if weight.ndim != 2:
        raise ValueError("Expected a matrix")
    if weight.numel() <= 1000 or min(weight.shape) < 7:
        display = np.array_str(weight.detach().cpu().float().numpy(), precision=1)
    else:
        rows = [0, 1, 2, weight.shape[0] - 3, weight.shape[0] - 2, weight.shape[0] - 1]
        cols = [0, 1, 2, weight.shape[1] - 3, weight.shape[1] - 2, weight.shape[1] - 1]
        edge = weight.detach()[rows][:, cols].cpu().float().numpy()
        summary = np.zeros((7, 7), dtype=np.float32)
        indices = [0, 1, 2, 4, 5, 6]
        summary[np.ix_(indices, indices)] = edge
        display = np.array2string(summary, precision=1, threshold=36)
    return hashlib.sha256(display.encode()).hexdigest()


class Readouts:
    def __init__(
        self, model, config: Config, tuned_config: dict, tuned_state: dict, jacobian_state: dict
    ):
        self.model, self.config = model, config
        self.blocks = model.model.layers
        self.norm, self.head = model.model.norm, model.lm_head
        self.n_blocks = len(self.blocks)
        d = model.config.hidden_size
        if self.n_blocks != config.n_blocks:
            raise ValueError("Model block count differs from the frozen configuration")
        for key, value in dict(
            base_model_name_or_path=config.model_id, d_model=d, num_hidden_layers=self.n_blocks
        ).items():
            if tuned_config.get(key) != value:
                raise ValueError(f"Tuned artifact mismatch: {key}")
        if tuned_config.get("base_model_revision") not in {None, config.model_revision}:
            raise ValueError("Tuned artifact base-model revision mismatch")
        actual_hash = unembedding_hash(self.head.weight)
        if tuned_config.get("unembed_hash") and actual_hash != tuned_config["unembed_hash"]:
            raise ValueError("Tuned artifact unembedding hash mismatch")
        artifact_device = self.head.weight.device if config.lens_device == "model" else "cpu"
        self.tuned = {}
        embedding_indices = []
        expected_keys = set()
        saved_indices = sorted({int(k.split(".")[0]) for k in tuned_state})
        if saved_indices != list(range(self.n_blocks)):
            raise ValueError("Unexpected tuned translator coverage; declare a supported artifact")
        for index in saved_indices:
            weight = tuned_state[f"{index}.weight"]
            bias = tuned_state.get(f"{index}.bias")
            expected_keys.add(f"{index}.weight")
            if tuned_config.get("bias", True):
                expected_keys.add(f"{index}.bias")
                if bias is None or bias.shape != (d,):
                    raise ValueError("Missing or malformed tuned bias")
            if weight.shape != (d, d) or not torch.isfinite(weight).all():
                raise ValueError("Malformed tuned translator")
            if bias is not None and not torch.isfinite(bias).all():
                raise ValueError("Nonfinite tuned bias")
            layer = index - 1 if config.tuned_indexing == "hf_hidden_states" else index
            if layer < 0:
                embedding_indices.append(index)
                continue
            self.tuned[layer] = (
                weight.to(device=artifact_device, dtype=self.head.weight.dtype),
                None
                if bias is None
                else bias.to(device=artifact_device, dtype=self.head.weight.dtype),
                index,
            )
        if set(tuned_state) != expected_keys:
            raise ValueError("Unexpected tensors in tuned checkpoint")
        if jacobian_state.get("d_model") != d or "J" not in jacobian_state:
            raise ValueError("Jacobian artifact width/format mismatch")
        self.jacobian = {}
        for key, value in jacobian_state["J"].items():
            layer = int(key)
            if not 0 <= layer < config.jacobian_target_layer:
                raise ValueError("Jacobian source layer is not below its declared target")
            if value.shape != (d, d) or not torch.isfinite(value).all():
                raise ValueError("Malformed Jacobian matrix")
            self.jacobian[layer] = value.float().to(artifact_device)
        if sorted(self.jacobian) != sorted(jacobian_state.get("source_layers", [])):
            raise ValueError("Jacobian source-layer metadata disagrees with matrices")
        if jacobian_state.get("target_layer", config.jacobian_target_layer) != (
            config.jacobian_target_layer
        ):
            raise ValueError("Jacobian target-layer mismatch")
        self.plan = layer_plan(
            dict(
                logit=list(range(self.n_blocks)),
                tuned=sorted(self.tuned),
                jacobian=sorted(self.jacobian),
            ),
            self.n_blocks,
        )
        self.metadata = dict(
            plan=self.plan,
            attention_implementation=getattr(model.config, "_attn_implementation", None),
            unembedding_hash=actual_hash,
            tuned_config=tuned_config,
            tuned_mapping={str(layer): t[2] for layer, t in self.tuned.items()},
            embedding_translators=embedding_indices,
            jacobian_target_layer=config.jacobian_target_layer,
            jacobian_n_prompts=jacobian_state.get("n_prompts"),
            conventions=dict(
                logit="raw post-block -> native norm/head/softcap",
                tuned="h + linear(h) -> native norm/head; artifact softcap="
                + str(config.tuned_softcap),
                jacobian="float32 h @ J.T -> native norm/head/softcap",
            ),
            missing_layer_reasons={
                "tuned": "Only the released translators are used; embedding is separate. "
                "Standard HF indexing omits a fitted final-block translator.",
                "jacobian": "Only released source keys below the fit target are supported.",
            },
        )

    @torch.inference_mode()
    def project(self, residual: torch.Tensor, lens: str, layer: int) -> torch.Tensor:
        h = residual
        if lens == "tuned":
            w, b, _ = self.tuned[layer]
            h = h.to(device=w.device, dtype=w.dtype)
            h = h + F.linear(h, w, b)
        elif lens == "jacobian":
            j = self.jacobian[layer]
            h = h.to(device=j.device, dtype=torch.float32) @ j.T
        elif lens not in {"native", "logit"}:
            raise ValueError(f"Unknown lens: {lens}")
        norm_weight = self.norm.weight
        h = self.norm(h.to(device=norm_weight.device, dtype=norm_weight.dtype))
        logits = self.head(h.to(device=self.head.weight.device, dtype=self.head.weight.dtype))
        cap = getattr(self.model.config, "final_logit_softcapping", None)
        if cap is not None and (lens != "tuned" or self.config.tuned_softcap):
            logits = cap * torch.tanh(logits / cap)
        return logits.float()
