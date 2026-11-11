"""Extract pre-generation EXP2 prompt states; never substitute raw-comment states."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from affect_geometry.classification.protocol import digest, prompt
from affect_geometry.logging import progress
from affect_geometry.persistence.correctness_geometry import GeometryStore, checksum


def validate_prompts(samples, manifest, tokenizer):
    rendered = []
    for row in samples.to_dict("records"):
        payload, _ = prompt(row, manifest["seed"])
        text = (
            tokenizer.apply_chat_template(
                [{"role": "user", "content": payload}], tokenize=False, add_generation_prompt=True
            )
            if manifest["chat"]
            else payload
        )
        if text != row["rendered_prompt"] or digest(text) != row["prompt_hash"]:
            raise ValueError("EXP2 prompt reconstruction mismatch")
        rendered.append(text)
    if manifest["chat"] and tokenizer.chat_template != manifest["chat_template"]:
        raise ValueError("Tokenizer template differs from EXP2")
    return rendered


def extract_prompt_states(
    samples,
    manifest,
    destination,
    *,
    batch_size=1,
    dtype="float32",
    device="auto",
    storage_dtype="float32",
):
    import torch

    from affect_geometry.classification.inference import Generator
    from affect_geometry.models.huggingface import decoder_layers

    generator = Generator(
        manifest["model_id"],
        manifest["model_revision"],
        manifest["chat"],
        dtype=dtype,
        device=device,
    )
    model, tokenizer = generator.model, generator.tokenizer
    samples = samples.sort_values("sample_id").reset_index(drop=True)
    rendered = validate_prompts(samples, manifest, tokenizer)
    layers = decoder_layers(model)
    metadata = dict(
        input_condition="classification_prompt_before_generation",
        model_id=manifest["model_id"],
        model_revision=manifest["model_revision"],
        tokenizer_revision=manifest["model_revision"],
        classification_run=manifest["run_id"],
        classification_manifest=manifest,
        prompt_template_hash=digest(manifest.get("chat_template") or ""),
        sample_ids=samples.sample_id.tolist(),
        prompt_hashes=samples.prompt_hash.tolist(),
        n_layers=len(layers),
        hidden_size=int(model.config.hidden_size),
        pooling="last_nonpadding_token",
        truncation="prohibited",
        dtype=dtype,
        storage_dtype=storage_dtype,
        batch_size=batch_size,
        device=device,
        layer_indexing="zero_based_post_block",
        torch_version=torch.__version__,
    )
    store = GeometryStore(destination, metadata)
    if store.valid("extraction", metadata):
        return
    paths = []
    arrays = []
    for layer in range(len(layers)):
        path = store.root / f"layer_{layer:03d}.npy"
        arrays.append(
            np.lib.format.open_memmap(
                path.with_suffix(".tmp"),
                mode="w+",
                dtype=storage_dtype,
                shape=(len(samples), model.config.hidden_size),
            )
        )
        paths.append(path)
    captured = [None] * len(layers)
    handles = []
    for index, layer in enumerate(layers):

        def hook(_module, _args, output, i=index):
            captured[i] = output[0] if isinstance(output, tuple) else output

        handles.append(layer.register_forward_hook(hook))
    positions = []
    try:
        for start in progress(
            range(0, len(samples), batch_size),
            description="Extracting classification prompts",
            unit="batch",
        ):
            text = rendered[start : start + batch_size]
            encoded = tokenizer(
                text,
                padding=True,
                add_special_tokens=not manifest["chat"],
                truncation=False,
                return_tensors="pt",
            )
            mask = encoded["attention_mask"]
            length = mask.sum(1).numpy()
            expected = samples.iloc[start : start + len(text)].input_token_count.to_numpy()
            if not np.array_equal(length, expected):
                raise ValueError("EXP2 token lengths do not match reconstructed prompts")
            width = mask.shape[1]
            if width + manifest["max_new_tokens"] > getattr(
                model.config, "max_position_embeddings", float("inf")
            ):
                raise ValueError("Context overflow; truncation is prohibited")
            last = (mask * torch.arange(width)).max(dim=1).values
            positions.extend(
                dict(
                    sample_id=samples.iloc[start + i].sample_id,
                    token_position=int(length[i] - 1),
                    padded_position=int(last[i]),
                    input_token_count=int(length[i]),
                )
                for i in range(len(text))
            )
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            backbone = model.get_decoder() if hasattr(model, "get_decoder") else model
            with torch.inference_mode():
                backbone(**encoded, use_cache=False, return_dict=True)
            for layer, hidden in enumerate(captured):
                if hidden is None:
                    raise RuntimeError("Post-block hook did not fire")
                state = hidden[
                    torch.arange(len(text), device=hidden.device), last.to(hidden.device)
                ]
                arrays[layer][start : start + len(text)] = state.float().cpu().numpy()
                captured[layer] = None
        import os

        for array, path in zip(arrays, paths, strict=True):
            array.flush()
            os.replace(path.with_suffix(".tmp"), path)
        paths.append(store.csv("samples.csv", samples))
        paths.append(store.csv("token_positions.csv", pd.DataFrame(positions)))
        store.complete("extraction", metadata, paths)
    finally:
        for handle in handles:
            handle.remove()


def open_prompt_states(root, samples, manifest):
    root = Path(root)
    metadata = json.loads((root / "manifest.json").read_text())
    if metadata.get("input_condition") != "classification_prompt_before_generation":
        raise ValueError("Raw-comment activations cannot be used for EXP3")
    for key in ("model_id", "model_revision"):
        if metadata.get(key) != manifest[key]:
            raise ValueError(f"Activation provenance mismatch: {key}")
    if metadata.get("classification_manifest") != manifest:
        raise ValueError("Activation classification source differs")
    sample_table = pd.read_csv(root / "samples.csv", dtype={"sample_id": str})
    if sample_table.sample_id.duplicated().any():
        raise ValueError("Duplicate activation IDs")
    ids = sample_table.sample_id.tolist()
    if metadata["sample_ids"] != ids:
        raise ValueError("Activation row mapping mismatch")
    if not set(samples.sample_id) <= set(ids):
        raise ValueError("Missing cohort activation IDs")
    subset = sample_table.set_index("sample_id").loc[samples.sample_id]
    if subset.prompt_hash.tolist() != samples.prompt_hash.tolist():
        raise ValueError("Activation prompt hashes differ from selected source")
    stage = json.loads((root / "checkpoints/extraction.json").read_text())
    from affect_geometry.persistence.correctness_geometry import fingerprint

    payload = {k: v for k, v in metadata.items() if k != "schema_version"}
    if stage["provenance"] != fingerprint(payload):
        raise ValueError("Activation stage provenance mismatch")
    if not all(checksum(root / p) == sha for p, sha in stage["artifacts"].items()):
        raise ValueError("Activation checksum mismatch")
    paths = [root / f"layer_{i:03d}.npy" for i in range(metadata["n_layers"])]
    for path in paths:
        array = np.load(path, mmap_mode="r")
        if array.shape != (len(ids), metadata["hidden_size"]):
            raise ValueError("Activation shape mismatch")
    return sample_table, paths, metadata, stage["artifacts"]
