"""Offline end-to-end validation with random tiny Gemma weights, never research data."""

from __future__ import annotations

import re
from dataclasses import replace

import torch

from affect_geometry.classification.protocol import EMOTIONS, digest, prompt, score
from affect_geometry.classification.storage import atomic_json
from affect_geometry.m03.configuration import Config
from affect_geometry.m03.readouts import Readouts, unembedding_hash
from affect_geometry.m03.storage import code_hash, file_hash, frozen_json, write_parquet


class ToyTokenizer:
    """Small reversible tokenizer; every candidate label spans multiple tokens."""

    bos_token_id, eos_token_id, pad_token_id = 0, 1, 2
    all_special_ids = [0, 1, 2]

    def __init__(self):
        self.tokens = ["<bos>", "<eos>", "<pad>"]
        self.lookup = {p: i for i, p in enumerate(self.tokens)}

    def encode(self, text, add_special_tokens=True):
        pieces = re.findall(r"\s+|[A-Za-z]+|[^A-Za-z\s]", text)
        split = []
        for piece in pieces:
            split.extend(
                [piece[:3], piece[3:]] if piece in EMOTIONS and len(piece) > 3 else [piece]
            )
        result = []
        for piece in split:
            if piece not in self.lookup:
                self.lookup[piece] = len(self.tokens)
                self.tokens.append(piece)
            result.append(self.lookup[piece])
        if len(self.tokens) >= 512:
            raise ValueError("Toy tokenizer vocabulary exhausted")
        return ([0] if add_special_tokens else []) + result

    def decode(self, ids, skip_special_tokens=False):
        return "".join(
            self.tokens[int(i)]
            for i in ids
            if not skip_special_tokens or int(i) not in self.all_special_ids
        )


def tiny_bundle(config: Config):
    from transformers import Gemma2Config, Gemma2ForCausalLM

    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = (
            Gemma2ForCausalLM(
                Gemma2Config(
                    vocab_size=512,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=3,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=8,
                    max_position_embeddings=2048,
                    sliding_window=256,
                    final_logit_softcapping=3.0,
                    attn_logit_softcapping=5.0,
                    bos_token_id=0,
                    eos_token_id=1,
                    pad_token_id=2,
                    attention_dropout=0,
                )
            )
            .eval()
            .requires_grad_(False)
        )
        model.config._attn_implementation = "eager"
        state = {}
        for i in range(3):
            state[f"{i}.weight"] = torch.randn(16, 16) * 0.01
            state[f"{i}.bias"] = torch.randn(16) * 0.01
        jstate = dict(
            J={i: torch.eye(16) + torch.randn(16, 16) * 0.01 for i in range(2)},
            n_prompts=2,
            d_model=16,
            source_layers=[0, 1],
        )
    tuned_config = dict(
        base_model_name_or_path=config.model_id,
        d_model=16,
        num_hidden_layers=3,
        bias=True,
        unembed_hash=unembedding_hash(model.lm_head.weight),
    )
    readouts = Readouts(model, config, tuned_config, state, jstate)
    return model, ToyTokenizer(), readouts, dict(**readouts.metadata, synthetic=True), 0.0


def prepare_smoke(run_id: str):
    config = replace(
        Config(),
        model_id="synthetic/tiny-gemma2",
        n_blocks=3,
        jacobian_target_layer=2,
        dtype="float32",
        device="cpu",
        pilot_size=2,
        diagnostic_size=1,
        reference_size=3,
        calibrated=True,
        length_normalized=True,
        alternate_prompt=True,
        bootstrap_replicates=32,
        native_atol=1e-5,
        native_rtol=1e-5,
        score_atol=1e-5,
        test_history="synthetic smoke test; not empirical research",
    )
    root = config.root(run_id)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, emotion in enumerate(("joy", "sadness", "neutral", "gratitude", "joy", "neutral")):
        split = "train" if i < 3 else "test"
        row = dict(
            id=str(i),
            sample_id=digest(f"synthetic:{i}"),
            text=f"Example number {i}.",
            source_split=split,
            emotion=emotion,
            label_id=EMOTIONS.index(emotion),
            labels=[EMOTIONS.index(emotion)],
        )
        payload, order = prompt(row)
        row.update(
            rendered_prompt=payload,
            label_order=order,
            prompt_hash=digest(payload),
            text_hash=digest(row["text"]),
            status="completed",
            **score("neutral" if i % 2 else "invalid sentence", emotion),
        )
        rows.append(row)
    parts = dict(pilot=rows[:2], reference=rows[:3], test=rows[3:])
    for name, part in parts.items():
        write_parquet(root / "prepared" / f"{name}.parquet", part)
    import pandas as pd

    pd.DataFrame(
        [
            dict(
                sample_id=r["sample_id"],
                text_hash=r["text_hash"],
                source_split=r["source_split"],
                cross_split_duplicate=False,
                within_split_count=1,
            )
            for r in rows
        ]
    ).to_csv(
        root / "prepared" / "duplicate_audit.csv",
        index=False,
    )
    frozen_json(
        root / "protocol.json",
        dict(
            protocol="m03_label_free_v1",
            synthetic=True,
            config=config.to_dict(),
            code_hash=code_hash(),
            prepared_sha256={p.name: file_hash(p) for p in sorted((root / "prepared").iterdir())},
        ),
    )
    atomic_json(
        root / "SYNTHETIC.json",
        dict(
            notice=(
                "Random tiny model and invented comments. "
                "These are software checks, not M03 results."
            )
        ),
    )
    return config, root


def smoke(run_id: str):
    from affect_geometry.m03.analysis import analyze
    from affect_geometry.m03.pipeline import calibrate, extract, pilot
    from affect_geometry.m03.plotting import plot

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config, root = prepare_smoke(run_id)
        bundle = tiny_bundle(config)
        pilot(config, run_id, bundle=bundle)
        extract(config, run_id, phase="reference", bundle=bundle)
        calibrate(config, run_id)
        extract(config, run_id, bundle=bundle)
        analyze(config, run_id, export_complete_scores=True)
        plot(root)
        return root
    finally:
        torch.set_num_threads(previous_threads)
