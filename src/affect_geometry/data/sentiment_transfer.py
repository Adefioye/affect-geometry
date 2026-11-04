"""Load frozen sentiment-manifold candidates and pair for the current tokenizer."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from affect_geometry.classification.environment import load_hf_token

PROMPT = "Review Text: {text} Review Sentiment:"


def load_candidates(spec):
    """Local CSV/Parquet or immutable materialized HF config; no sibling imports."""
    if spec.get("path"):
        path = Path(spec["path"])
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    else:
        from datasets import load_dataset
        from huggingface_hub import snapshot_download

        if not spec.get("revision"):
            raise ValueError("Remote datasets require an immutable revision")
        snapshot = Path(
            snapshot_download(
                spec["repo_id"],
                repo_type="dataset",
                revision=spec["revision"],
                allow_patterns=f"{spec['config']}/{spec['split']}-*.parquet",
                token=load_hf_token(),
            )
        )
        if snapshot.name != spec["revision"]:
            raise ValueError("Dataset revision must be the resolved commit SHA")
        shards = sorted((snapshot / spec["config"]).glob(f"{spec['split']}-*.parquet"))
        if not shards:
            raise FileNotFoundError("No candidate shards; check config and revision")
        frame = load_dataset(
            "parquet", data_files=[str(p) for p in shards], split="train"
        ).to_pandas()
    required = {"example_id", "text", "label"}
    if not required <= set(frame.columns):
        raise ValueError(f"Missing dataset fields: {required - set(frame.columns)}")
    if "split" in frame:
        frame = frame.loc[frame["split"].eq(spec.get("split", "test"))].copy()
    if frame.empty or frame.example_id.isna().any() or frame.example_id.duplicated().any():
        raise ValueError("Empty candidates or invalid example IDs")
    if not frame.label.isin([0, 1]).all() or frame.text.isna().any():
        raise ValueError("Expected non-null texts with binary gold labels")
    frame["example_id"] = frame.example_id.astype(str)
    frame["prompt"] = frame.text.map(lambda t: PROMPT.format(text=t))
    return frame


def make_pairs(frame, tokenizer, *, seed=42, max_tokens=128, max_pairs=None):
    """Equal full-prompt token lengths, no replacement, both intervention directions."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(lambda: {0: [], 1: []})
    audit = []
    for row in frame.sort_values("example_id").to_dict("records"):
        n = len(tokenizer.encode(row["prompt"], add_special_tokens=True))
        keep = 0 < n <= max_tokens
        audit.append({**row, "token_count": n, "within_budget": keep})
        if keep:
            groups[n][int(row["label"])].append(row)
    matches = []
    for n in sorted(groups):
        negative, positive = groups[n][0], groups[n][1]
        rng.shuffle(negative)
        rng.shuffle(positive)
        matches.extend((n, neg, pos) for neg, pos in zip(negative, positive))
    rng.shuffle(matches)
    if max_pairs is not None:
        if max_pairs < 1:
            raise ValueError("max_pairs must be positive")
        matches = matches[:max_pairs]
    records = []
    for i, (n, negative, positive) in enumerate(matches):
        for recipient, donor in ((negative, positive), (positive, negative)):
            records.append(
                dict(
                    pair_id=i,
                    case_id=f"{i}-{recipient['label']}-to-{donor['label']}",
                    token_count=n,
                    recipient_id=recipient["example_id"],
                    donor_id=donor["example_id"],
                    recipient_prompt=recipient["prompt"],
                    donor_prompt=donor["prompt"],
                    recipient_label=int(recipient["label"]),
                    donor_label=int(donor["label"]),
                )
            )
    if not records:
        raise ValueError("No opposite-label equal-length pairs for this tokenizer")
    return pd.DataFrame(records), pd.DataFrame(audit)
