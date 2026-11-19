"""EXP2 source/baseline verification and label-independent sample selection."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from affect_geometry.classification.data import EXPECTED
from affect_geometry.classification.environment import load_hf_token
from affect_geometry.classification.protocol import EMOTIONS, digest, parse_strict, prompt
from affect_geometry.logging import progress
from affect_geometry.m03.configuration import Config, fingerprint
from affect_geometry.m03.storage import file_hash


def baseline_root(config: Config, cache_dir: Path) -> Path:
    root = Path(config.baseline_root)
    if (root / "manifest.json").is_file():
        return root
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            config.baseline_repo,
            repo_type="dataset",
            revision=config.baseline_revision,
            allow_patterns=["manifest.json", "data/*.parquet", "dataset/*.parquet"],
            cache_dir=cache_dir,
            token=load_hf_token(),
        )
    )


def validate_baseline(
    rows: list[dict], manifest: dict, config: Config, *, check_counts: bool = True
) -> None:
    expected = dict(
        model_id=config.model_id,
        model_revision=config.model_revision,
        dataset_id=config.dataset_id,
        dataset_revision=config.dataset_revision,
        seed=config.seed,
        dtype=config.dtype,
        chat=False,
        max_new_tokens=config.max_new_tokens,
        protocol="rq2_single_label_v1",
    )
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Baseline manifest mismatch: {key}")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate baseline sample IDs")
    for row in progress(rows, description="Verifying pinned EXP2 records", unit="comment"):
        if row["source_split"] not in EXPECTED or len(row["labels"]) != 1:
            raise ValueError("Baseline population must preserve original single-label splits")
        if EMOTIONS[row["label_id"]] != row["emotion"] or row["labels"][0] != row["label_id"]:
            raise ValueError("Baseline label vocabulary mismatch")
        identity = f"{config.dataset_id}@{config.dataset_revision}"
        if row["sample_id"] != digest(f"{identity}:{row['source_split']}:{row['id']}"):
            raise ValueError("Source sample ID mismatch")
        if digest(row["text"]) != row["text_hash"]:
            raise ValueError("Source text hash mismatch")
        payload, order = prompt(row, config.seed)
        if (
            row["rendered_prompt"] != payload
            or row["prompt_hash"] != digest(payload)
            or list(row["label_order"]) != order
        ):
            raise ValueError("Baseline prompt/order mismatch")
        prediction = parse_strict(row["raw_response"])
        if (
            row["status"] != "completed"
            or row["parser_version"] != "strict_label_v1"
            or row["predicted_emotion"] != prediction
            or bool(row["correct"]) != (prediction == row["emotion"])
            or bool(row["valid_label"]) != (prediction is not None)
        ):
            raise ValueError("Unresolved or inconsistently parsed baseline record")
        if row.get("was_truncated"):
            raise ValueError("Truncated baseline prompts are unsupported")
        for key in ("model_id", "model_revision"):
            if key in row and row[key] != getattr(config, key):
                raise ValueError(f"Baseline row provenance mismatch: {key}")
    if check_counts:
        actual = pd.Series([r["source_split"] for r in rows]).value_counts().to_dict()
        if actual != EXPECTED:
            raise ValueError(f"Unexpected single-label counts: {actual}, expected {EXPECTED}")


def load_baseline(config: Config, cache_dir: Path) -> tuple[list[dict], dict]:
    root = baseline_root(config, cache_dir)
    files = sorted((root / "data").glob("*.parquet"))
    if not files:
        files = sorted((root / "dataset").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No baseline Parquet files under {root}")
    frame = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    rows = frame.to_dict("records")
    manifest = json.loads((root / "manifest.json").read_text())
    validate_baseline(rows, manifest, config)
    return rows, dict(
        manifest=manifest,
        files_sha256={p.name: file_hash(p) for p in files},
        baseline_repo=config.baseline_repo,
        baseline_revision=config.baseline_revision,
    )


def select_unique(rows: list[dict], size: int, seed: int) -> list[dict]:
    by_text = {}
    for row in sorted(rows, key=lambda r: r["sample_id"]):
        by_text.setdefault(row["text_hash"], row)
    return sorted(
        by_text.values(), key=lambda r: (digest(f"{seed}:{r['text_hash']}"), r["sample_id"])
    )[:size]


def selections(rows: list[dict], config: Config) -> tuple[dict[str, list[dict]], pd.DataFrame]:
    train = [r for r in rows if r["source_split"] == "train"]
    test = sorted((r for r in rows if r["source_split"] == "test"), key=lambda r: r["sample_id"])
    groups = pd.DataFrame(
        [
            dict(sample_id=r["sample_id"], text_hash=r["text_hash"], source_split=r["source_split"])
            for r in rows
        ]
    )
    split_counts = groups.groupby("text_hash")["source_split"].nunique()
    groups["cross_split_duplicate"] = groups.text_hash.map(split_counts).gt(1)
    counts = groups.groupby(["source_split", "text_hash"]).size()
    groups["within_split_count"] = [
        counts[(r.source_split, r.text_hash)] for r in groups.itertuples()
    ]
    return dict(
        pilot=select_unique(train, config.pilot_size, config.seed),
        reference=select_unique(train, config.reference_size, config.seed),
        test=test,
    ), groups


def population_hash(rows: list[dict]) -> str:
    return fingerprint(
        [(r["sample_id"], r["text_hash"], r["source_split"], r["prompt_hash"]) for r in rows]
    )


def render(row: dict, variant: str, config: Config) -> str:
    if variant == "canonical":
        return row["rendered_prompt"]
    if variant == "alternate_menu":
        order = list(reversed(row["label_order"]))
        return (
            "Classify this text into exactly one emotion from this list:\n"
            + ", ".join(order)
            + "\nOutput only the emotion word.\n\n"
            + f"Text: {row['text']}\nEmotion:"
        )
    if variant == "alternate_prompt":
        return (
            "Choose exactly one emotion for the text below.\nAllowed emotions: "
            + ", ".join(row["label_order"])
            + "\nRespond with the emotion word only.\n\n"
            + f"Text: {row['text']}\nEmotion:"
        )
    raise ValueError(f"Unknown prompt variant: {variant}")
