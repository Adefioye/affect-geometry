"""Read and audit saved EXP2 records without rerunning classification."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from affect_geometry.classification.data import EXPECTED
from affect_geometry.classification.protocol import PROTOCOL, digest
from affect_geometry.persistence.correctness_geometry import checksum


def read_classification(root, model_id, allow_partial=False):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    for key in ("model_revision", "dataset_revision", "protocol", "sample_ids_hash", "chat"):
        if key not in manifest:
            raise ValueError(f"Classification manifest missing {key}")
    if manifest["model_id"] != model_id or manifest["protocol"] != PROTOCOL:
        raise ValueError("Classification model/protocol mismatch")
    for key in ("model_revision", "dataset_revision"):
        if manifest[key] in ("main", "default", None, ""):
            raise ValueError(f"Unpinned {key}")
    paths = [root / "dataset" / f"{split}.parquet" for split in EXPECTED]
    rows = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    required = {
        "sample_id",
        "source_split",
        "source_row",
        "emotion",
        "text",
        "text_hash",
        "status",
        "compat_correct",
        "compat_parser_version",
        "correct",
        "valid_label",
        "compat_valid_label",
        "predicted_emotion",
        "compat_predicted_emotion",
        "model_revision",
        "tokenizer_revision",
        "protocol_version",
        "run_id",
        "rendered_prompt",
        "prompt_hash",
        "input_token_count",
        "was_truncated",
    }
    if required - set(rows):
        raise ValueError(f"Classification records missing {required - set(rows)}")
    if rows.sample_id.isna().any() or rows.sample_id.duplicated().any():
        raise ValueError("Duplicate/missing source IDs")
    if not rows.text.map(digest).eq(rows.text_hash).all():
        raise ValueError("Source text hash mismatch")
    ordered = pd.concat([rows[rows.source_split.eq(s)].sort_values("source_row") for s in EXPECTED])
    if digest("\n".join(ordered.sample_id)) != manifest["sample_ids_hash"]:
        raise ValueError("Classification sample manifest hash mismatch")
    for column, value in [
        ("model_revision", manifest["model_revision"]),
        ("tokenizer_revision", manifest["model_revision"]),
        ("protocol_version", PROTOCOL),
        ("run_id", manifest["run_id"]),
    ]:
        if not rows[column].eq(value).all():
            raise ValueError(f"Classification row provenance mismatch: {column}")
    done = rows.loc[rows.status.eq("completed") & rows.compat_correct.notna()]
    if set(done.compat_parser_version) != {"choi_weber_first_word_v1"}:
        raise ValueError("Unrecognized compatibility parser")
    for row in done.to_dict("records"):
        if pd.isna(row["input_token_count"]) or row["input_token_count"] < 1:
            raise ValueError("Missing/invalid classification prompt token count")
        if not isinstance(row.get("rendered_prompt"), str) or digest(
            row["rendered_prompt"]
        ) != row.get("prompt_hash"):
            raise ValueError("Saved prompt hash mismatch")
        if row.get("was_truncated"):
            raise ValueError("EXP2 prompt truncation is prohibited")
    counts = rows.groupby("source_split").size().to_dict()
    full = counts == EXPECTED and not manifest.get("pilot_limit") and len(done) == len(rows)
    if not full and not allow_partial:
        raise ValueError("Partial EXP2 source; explicit pilot mode is required")
    rows["text_length"] = rows.text.str.len()
    rows["duplicate_text_group"] = rows.text_hash
    return (
        rows,
        manifest,
        dict(
            full_population=full,
            counts=counts,
            unresolved=len(rows) - len(done),
            checksums={str(p): checksum(p) for p in [root / "manifest.json", *paths]},
        ),
    )


def read_targets(path, provenance_path, labels):
    path, provenance_path = Path(path), Path(provenance_path)
    provenance = json.loads(provenance_path.read_text())
    if checksum(path) != provenance.get("reference_sha256"):
        raise ValueError("NRC reference checksum mismatch")
    if not provenance.get("version") or not provenance.get("matching"):
        raise ValueError("NRC version and mapping provenance are required")
    frame = pd.read_csv(path)
    if frame.emotion.duplicated().any():
        raise ValueError("Duplicate NRC emotion mapping")
    if "neutral" not in labels or set(labels) - set(frame.emotion):
        raise ValueError("NRC coverage including neutral is required")
    target = frame.set_index("emotion").loc[list(labels), ["valence", "arousal"]]
    if not np.isfinite(target.to_numpy()).all():
        raise ValueError("Missing NRC target values")
    return target.to_numpy(), dict(
        **provenance, mapping=frame.to_dict("records"), file_sha256=checksum(path)
    )
