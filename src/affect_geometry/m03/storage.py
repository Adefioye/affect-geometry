"""Atomic, context-checked score shards and frozen manifests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from affect_geometry.classification.storage import atomic_json
from affect_geometry.m03.configuration import fingerprint


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hash() -> str:
    root = Path(__file__).parent
    paths = sorted(root.glob("*.py")) + [
        root.parent / "classification" / name for name in ("protocol.py", "data.py")
    ]
    return fingerprint({str(p.relative_to(root.parent)): file_hash(p) for p in paths})


def frozen_json(path: Path, value: dict) -> None:
    value = json.loads(json.dumps(value, allow_nan=False))
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Frozen configuration differs: {path}; use a new run ID")
    atomic_json(path, value)


def write_parquet(path: Path, rows) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    (rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)).to_parquet(tmp, index=False)
    os.replace(tmp, path)


class ScoreStore:
    def __init__(self, root: Path, context: dict, keys: list[tuple[str, int]]):
        self.root, self.context, self.keys = Path(root), context, keys
        self.root.mkdir(parents=True, exist_ok=True)
        frozen_json(self.root / "context.json", dict(context=context, keys=keys))

    def path(self, sample_id: str) -> Path:
        if not sample_id.isalnum():
            raise ValueError("Unsafe sample ID")
        return self.root / f"{sample_id}.npz"

    def _validate(self, values: dict) -> None:
        expected = (len(self.keys), 28)
        for key in ("label_scores", "stop_scores", "token_counts"):
            if values[key].shape != expected or not np.isfinite(values[key]).all():
                raise ValueError(f"Incomplete/nonfinite 28-label shard: {key}")
        if (values["token_counts"] < 1).any():
            raise ValueError("Missing label tokens")

    def get(self, sample_id: str, identity: dict) -> dict | None:
        path = self.path(sample_id)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata != dict(context=self.context, identity=identity):
                raise ValueError(f"Cache identity differs for {sample_id}")
            values = {key: data[key].copy() for key in data.files if key != "metadata"}
        self._validate(values)
        return values

    def put(self, sample_id: str, identity: dict, values: dict) -> None:
        self._validate(values)
        target = self.path(sample_id)
        tmp = target.with_suffix(".tmp")
        with tmp.open("wb") as stream:
            np.savez_compressed(
                stream,
                metadata=json.dumps(
                    dict(context=self.context, identity=identity), sort_keys=True, allow_nan=False
                ),
                **values,
            )
        os.replace(tmp, target)


def score_keys(plan: dict) -> list[tuple[str, int]]:
    return [("native", plan["n_blocks"] - 1)] + [
        (lens, layer) for lens, layers in plan["coverage"].items() for layer in layers
    ]
