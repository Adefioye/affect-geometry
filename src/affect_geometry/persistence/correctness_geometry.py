"""Versioned provenance, atomic artifacts, and checksum-validated stage completion."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True).encode()).hexdigest()


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class GeometryStore:
    def __init__(self, root, manifest):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = jsonable(dict(schema_version=SCHEMA_VERSION, **manifest))
        path = self.root / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != self.manifest:
            raise ValueError("Run provenance changed; use a new run ID")
        self.json("manifest.json", self.manifest)

    def _write(self, relative, writer):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        writer(tmp)
        os.replace(tmp, path)
        return path

    def json(self, relative, value):
        return self._write(
            relative,
            lambda p: p.write_text(
                json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
            ),
        )

    def csv(self, relative, frame):
        return self._write(relative, lambda p: frame.to_csv(p, index=False))

    def arrays(self, relative, **arrays):
        def write(path):
            with path.open("wb") as stream:
                np.savez_compressed(stream, **arrays)

        return self._write(relative, write)

    def complete(self, stage, provenance, paths):
        return self.json(
            f"checkpoints/{stage}.json",
            dict(
                provenance=fingerprint(provenance),
                artifacts={str(Path(p).relative_to(self.root)): checksum(p) for p in paths},
            ),
        )

    def valid(self, stage, provenance):
        path = self.root / f"checkpoints/{stage}.json"
        if not path.exists():
            return False
        record = json.loads(path.read_text())
        return record["provenance"] == fingerprint(provenance) and all(
            (self.root / p).is_file() and checksum(self.root / p) == digest
            for p, digest in record["artifacts"].items()
        )

    def finish(self):
        paths = [
            p
            for p in self.root.rglob("*")
            if p.is_file()
            and p != self.root / "checkpoints/run.json"
            and not p.name.endswith(".tmp")
        ]
        self.complete("run", self.manifest, paths)
