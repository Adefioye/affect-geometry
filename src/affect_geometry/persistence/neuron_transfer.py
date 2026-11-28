"""Atomic stage artifacts and transactionally resumable, configuration-bound records."""

import io
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from affect_geometry.persistence.correctness_geometry import (
    GeometryStore,
    fingerprint,
    jsonable,
)


class NeuronStore(GeometryStore):
    def parquet(self, relative, frame):
        return self._write(relative, lambda p: frame.to_parquet(p, index=False))

    def read_json(self, relative):
        return json.loads((self.root / relative).read_text())

    def require(self, stage, provenance):
        if not self.valid(stage, provenance):
            raise ValueError(f"Missing/stale {stage} stage; rerun its predecessors")

    def stream_parquet(self, relative, records, schema):
        import pyarrow as pa
        import pyarrow.parquet as pq

        def write(path):
            with pq.ParquetWriter(path, schema) as writer:
                batch = []
                for row in records:
                    batch.append(row)
                    if len(batch) == 512:
                        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                        batch.clear()
                if batch:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))

        return self._write(relative, write)


class RecordLedger:
    """One SQLite database per stage; no partially written scientific records survive crashes."""

    def __init__(self, path, provenance):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (fingerprint TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, payload TEXT)")
        found = self.db.execute("SELECT fingerprint FROM metadata").fetchone()
        expected = fingerprint(provenance)
        if found and found[0] != expected:
            self.db.close()
            raise ValueError("Ledger provenance changed; use a new run ID")
        if not found:
            self.db.execute("INSERT INTO metadata VALUES (?)", (expected,))
            self.db.commit()

    def get(self, key):
        row = self.db.execute("SELECT payload FROM records WHERE id=?", (str(key),)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, value):
        payload = json.dumps(jsonable(value), sort_keys=True, allow_nan=False)
        existing = self.db.execute("SELECT payload FROM records WHERE id=?", (str(key),)).fetchone()
        if existing and existing[0] != payload:
            raise ValueError(f"Attempt to overwrite a completed condition: {key}")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO records VALUES (?, ?)", (str(key), payload))

    def rows(self):
        for row in self.db.execute("SELECT payload FROM records ORDER BY id"):
            yield json.loads(row[0])

    def frame(self):
        return pd.DataFrame(self.rows())

    def put_attribution(self, key, value, totals):
        """Update all-neuron corpus sums in the same transaction as the sparse prompt record."""
        if self.get(key) is not None:
            return
        self.db.execute("CREATE TABLE IF NOT EXISTS sums (layer INTEGER PRIMARY KEY, data BLOB)")
        with self.db:
            for layer, values in totals.items():
                previous = self.db.execute(
                    "SELECT data FROM sums WHERE layer=?", (layer,)
                ).fetchone()
                values = np.asarray(values, dtype=np.float64)
                if previous:
                    values = values + np.load(io.BytesIO(previous[0]), allow_pickle=False)
                buffer = io.BytesIO()
                np.save(buffer, values, allow_pickle=False)
                self.db.execute(
                    "INSERT OR REPLACE INTO sums VALUES (?, ?)", (layer, buffer.getvalue())
                )
            payload = json.dumps(jsonable(value), sort_keys=True, allow_nan=False)
            self.db.execute("INSERT INTO records VALUES (?, ?)", (str(key), payload))

    def importance_means(self):
        exists = self.db.execute("SELECT name FROM sqlite_master WHERE name='sums'").fetchone()
        count = self.db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        frames = []
        if exists and count:
            for layer, data in self.db.execute("SELECT layer, data FROM sums ORDER BY layer"):
                values = np.load(io.BytesIO(data), allow_pickle=False) / count
                frames.append(
                    pd.DataFrame(
                        dict(layer=layer, channel=np.arange(len(values)), importance=values)
                    )
                )
        return (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["layer", "channel", "importance"])
        )

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
