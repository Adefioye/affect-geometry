"""Transactional resume ledger and typed Parquet snapshots."""

import json
import os
import sqlite3
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


class Ledger:
    def __init__(self, root, manifest):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        target = root / "manifest.json"
        if target.exists() and json.loads(target.read_text()) != manifest:
            raise ValueError("Resume configuration differs; choose a new run directory")
        atomic_json(target, manifest)
        (root / "checkpoints").mkdir(exist_ok=True)
        self.db = sqlite3.connect(root / "checkpoints" / "predictions.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, payload TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS attempts (id TEXT, payload TEXT)")

    def get(self, sample_id):
        item = self.db.execute("SELECT payload FROM records WHERE id=?", (sample_id,)).fetchone()
        return json.loads(item[0]) if item else None

    def put(self, row):
        payload = json.dumps(row, ensure_ascii=False)
        with self.db:
            self.db.execute("INSERT INTO attempts VALUES (?,?)", (row["sample_id"], payload))
            self.db.execute(
                "INSERT OR REPLACE INTO records VALUES (?,?)", (row["sample_id"], payload)
            )

    def rows(self):
        return [
            json.loads(r[0]) for r in self.db.execute("SELECT payload FROM records ORDER BY id")
        ]

    def close(self):
        self.db.close()


def snapshot(root, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root) / "dataset"
    root.mkdir(exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    table = pa.Table.from_pylist([{k: row.get(k) for k in keys} for row in rows])
    # Explicit nullable types also work when a whole snapshot contains errors or invalid outputs.
    types = {
        "correct": pa.bool_(),
        "valid_label": pa.bool_(),
        "format_compliant": pa.bool_(),
        "compat_correct": pa.bool_(),
        "compat_valid_label": pa.bool_(),
        "predicted_emotion": pa.string(),
        "compat_predicted_emotion": pa.string(),
        "predicted_label_id": pa.int64(),
        "generated_token_ids": pa.list_(pa.int64()),
    }
    for key, dtype in types.items():
        if key in table.column_names:
            i = table.column_names.index(key)
            table = table.set_column(i, key, table[key].cast(dtype))
    for split in ("train", "validation", "test"):
        import pyarrow.compute as pc

        part = table.filter(pc.equal(table["source_split"], split))
        tmp = root / f"{split}.tmp"
        pq.write_table(part, tmp)
        os.replace(tmp, root / f"{split}.parquet")
