"""Preserve source attributes and source splits, excluding only multilabel rows."""

from .protocol import EMOTIONS, digest

EXPECTED = {"train": 36308, "validation": 4548, "test": 4590}
SOURCE = {"train": 43410, "validation": 5426, "test": 5427}


def prepare(dataset, identity):
    rows, audit = [], []
    for split in EXPECTED:
        part = dataset[split]
        names = list(part.features["labels"].feature.names)
        if names != EMOTIONS:
            raise ValueError("Unexpected source vocabulary")
        retained = 0
        for index, original in enumerate(part):
            if len(original["labels"]) != 1:
                continue
            row = dict(original)
            row.update(
                sample_id=digest(f"{identity}:{split}:{row['id']}"),
                source_split=split,
                source_row=index,
                text_hash=digest(row["text"]),
                label_id=row["labels"][0],
                emotion=names[row["labels"][0]],
            )
            rows.append(row)
            retained += 1
        audit.append(
            dict(split=split, source=len(part), retained=retained, excluded=len(part) - retained)
        )
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source identifiers")
    return rows, audit
