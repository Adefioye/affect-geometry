from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from affect_geometry.config import DatasetConfig
from affect_geometry.logging import get_logger, progress


def prepare_goemotions(config: DatasetConfig, destination: Path) -> pd.DataFrame:
    """Download GoEmotions and persist only examples with exactly one label."""
    from datasets import load_dataset

    logger = get_logger()
    logger.info(
        "dataset.download name=%s subset=%s revision=%s",
        config.name,
        config.subset,
        config.revision,
    )
    dataset = load_dataset(
        config.name,
        config.subset,
        revision=config.revision,
    )
    frames: list[pd.DataFrame] = []
    label_names: list[str] | None = None
    for split in progress(
        config.splits,
        description="Preparing GoEmotions splits",
        unit="split",
    ):
        part = dataset[split]
        names = list(part.features["labels"].feature.names)
        label_names = label_names or names
        records = []
        for row_index, row in progress(
            enumerate(part),
            description=f"Filtering single-label {split}",
            total=len(part),
            unit="sample",
            leave=False,
        ):
            labels = row["labels"]
            if len(labels) != 1:
                continue
            label_id = int(labels[0])
            emotion = names[label_id]
            if not config.include_neutral and emotion == "neutral":
                continue
            text = str(row["text"])
            source_id = str(row.get("id", row_index))
            digest = hashlib.sha1(f"{split}:{source_id}:{text}".encode()).hexdigest()[:16]
            records.append(
                {
                    "sample_id": digest,
                    "source_id": source_id,
                    "source_row": row_index,
                    "split": split,
                    "label_id": label_id,
                    "emotion": emotion,
                    "text": text,
                }
            )
        frames.append(pd.DataFrame.from_records(records))
        logger.info(
            "dataset.split_complete split=%s retained=%d source=%d", split, len(records), len(part)
        )
    result = pd.concat(frames, ignore_index=True)
    destination.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination / "samples.csv", index=False)
    counts = result.groupby(["split", "emotion"], as_index=False).size()
    counts.to_csv(destination / "class_counts.csv", index=False)
    pd.DataFrame({"label_id": range(len(label_names or [])), "emotion": label_names or []}).to_csv(
        destination / "labels.csv", index=False
    )
    metadata = {
        "dataset": config.name,
        "subset": config.subset,
        "requested_revision": config.revision,
        "fingerprints": {split: dataset[split]._fingerprint for split in config.splits},
        "single_label_samples": len(result),
    }
    (destination / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info("dataset.complete retained=%d destination=%s", len(result), destination)
    return result


def load_samples(path: Path, split: str = "all") -> pd.DataFrame:
    samples = pd.read_csv(path)
    if split != "all":
        samples = samples.loc[samples["split"] == split]
    return samples.reset_index(drop=True)
