import json
from pathlib import Path

import numpy as np
import pandas as pd

from affect_geometry.activations.source import ActivationSource, align_activation_sources


def _write_source(
    root: Path,
    pooling: str,
    sample_ids: list[str],
    values: list[float],
) -> None:
    (root / "dataset").mkdir(parents=True)
    (root / "activations" / pooling).mkdir(parents=True)
    emotions = {"a": "joy", "b": "sadness", "c": "neutral"}
    samples = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "source_id": sample_ids,
            "source_row": range(len(sample_ids)),
            "split": "train",
            "label_id": [0 if emotions[item] == "joy" else 1 for item in sample_ids],
            "emotion": [emotions[item] for item in sample_ids],
            "text": "synthetic",
        }
    )
    samples.to_csv(root / "dataset" / "samples.csv", index=False)
    np.save(
        root / "activations" / pooling / "layer_000.npy",
        np.asarray(values, dtype=np.float32)[:, None],
    )
    (root / "activations" / "metadata.json").write_text(
        json.dumps(
            {
                "n_samples": len(sample_ids),
                "n_layers": 1,
                "hidden_size": 1,
                "model": "synthetic",
                "model_revision": "same",
            }
        )
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    "run_id": root.name,
                    "model": {"name": "synthetic"},
                    "activations": {"poolings": [pooling]},
                }
            }
        )
    )


def test_activation_sources_align_rows_by_sample_id(tmp_path: Path) -> None:
    last_root, mean_root = tmp_path / "last", tmp_path / "mean"
    _write_source(last_root, "last_token", ["a", "b", "c"], [1, 2, 3])
    _write_source(mean_root, "mean", ["c", "a", "b"], [3, 1, 2])

    last = ActivationSource.open(last_root, "last_token")
    mean = ActivationSource.open(mean_root, "mean")
    alignment = align_activation_sources(last, mean)

    assert alignment.candidate_row_order.tolist() == [1, 2, 0]
    aligned = mean.load_layer(0, row_order=alignment.candidate_row_order)
    assert aligned[:, 0].tolist() == [1, 2, 3]
    assert alignment.samples["emotion"].tolist() == ["joy", "sadness", "neutral"]
