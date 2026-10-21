import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import affect_geometry.experiments.mean_activation_extraction as extraction_module
from affect_geometry.experiments.mean_activation_extraction import (
    extract_mean_pooled_activations,
    mean_extraction_config,
)


def _write_reference(root: Path) -> pd.DataFrame:
    (root / "dataset").mkdir(parents=True)
    (root / "activations").mkdir()
    samples = pd.DataFrame(
        {
            "sample_id": ["sample-c", "sample-a", "sample-b"],
            "source_id": ["c", "a", "b"],
            "source_row": [2, 0, 1],
            "split": ["train", "train", "validation"],
            "label_id": [2, 0, 1],
            "emotion": ["neutral", "joy", "sadness"],
            "text": ["plain", "great", "bad"],
        }
    )
    samples.to_csv(root / "dataset" / "samples.csv", index=False)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    "run_id": "reference-run",
                    "seed": 17,
                    "dataset": {},
                    "model": {
                        "name": "synthetic/model",
                        "revision": None,
                        "device": "auto",
                        "dtype": "auto",
                        "batch_size": 8,
                        "max_length": 256,
                    },
                    "activations": {
                        "poolings": ["last_token"],
                        "storage_dtype": "float32",
                    },
                }
            }
        )
    )
    (root / "activations" / "metadata.json").write_text(
        json.dumps(
            {
                "n_samples": len(samples),
                "n_layers": 1,
                "hidden_size": 3,
                "model": "synthetic/model",
                "model_revision": "default",
                "resolved_model_revision": "fixed-commit",
            }
        )
    )
    return samples


@pytest.mark.parametrize("missing_revision", [False, True])
def test_mean_extraction_preserves_reference_and_validates_store(
    tmp_path: Path,
    monkeypatch,
    missing_revision: bool,
) -> None:
    reference = tmp_path / "reference"
    expected_samples = _write_reference(reference)
    if missing_revision:
        metadata_path = reference / "activations" / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["resolved_model_revision"] = None
        metadata_path.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="explicit model_revision"):
            mean_extraction_config(reference, run_id="missing-revision")

    def fake_extract(samples, destination, model_config, activation_config):
        assert samples["sample_id"].tolist() == expected_samples["sample_id"].tolist()
        assert model_config.revision == "fixed-commit"
        assert model_config.max_length == 256
        assert activation_config.poolings == ["mean"]
        (destination / "mean").mkdir(parents=True)
        np.save(
            destination / "mean" / "layer_000.npy",
            np.arange(9, dtype=np.float32).reshape(3, 3),
        )
        return {
            "n_samples": 3,
            "n_layers": 1,
            "hidden_size": 3,
            "model": "synthetic/model",
            "model_revision": "fixed-commit",
            "resolved_model_revision": "fixed-commit",
        }

    monkeypatch.setattr(extraction_module, "extract_activations", fake_extract)
    config = mean_extraction_config(
        reference,
        run_id="20261020T120000Z",
        output_root=tmp_path / "outputs",
        log_root=tmp_path / "logs",
        model_revision="fixed-commit" if missing_revision else None,
    )
    result = extract_mean_pooled_activations(reference, config)

    assert result.n_samples == 3
    assert result.n_layers == 1
    assert result.hidden_size == 3
    copied = pd.read_csv(result.run_directory / "dataset" / "samples.csv")
    assert copied["sample_id"].tolist() == expected_samples["sample_id"].tolist()
    manifest = json.loads((result.run_directory / "manifest.json").read_text())
    assert manifest["config"]["activations"]["poolings"] == ["mean"]
    assert manifest["config"]["model"]["revision"] == "fixed-commit"
    metadata = json.loads((result.run_directory / "activations" / "metadata.json").read_text())
    assert metadata["reference_model_identity_verified"] is (not missing_revision)
