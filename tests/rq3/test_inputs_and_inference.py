import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from affect_geometry.activations.classification_prompt import (
    extract_prompt_states,
    open_prompt_states,
)
from affect_geometry.analysis.geometry_inference import infer
from affect_geometry.classification.protocol import digest, prompt
from affect_geometry.data.classification_source import read_classification, read_targets
from affect_geometry.persistence.correctness_geometry import GeometryStore


def test_prompt_extraction_pre_generation_last_nonpadding(monkeypatch, tmp_path):
    import torch

    from affect_geometry.classification import inference

    class Tokenizer:
        chat_template = None

        def __call__(self, texts, **kwargs):
            assert kwargs["truncation"] is False
            lengths = [len(text) for text in texts]
            ids = torch.zeros((len(texts), max(lengths)), dtype=torch.long)
            mask = ids.clone()
            for i, text in enumerate(texts):
                ids[i, : len(text)] = torch.tensor([ord(c) % 7 + 1 for c in text])
                mask[i, : len(text)] = 1
            return {"input_ids": ids, "attention_mask": mask}

    class Block(torch.nn.Module):
        def forward(self, hidden):
            return hidden + 1

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(layers=[Block(), Block()])
            self.config = SimpleNamespace(hidden_size=3, max_position_embeddings=100000)
            self.device = torch.device("cpu")

        def forward(self, input_ids, **kwargs):
            hidden = input_ids[..., None].repeat(1, 1, 3).float()
            for layer in self.model.layers:
                hidden = layer(hidden)
            return hidden

        def generate(self, **kwargs):
            raise AssertionError("RQ3 must never generate answers")

    class Generator:
        def __init__(self, *args, **kwargs):
            self.model, self.tokenizer = Model(), Tokenizer()

    monkeypatch.setattr(inference, "Generator", Generator)
    rows = []
    for i, text in enumerate(["short", "a much longer source comment"]):
        row = dict(sample_id=str(i), text=text, emotion="neutral", label_id=27)
        rendered, _ = prompt(row, 42)
        rows.append(
            dict(
                row,
                rendered_prompt=rendered,
                prompt_hash=digest(rendered),
                input_token_count=len(rendered),
            )
        )
    frame = pd.DataFrame(rows)
    manifest = dict(
        model_id="fixture",
        model_revision="pinned",
        seed=42,
        chat=False,
        chat_template=None,
        run_id="source",
        max_new_tokens=10,
    )
    extract_prompt_states(frame, manifest, tmp_path, batch_size=2)
    _, paths, meta, _ = open_prompt_states(tmp_path, frame, manifest)
    for layer, path in enumerate(paths):
        assert np.all(np.load(path) == ord(":") % 7 + 1 + layer + 1)
    positions = pd.read_csv(tmp_path / "token_positions.csv")
    assert positions.token_position.tolist() == (frame.input_token_count - 1).tolist()
    assert meta["input_condition"] == "classification_prompt_before_generation"
    changed = frame.copy()
    changed.loc[0, "prompt_hash"] = "incorrect"
    with pytest.raises(ValueError, match="hashes"):
        open_prompt_states(tmp_path, changed, manifest)


def test_permutations_refit_selection_all_cells_and_resume(monkeypatch, tmp_path):
    from affect_geometry.analysis import geometry_inference

    calls = []
    original = geometry_inference.evaluate

    def tracked(candidates, y, **kwargs):
        calls.append(y.copy())
        return original(candidates, y, **kwargs)

    monkeypatch.setattr(geometry_inference, "evaluate", tracked)
    rng = np.random.default_rng(9)
    y = rng.normal(size=(6, 1))
    x = rng.normal(size=(6, 1))
    cells = {("all", "mds", layer): {"fixed": x + layer} for layer in (0, 1)}
    schedule = np.array([[5, 4, 3, 2, 1, 0], [1, 2, 3, 4, 5, 0]])
    store = GeometryStore(tmp_path, {"test": True})
    result = infer(cells, y, [0, 1], schedule, selection="fixed", store=store)
    assert len(calls) == 4
    assert np.array_equal(calls[0], y[schedule[0]])
    assert np.array_equal(calls[1], y[schedule[0]])
    assert np.allclose(result["aggregate_null"][:, 0], result["layerwise_null"].mean(1))
    restored = infer(cells, y, [0, 1], schedule, selection="fixed", store=store)
    assert len(calls) == 4
    assert np.array_equal(restored["layerwise_null"], result["layerwise_null"])
    record = json.loads((tmp_path / "inference/permutations/0000.json").read_text())
    assert len(record["transformations"][0]["folds"]) == 6


def test_nrc_mapping_and_missing_full_source(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    target, meta = read_targets(
        root / "data/reference/goemotions_nrc_vad.csv",
        root / "data/reference/goemotions_nrc_vad.provenance.json",
        ["neutral", "joy"],
    )
    assert target.shape == (2, 2) and meta["version"] == "2.1"
    with pytest.raises(ValueError):
        read_targets(
            root / "data/reference/goemotions_nrc_vad.csv",
            root / "data/reference/goemotions_nrc_vad.provenance.json",
            ["joy"],
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(model_id="fixture", model_revision="main"))
    )
    with pytest.raises(ValueError):
        read_classification(tmp_path, "fixture")
