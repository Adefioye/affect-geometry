import json

import numpy as np
import pytest

from affect_geometry.classification.protocol import EMOTIONS, digest
from affect_geometry.m03 import pipeline
from affect_geometry.m03.batch_pilot import diagnostic
from affect_geometry.m03.batch_pilot.comparison import layer_tables, validate_protocols
from affect_geometry.m03.configuration import Config
from affect_geometry.m03.extraction import Scorer, identity
from affect_geometry.m03.smoke import prepare_smoke, tiny_bundle


def test_candidate_failure_is_recorded_without_opening_regular_gate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config, root = prepare_smoke("diagnostic")
    bundle = tiny_bundle(config)
    monkeypatch.setattr(diagnostic.artifacts, "load", lambda *_: bundle)

    def scores(self, requests, *, shared=True, **kwargs):
        return [
            dict(
                label_scores=np.full((len(self.keys), 28), -1.0 if shared else -2.0),
                stop_scores=np.full((len(self.keys), 28), -1.0),
                token_counts=np.ones((len(self.keys), 28)),
                prefix_count=np.asarray(1),
                extraction_seconds=np.asarray(1.0),
            )
            for _ in requests
        ]

    monkeypatch.setattr(Scorer, "score", scores)
    with pytest.raises(AssertionError, match="Mismatched elements"):
        pipeline.pilot(config, "diagnostic", bundle=bundle)
    metadata = bundle[3]
    rows = pipeline.read_samples(root, "pilot", "canonical", config)
    requests = pipeline.paths_for(rows, "canonical", bundle[1], config)
    index = {
        row["sample_id"]: identity(req, digest(row["rendered_prompt"]), "canonical")
        for row, req in zip(rows, requests, strict=True)
    }
    diagnostic.run(
        config,
        "diagnostic",
        dict(
            labels=EMOTIONS, plan=metadata["plan"], versions=pipeline.versions(), readouts=metadata
        ),
        index,
    )
    check = json.loads((root / "technical_check.json").read_text())
    assert not check["candidate_score_agreement_passed"]
    assert "Mismatched elements" in check["failure"]
    assert diagnostic.read_checks(root, diagnostic=True)["complete"]
    assert len(pipeline.cached(root, "pilot", "canonical", config)[2]) == 2
    with pytest.raises(FileNotFoundError, match="pilot_checks.json"):
        pipeline.extract(config, "diagnostic", phase="reference", bundle=bundle)


def test_native_assertion_still_fails_diagnostic(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Native mismatch")

    monkeypatch.setattr(Scorer, "score", fail)
    scorer = object.__new__(diagnostic.DiagnosticScorer)
    with pytest.raises(RuntimeError, match="Scoring failed inside") as exc:
        scorer.check([1, 2])
    assert str(exc.value.__cause__) == "Native mismatch"


def test_paired_agreement_uses_complete_scores_and_separate_stop_component():
    label1 = np.full((1, 1, 28), -10.0)
    label1[0, 0, :2] = (-1.0, -1.125)
    label8 = label1.copy()
    label8[0, 0, 2] -= 0.25  # The threshold is strict.
    stop1 = np.full_like(label1, -1.0)
    stop8 = stop1.copy()
    stop8[0, 0, 0] -= 0.5
    counts = np.ones_like(label1)
    agreement, differences, flips = layer_tables(
        ["sample1"],
        [("logit", 1)],
        dict(label_scores=label1, stop_scores=stop1, token_counts=counts),
        dict(label_scores=label8, stop_scores=stop8, token_counts=counts),
        normalized=True,
        context=dict(tolerance=0.25),
    )
    assert dict(zip(agreement.score_mode, agreement.flip_count, strict=True)) == {
        "total": 1,
        "label_only": 0,
        "length_normalized": 1,
    }
    assert differences.count_above_tolerance.tolist() == [0, 1]
    assert flips.bs1_prediction.tolist() == [EMOTIONS[0], EMOTIONS[0]]
    assert flips.bs8_prediction.tolist() == [EMOTIONS[1], EMOTIONS[1]]
    assert flips.bs1_top_two_margin.tolist() == [0.125, 0.0625]


def test_pair_allows_only_batch_setting_differences():
    left = dict(config=Config().to_dict(), labels=EMOTIONS)
    right = dict(left, config=dict(left["config"], prefix_batch_size=1, projection_batch_size=1))
    validate_protocols(left, right)
    right["config"]["dtype"] = "float32"
    with pytest.raises(ValueError, match="differ beyond"):
        validate_protocols(left, right)
