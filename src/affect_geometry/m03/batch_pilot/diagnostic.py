"""Run the standard scorer with a pilot-only candidate-agreement diagnostic."""

import json

from affect_geometry.classification.protocol import EMOTIONS, digest
from affect_geometry.classification.storage import atomic_json
from affect_geometry.logging import get_logger
from affect_geometry.m03 import artifacts, pipeline
from affect_geometry.m03.configuration import fingerprint
from affect_geometry.m03.data import render
from affect_geometry.m03.extraction import Scorer, identity
from affect_geometry.m03.storage import frozen_json


class DiagnosticScorer(Scorer):
    def score(self, *args, **kwargs):
        # Keep native-logit assertions and any other scoring assertions fatal.
        try:
            return super().score(*args, **kwargs)
        except AssertionError as exc:
            raise RuntimeError("Scoring failed inside the technical check") from exc

    def check(self, requests):
        try:
            return dict(candidate_score_agreement_passed=True, **self.technical_checks(requests))
        except AssertionError as exc:
            # The remaining assertions in technical_checks compare candidate scores.
            get_logger().exception("m03.diagnostic_candidate_score_agreement_failed")
            return dict(
                candidate_score_agreement_passed=False,
                candidate_score_atol=self.config.score_atol,
                native_max_abs_logit_error=self.native_max_error,
                native_atol=self.config.native_atol,
                native_rtol=self.config.native_rtol,
                checked_sample_count=len(requests[:2]),
                failure=str(exc),
            )


def run(config, run_id, reference_freeze, reference_index):
    root = pipeline.setup(config, run_id, "m03.batch_pilot.diagnostic")
    if (root / "pilot_checks.json").exists():
        raise ValueError("Use a separate run ID for the diagnostic pilot")
    manifest = pipeline.protocol(root, config)
    bundle = artifacts.load(config, root / "assets" / "hub")
    model, tokenizer, readouts, metadata, _ = bundle
    frozen = dict(
        readouts=metadata,
        plan=readouts.plan,
        versions=pipeline.versions(),
        protocol_hash=fingerprint(manifest),
        labels=EMOTIONS,
    )
    expected = dict(reference_freeze, protocol_hash=fingerprint(manifest))
    if fingerprint(frozen) != fingerprint(expected):
        raise ValueError("Diagnostic model, lens artifacts, or environment differ from bs1")
    rows = pipeline.read_samples(root, "pilot", "canonical", config)
    requests = pipeline.paths_for(rows, "canonical", tokenizer, config)
    index = {
        row["sample_id"]: identity(request, digest(render(row, "canonical", config)), "canonical")
        for row, request in zip(rows, requests, strict=True)
    }
    if index != reference_index or any(tuple(req.labels) != tuple(EMOTIONS) for req in requests):
        raise ValueError("Diagnostic canonical prompts or candidate token paths differ from bs1")
    checks = DiagnosticScorer(model, tokenizer, readouts, config).check(requests)
    checks.update(
        mode="candidate_score_agreement_diagnostic",
        phase="pilot",
        variant="canonical",
        checked_sample_ids=[row["sample_id"] for row in rows[:2]],
        gold_labels_used=False,
    )
    atomic_json(root / "technical_check.json", checks)
    frozen_json(root / "freeze.json", frozen)
    scorer, runtime = pipeline._extract(
        config, root, "pilot", "canonical", manifest, frozen, bundle
    )
    atomic_json(
        root / "diagnostic_checks.json",
        dict(
            checks,
            complete=True,
            pilot_samples=len(rows),
            all_prefix_native_max_abs_error=scorer.native_max_error,
            runtime=runtime,
        ),
    )


def read_checks(root, *, diagnostic):
    filename = "diagnostic_checks.json" if diagnostic else "pilot_checks.json"
    checks = json.loads((root / filename).read_text())
    if not checks.get("complete") or checks.get("gold_labels_used") is not False:
        raise ValueError(f"Incomplete or invalid pilot checks: {root / filename}")
    return checks
