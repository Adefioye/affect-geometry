"""Orchestrate classification with an injectable generator for offline verification."""

from affect_geometry.logging import get_logger, progress

from .protocol import PROTOCOL, score
from .storage import Ledger, snapshot


def classify(rows, generator, root, manifest, retries=2):
    ledger = Ledger(root, manifest)
    try:
        for row in progress(rows, description="Classifying single-label comments", unit="sample"):
            previous = ledger.get(row["sample_id"])
            if previous and previous["status"] == "completed":
                continue
            for _ in range(retries + 1):
                previous = ledger.get(row["sample_id"])
                record = dict(
                    row,
                    model_id=manifest["model_id"],
                    model_revision=manifest["model_revision"],
                    tokenizer_revision=manifest["model_revision"],
                    run_id=manifest["run_id"],
                    protocol_version=PROTOCOL,
                    attempt_count=(previous["attempt_count"] if previous else 0) + 1,
                )
                try:
                    generated = generator(row, manifest["seed"])
                    record.update(generated)
                    record.update(score(generated["raw_response"], row["emotion"]))
                    record.update(status="completed", error_type=None)
                except Exception as exc:
                    get_logger().exception("classification.error sample=%s", row["sample_id"])
                    record.update({k: None for k in score("", row["emotion"])})
                    record.update(
                        status="error",
                        outcome="inference_error",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        generation_finish_reason=None,
                        raw_response=None,
                        generated_token_ids=None,
                    )
                ledger.put(record)
                if record["status"] == "completed":
                    break
        result = ledger.rows()
        if {r["sample_id"] for r in result} != {r["sample_id"] for r in rows}:
            raise ValueError("Ledger source rows differ")
        snapshot(root, result)
        return result
    finally:
        ledger.close()
