"""Orchestrate classification with an injectable generator for offline verification."""

from affect_geometry.logging import get_logger, progress

from .protocol import PROTOCOL, score
from .storage import Ledger, snapshot


def classify(rows, generator, root, manifest, retries=2):
    ledger = Ledger(root, manifest)
    try:
        pending = [
            row
            for row in rows
            if not (
                (previous := ledger.get(row["sample_id"])) and previous["status"] == "completed"
            )
        ]
        batch_size = manifest.get("batch_size", 1)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

        def predictions():
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                cached = None
                if hasattr(generator, "batch"):
                    try:
                        cached = generator.batch(batch, manifest["seed"])
                        if len(cached) != len(batch):
                            raise ValueError("Generator returned wrong batch length")
                    except Exception:
                        get_logger().exception(
                            "classification.batch_failed size=%d; retry singly", len(batch)
                        )
                        cached = None
                    if cached is None:
                        # The batch call's exception frame has been released before clearing CUDA.
                        import sys

                        torch = sys.modules.get("torch")
                        if torch is not None and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                for index, row in enumerate(batch):
                    yield row, cached[index] if cached is not None else None

        for row, cached in progress(
            predictions(),
            total=len(pending),
            description="Classifying single-label comments",
            unit="sample",
        ):
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
                    generated = cached if cached is not None else generator(row, manifest["seed"])
                    cached = None
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
