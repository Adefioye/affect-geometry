"""Preparation, technical pilot, frozen extraction, and cached calibration stages."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from affect_geometry.classification.protocol import EMOTIONS, digest
from affect_geometry.classification.storage import atomic_json
from affect_geometry.logging import configure_file_logging, get_logger, progress
from affect_geometry.m03 import PROTOCOL, artifacts
from affect_geometry.m03.configuration import Config, fingerprint
from affect_geometry.m03.data import load_baseline, population_hash, render, selections
from affect_geometry.m03.extraction import Scorer, identity
from affect_geometry.m03.paths import canonical_paths
from affect_geometry.m03.rules import arms, log_prob
from affect_geometry.m03.storage import (
    ScoreStore,
    code_hash,
    file_hash,
    frozen_json,
    score_keys,
    write_parquet,
)


def setup(config: Config, run_id: str, command: str) -> Path:
    root = config.root(run_id)
    root.mkdir(parents=True, exist_ok=True)
    configure_file_logging(
        Path("logs") / f"{config.model_id.replace('/', '--')}__m03__{run_id}",
        command,
        model=config.model_id,
        run_id=run_id,
        output=root,
    )
    return root


def versions() -> dict:
    packages = {
        p: importlib.metadata.version(p)
        for p in (
            "torch",
            "transformers",
            "numpy",
            "pandas",
            "pyarrow",
            "huggingface-hub",
            "scipy",
            "matplotlib",
        )
    }
    return dict(python=platform.python_version(), **packages)


def protocol(root: Path, config: Config) -> dict:
    manifest = json.loads((root / "protocol.json").read_text())
    if manifest["config"] != config.to_dict() or manifest["code_hash"] != code_hash():
        raise ValueError("Frozen config or implementation changed; use a new run ID")
    for name, expected in manifest["prepared_sha256"].items():
        if file_hash(root / "prepared" / name) != expected:
            raise ValueError(f"Prepared source changed: {name}")
    return manifest


def prepare(config: Config, run_id: str) -> Path:
    root = setup(config, run_id, "m03.prepare")
    if (root / "protocol.json").exists():
        protocol(root, config)
        return root
    rows, source = load_baseline(config, root / "assets" / "hub")
    parts, duplicates = selections(rows, config)
    for name, part in parts.items():
        write_parquet(root / "prepared" / f"{name}.parquet", part)
    duplicates.to_csv(root / "prepared" / "duplicate_audit.csv", index=False)
    files = sorted((root / "prepared").iterdir())
    frozen_json(
        root / "protocol.json",
        dict(
            protocol=PROTOCOL,
            config=config.to_dict(),
            code_hash=code_hash(),
            source=source,
            labels=EMOTIONS,
            population_hash=population_hash(rows),
            selected_ids={name: [r["sample_id"] for r in part] for name, part in parts.items()},
            prepared_sha256={p.name: file_hash(p) for p in files},
            selection_rule="unique training text; SHA256(seed:text_hash), sample_id tie break",
            test_history=config.test_history,
        ),
    )
    get_logger().info("m03.prepared counts=%s", {k: len(v) for k, v in parts.items()})
    return root


def read_samples(root: Path, phase: str, variant: str, config: Config) -> list[dict]:
    if phase not in {"pilot", "reference", "test"}:
        raise ValueError("Only pilot/train-reference/test extraction is implemented")
    if variant not in {"canonical", "alternate_menu", "alternate_prompt"}:
        raise ValueError("Unknown prompt variant")
    if phase != "test" and variant != "canonical":
        raise ValueError("Reference and technical pilot use the canonical prompt")
    if variant == "alternate_menu" and not config.alternate_menu:
        raise ValueError("Alternate menu was not predeclared")
    if variant == "alternate_prompt" and not config.alternate_prompt:
        raise ValueError("Alternate prompt was not predeclared")
    rows = pd.read_parquet(root / "prepared" / f"{phase}.parquet").to_dict("records")
    if variant == "alternate_prompt":
        rows = sorted(rows, key=lambda r: digest(f"{config.seed}:{r['sample_id']}"))[
            : config.pilot_size
        ]
    return rows


def paths_for(rows, variant, tokenizer, config):
    return [
        canonical_paths(
            tokenizer,
            render(row, variant, config),
            label_prefix=config.label_prefix,
            stop_token_id=config.stop_token_id,
            max_new_tokens=config.max_new_tokens,
        )
        for row in rows
    ]


def open_store(root, phase, variant, frozen, manifest, keys):
    thresholds_hash = (
        file_hash(root / "thresholds.json")
        if phase == "test" and (root / "thresholds.json").exists()
        else None
    )
    return ScoreStore(
        root / "scores" / phase / variant,
        dict(
            freeze_hash=fingerprint(frozen),
            protocol_hash=fingerprint(manifest),
            phase=phase,
            variant=variant,
            thresholds_hash=thresholds_hash,
        ),
        keys,
    )


def _extract(config, root, phase, variant, manifest, frozen, bundle):
    model, tokenizer, readouts, metadata, load_seconds = bundle
    if fingerprint(metadata) != fingerprint(frozen["readouts"]):
        raise ValueError("Loaded lens artifacts differ from the frozen pilot")
    if versions() != frozen["versions"]:
        raise ValueError("Software versions differ from the frozen pilot")
    rows = read_samples(root, phase, variant, config)
    requests = paths_for(rows, variant, tokenizer, config)
    identities = {
        r["sample_id"]: identity(p, digest(render(r, variant, config)), variant)
        for r, p in zip(rows, requests, strict=True)
    }
    store = open_store(root, phase, variant, frozen, manifest, score_keys(readouts.plan))
    frozen_json(store.root / "index.json", identities)
    diagnostics = (
        set(
            sorted(identities, key=lambda s: digest(f"{config.seed}:{s}"))[: config.diagnostic_size]
        )
        if phase == "test" and variant == "canonical"
        else set()
    )
    scorer = Scorer(model, tokenizer, readouts, config)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    extracted, reused, prefixes, replayed = 0, 0, 0, 0
    chunks = list(range(0, len(rows), config.prefix_batch_size))
    for start in progress(chunks, description=f"M03 {phase}/{variant} comments", unit="batch"):
        indices = []
        batch_indices = list(range(start, min(start + config.prefix_batch_size, len(rows))))
        for i in batch_indices:
            sid = rows[i]["sample_id"]
            if store.get(sid, identities[sid]) is None:
                indices.append(i)
            else:
                reused += 1
        if not indices:
            continue
        try:
            # Preserve padding and projection batch context after a partial interruption.
            # Completed rows in an unfinished batch are replayed but never overwritten.
            values = scorer.score(
                [requests[i] for i in batch_indices],
                diagnostic=[rows[i]["sample_id"] in diagnostics for i in batch_indices],
                native_check=phase == "pilot",
            )
            pending = set(indices)
            replayed += len(batch_indices) - len(indices)
            for i, value in zip(batch_indices, values, strict=True):
                prefixes += int(value["prefix_count"])
                if i not in pending:
                    continue
                sid = rows[i]["sample_id"]
                store.put(sid, identities[sid], value)
                extracted += 1
        except Exception as exc:
            get_logger().exception(
                "m03.extraction_failed phase=%s samples=%s",
                phase,
                [rows[i]["sample_id"] for i in indices],
            )
            atomic_json(
                store.root / "last_error.json",
                dict(
                    samples=[rows[i]["sample_id"] for i in indices],
                    error=str(exc),
                ),
            )
            raise
    for row in rows:
        if store.get(row["sample_id"], identities[row["sample_id"]]) is None:
            raise ValueError("Incomplete extraction; metrics are forbidden")
    elapsed = time.perf_counter() - started
    runtime = dict(
        phase=phase,
        variant=variant,
        examples=len(rows),
        newly_extracted=extracted,
        reused=reused,
        replayed_for_batch_context=replayed,
        unique_candidate_prefixes=prefixes,
        backbone_batches=scorer.calls,
        extraction_seconds=elapsed,
        load_seconds=load_seconds,
        seconds_per_new_comment=elapsed / max(extracted, 1),
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
        peak_device_memory_bytes=torch.cuda.max_memory_allocated()
        if torch.cuda.is_available()
        else 0,
        prefix_batch_size=config.prefix_batch_size,
        projection_batch_size=config.projection_batch_size,
        device=config.device,
        host=platform.platform(),
        readout_count=len(scorer.keys),
        early_exit=False,
        kv_cache=False,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    atomic_json(root / "runtime" / f"{phase}_{variant}_{stamp}.json", runtime)
    atomic_json(store.root / "complete.json", dict(n=len(rows), index_hash=fingerprint(identities)))
    return scorer, runtime


def pilot(config: Config, run_id: str, *, bundle=None) -> Path:
    root = setup(config, run_id, "m03.pilot")
    manifest = protocol(root, config)
    bundle = bundle or artifacts.load(config, root / "assets" / "hub")
    model, tokenizer, readouts, metadata, _ = bundle
    rows = read_samples(root, "pilot", "canonical", config)
    requests = paths_for(rows, "canonical", tokenizer, config)
    scorer = Scorer(model, tokenizer, readouts, config)
    checks = scorer.technical_checks(requests)
    frozen = dict(
        readouts=metadata,
        plan=readouts.plan,
        versions=versions(),
        protocol_hash=fingerprint(manifest),
        labels=EMOTIONS,
    )
    # Artifacts and settings are frozen before any test extraction or accuracy calculation.
    frozen_json(root / "freeze.json", frozen)
    all_scorer, runtime = _extract(config, root, "pilot", "canonical", manifest, frozen, bundle)
    checks.update(
        pilot_samples=len(rows),
        gold_labels_used=False,
        all_prefix_native_max_abs_error=all_scorer.native_max_error,
        runtime=runtime,
        complete=True,
    )
    atomic_json(root / "pilot_checks.json", checks)
    return root


def extract(config: Config, run_id: str, *, phase="test", variant="all", bundle=None) -> Path:
    root = setup(config, run_id, "m03.extract")
    manifest = protocol(root, config)
    frozen = json.loads((root / "freeze.json").read_text())
    checks = json.loads((root / "pilot_checks.json").read_text())
    if not checks.get("complete"):
        raise ValueError("Complete the technical pilot first")
    if phase == "pilot":
        raise ValueError("Use the pilot command for technical checks")
    if phase == "reference" and not config.calibrated:
        raise ValueError("Unlabeled-reference calibration was not predeclared")
    if phase == "test" and config.calibrated and not (root / "thresholds.json").exists():
        raise ValueError("Freeze unlabeled training thresholds before test extraction")
    variants = [variant] if variant != "all" else ["canonical"]
    if variant == "all" and phase == "test":
        variants += [
            v
            for v, enabled in (
                ("alternate_menu", config.alternate_menu),
                ("alternate_prompt", config.alternate_prompt),
            )
            if enabled
        ]
    bundle = bundle or artifacts.load(config, root / "assets" / "hub")
    for item in variants:
        _extract(config, root, phase, item, manifest, frozen, bundle)
    return root


def cached(root: Path, phase: str, variant: str, config: Config):
    manifest = protocol(root, config)
    frozen = json.loads((root / "freeze.json").read_text())
    keys = score_keys(frozen["plan"])
    store = open_store(root, phase, variant, frozen, manifest, keys)
    rows = read_samples(root, phase, variant, config)
    index = json.loads((store.root / "index.json").read_text())
    complete = json.loads((store.root / "complete.json").read_text())
    if complete != dict(n=len(rows), index_hash=fingerprint(index)):
        raise ValueError("Extraction completeness marker differs")
    if set(index) != {r["sample_id"] for r in rows}:
        raise ValueError("Score population differs from prepared sample IDs")
    values = []
    for row in rows:
        sid = row["sample_id"]
        if index[sid]["prompt_hash"] != digest(render(row, variant, config)):
            raise ValueError("Cached prompt differs from the prepared sample")
        value = store.get(sid, index[sid])
        if value is None:
            raise ValueError(f"Missing score shard: {sid}")
        # Full vocabulary arrays are only loaded one sample at a time and discarded here.
        value.pop("diagnostic_logp", None)
        values.append(value)
    return rows, keys, values, frozen


def calibrate(config: Config, run_id: str) -> Path:
    if not config.calibrated:
        raise ValueError("Reference calibration was not predeclared")
    root = setup(config, run_id, "m03.calibrate")
    if (root / "thresholds.json").exists():
        frozen = json.loads((root / "freeze.json").read_text())
        existing = json.loads((root / "thresholds.json").read_text())
        if existing["freeze_hash"] != fingerprint(frozen):
            raise ValueError("Reference thresholds differ from frozen artifacts")
        return root
    if (root / "scores" / "test" / "canonical" / "index.json").exists():
        raise ValueError("Reference thresholds must be frozen before test extraction")
    rows, keys, values, frozen = cached(root, "reference", "canonical", config)
    scores = {
        key: np.stack([v["label_scores"][i] + v["stop_scores"][i] for v in values])
        for i, key in enumerate(keys)
    }
    thresholds = {
        a.name: np.quantile(np.exp(log_prob(a.scores)), 27 / 28, axis=0, method="linear").tolist()
        for a in arms(scores, frozen["plan"])
        if a.headline
    }
    frozen_json(
        root / "thresholds.json",
        dict(
            thresholds=thresholds,
            quantile=27 / 28,
            quantile_method="linear",
            comparison=">",
            sample_ids=[r["sample_id"] for r in rows],
            freeze_hash=fingerprint(frozen),
            reference_split="train",
            gold_labels_used=False,
        ),
    )
    return root
