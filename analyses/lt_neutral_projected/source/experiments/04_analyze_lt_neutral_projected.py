#!/usr/bin/env python
"""Build the complete LT-NeutralProjected audit and embedded Markdown report."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

from affect_geometry.analysis.projected_audit import extended_projected_tables
from affect_geometry.logging import configure_file_logging, get_logger
from affect_geometry.reporting.projected_audit import render_projected_audit
from affect_geometry.reporting.projected_bundle import bundle_projected_report
from affect_geometry.reporting.projected_report import write_projected_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=Path(
            "outputs/lt_neutral_projected/qwen--qwen3-0.6b/lt-neutral-projected-full-space-20261021"
        ),
    )
    parser.add_argument("--report", type=Path, default=Path("analyses/lt_neutral_projected"))
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--bundle-only", action="store_true")
    args = parser.parse_args()
    artifact = args.run / "report"
    configure_file_logging(
        "logs/qwen--qwen3-0.6b__lt-neutral-projected-analysis-20261021",
        "projected_report",
        run=args.run,
        report=args.report,
    )
    if not args.render_only and not args.bundle_only:
        extended_projected_tables(
            args.run,
            artifact / "tables",
            permutations=4999,
            eigengap_permutations=999,
            repetitions=200,
            seed=1701,
        )
    if not args.bundle_only:
        render_projected_audit(artifact)
    write_projected_report(args.run, args.report)
    versions = {
        p: importlib.metadata.version(p)
        for p in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "tqdm"]
    }
    (artifact / "analysis_environment.json").write_text(json.dumps(versions, indent=2) + "\n")
    source_hashes = {}
    source_paths = [
        *Path("src/affect_geometry/analysis").glob("*.py"),
        *Path("src/affect_geometry/embeddings").glob("*.py"),
        *Path("src/affect_geometry/reporting").glob("*projected*.py"),
        *Path("src/affect_geometry/logging").glob("*.py"),
        Path("src/affect_geometry/config.py"),
        Path(__file__),
        Path("tests/test_projected_audit.py"),
    ]
    for source in source_paths:
        relative = source.resolve().relative_to(Path.cwd().resolve())
        target = artifact / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    (artifact / "analysis_source_sha256.json").write_text(
        json.dumps(source_hashes, indent=2) + "\n"
    )
    bundle_projected_report(args.run, args.report)
    get_logger().info("command.complete command=projected_report output=%s", args.report)


if __name__ == "__main__":
    main()
