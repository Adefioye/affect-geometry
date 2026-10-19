#!/usr/bin/env python
"""Recompute the comprehensive report's CSVs and figures from an existing run."""

import argparse
import json
from pathlib import Path

from affect_geometry.analysis.comprehensive import compute_report_tables
from affect_geometry.analysis.sampling import sampling_checks
from affect_geometry.config import ExperimentConfig, ModelConfig
from affect_geometry.logging import configure_run_logging, get_logger
from affect_geometry.reporting.comprehensive import render_report_figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=Path("outputs/qwen--qwen3-0.6b/20261019T091428Z")
    )
    parser.add_argument("--report", type=Path, default=Path("analyses"))
    parser.add_argument("--permutations", type=int, default=4999)
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())["config"]
    config = ExperimentConfig(
        run_id=manifest["run_id"],
        model=ModelConfig(**manifest["model"]),
        log_root=manifest.get("log_root", "logs"),
    )
    configure_run_logging(config, "comprehensive_report")
    if not args.plots_only:
        compute_report_tables(args.run, args.report / "tables", permutations=args.permutations)
        sampling_checks(args.run, args.report / "tables")
    render_report_figures(args.report)
    get_logger().info("command.complete command=comprehensive_report report=%s", args.report)


if __name__ == "__main__":
    main()
