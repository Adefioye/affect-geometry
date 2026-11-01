#!/usr/bin/env python3
"""Thin RQ3 entry point: validate, extract, run, or replay persisted figures."""

import argparse
from pathlib import Path

from affect_geometry.experiments.correctness_geometry import run_panel
from affect_geometry.experiments.rq3_config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "extract", "run", "report"])
    parser.add_argument("--config", type=Path, default=Path("configs/rq3_geometry.json"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.command == "report":
        if args.run_dir is None:
            parser.error("--run-dir is required for report")
        from affect_geometry.reporting.correctness_geometry import render

        render(args.run_dir)
    else:
        config_path = args.config if args.config.is_absolute() else args.project_root / args.config
        run_panel(load_config(config_path), args.project_root, args.command)


if __name__ == "__main__":
    main()
