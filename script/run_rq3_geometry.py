#!/usr/bin/env python3
"""Stage 2: analyze prepared cohort activations using one model's RQ3 configuration."""

import argparse
import json
from pathlib import Path

from affect_geometry.experiments.rq3_stages import load_analysis_config, run_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", nargs="?", default="run", choices=["run", "validate", "report", "compare"]
    )
    parser.add_argument("--config", type=Path, help="configs/rq3/{gemma2,apertus,olmo2}.json")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--compare-runs", nargs=3, type=Path, help="Completed runs, one per RQ3 model"
    )
    parser.add_argument("--comparison-id", default="rq3-model-comparison")
    parser.add_argument(
        "--no-upload", action="store_true", help="Write analysis artifacts locally only"
    )
    args = parser.parse_args()
    if args.command == "report":
        if args.run_dir is None:
            parser.error("--run-dir is required for report")
        from affect_geometry.experiments.rq3_stages import regenerate_report

        regenerate_report(args.project_root / args.run_dir)
    elif args.command == "compare":
        if not args.compare_runs:
            parser.error("--compare-runs requires three completed model runs")
        from affect_geometry.experiments.rq3_stages import compare_analyses

        print(
            json.dumps(
                compare_analyses(
                    args.compare_runs,
                    args.project_root,
                    args.comparison_id,
                    publish=not args.no_upload,
                ),
                indent=2,
            )
        )
    else:
        if args.config is None:
            parser.error("--config is required; select one model configuration")
        path = args.config if args.config.is_absolute() else args.project_root / args.config
        print(
            json.dumps(
                run_analysis(
                    load_analysis_config(path),
                    args.project_root,
                    publish=not args.no_upload,
                    validate_only=args.command == "validate",
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
