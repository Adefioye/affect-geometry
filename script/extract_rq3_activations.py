#!/usr/bin/env python3
"""Stage 1: extract raw/projected cohort activations for the three RQ3 models and upload."""

import argparse
import json
from pathlib import Path

from affect_geometry.experiments.rq3_stages import load_extraction_config, run_extraction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="run", choices=["run", "validate", "upload"])
    parser.add_argument("--config", type=Path, default=Path("configs/rq3/extract_activations.json"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--no-upload", action="store_true", help="Prepare local bundles only")
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else args.project_root / args.config
    result = run_extraction(
        load_extraction_config(path),
        args.project_root,
        command=args.command,
        publish=not args.no_upload,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
