#!/usr/bin/env python3
"""Publish complete saved RQ3 results to their model's activation dataset repository."""

import argparse
import json
from pathlib import Path

from affect_geometry.experiments.rq3_stages import load_analysis_config, upload_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    config = load_analysis_config(args.project_root / args.config)
    print(json.dumps(upload_analysis(config, args.project_root), indent=2))


if __name__ == "__main__":
    main()
