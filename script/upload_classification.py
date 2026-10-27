#!/usr/bin/env python3
"""Publish a previously evaluated single-model dataset and its local metrics; no inference."""

import argparse
from pathlib import Path

from affect_geometry.classification.configuration import load_config
from affect_geometry.classification.publish import publish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--config", type=Path, help="Use this model config to locate its local run")
    parser.add_argument("--repo-id", required=True, help="Destination namespace/dataset-name")
    parser.add_argument("--public", action="store_true", help="Default is private")
    args = parser.parse_args()
    root = load_config(args.config)["run_dir"] if args.config else args.run_dir
    if args.config:
        import json

        from affect_geometry.classification.protocol import MODELS

        expected = MODELS[load_config(args.config)["model"]][0]
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["model_id"] != expected:
            parser.error("Config model does not match the saved run")
    publish(root, args.repo_id, private=not args.public)


if __name__ == "__main__":
    main()
