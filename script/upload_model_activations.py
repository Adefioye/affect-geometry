#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import numpy as np
from _upload_common import (
    load_manifest,
    print_plan,
    resolve_model,
    resolve_run_directory,
    safe_slug,
    selected_files,
    upload_selection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a validated model activation store to a Hugging Face dataset"
    )
    parser.add_argument("--run-id", required=True, help="Directory name below --output-root")
    parser.add_argument("--model", help="Expected model name; defaults to the run manifest")
    parser.add_argument(
        "--repo-id", required=True, help="Hugging Face dataset repo, e.g. user/activations"
    )
    parser.add_argument("--output-root", default="outputs", help="Local output root")
    parser.add_argument(
        "--path-in-repo",
        help="Remote directory; defaults to models/<model>/runs/<run-id>",
    )
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--private", dest="private", action="store_true")
    visibility.add_argument("--public", dest="private", action="store_false")
    parser.set_defaults(private=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate and show the upload only")
    return parser.parse_args()


def validate_activations(run_directory, manifest: dict) -> None:
    metadata_path = run_directory / "activations" / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Missing activations/metadata.json. Extraction may be incomplete; refusing to upload."
        )
    metadata = json.loads(metadata_path.read_text())
    n_samples = int(metadata["n_samples"])
    n_layers = int(metadata["n_layers"])
    hidden_size = int(metadata["hidden_size"])
    try:
        poolings = manifest["config"]["activations"]["poolings"]
    except (KeyError, TypeError) as error:
        raise ValueError("manifest.json does not list activation poolings") from error

    for pooling in poolings:
        directory = run_directory / "activations" / pooling
        layer_files = sorted(directory.glob("layer_*.npy"))
        if len(layer_files) != n_layers:
            raise ValueError(
                f"Expected {n_layers} layers for pooling {pooling!r}, found {len(layer_files)}"
            )
        for path in layer_files:
            array = np.load(path, mmap_mode="r")
            if array.shape != (n_samples, hidden_size):
                raise ValueError(
                    f"Unexpected activation shape in {path}: {array.shape}; "
                    f"expected {(n_samples, hidden_size)}"
                )


def main() -> None:
    args = parse_args()
    run_directory = resolve_run_directory(args.output_root, args.run_id)
    manifest = load_manifest(run_directory)
    model = resolve_model(manifest, args.model)
    validate_activations(run_directory, manifest)

    directories = ["dataset", "activations"]
    files = selected_files(run_directory, directories)
    path_in_repo = args.path_in_repo or (
        f"models/{safe_slug(model)}/runs/{safe_slug(args.run_id)}"
    )
    print_plan(
        run_directory=run_directory,
        repo_id=args.repo_id,
        path_in_repo=path_in_repo,
        files=files,
        private=args.private,
    )
    if args.dry_run:
        print("Dry run complete; nothing was uploaded.")
        return

    upload_selection(
        run_directory=run_directory,
        repo_id=args.repo_id,
        path_in_repo=path_in_repo,
        allow_patterns=["manifest.json", "dataset/**", "activations/**"],
        private=args.private,
        commit_message=f"Upload activations for {model} run {args.run_id}",
    )
    print("Model activations uploaded successfully.")


if __name__ == "__main__":
    main()
