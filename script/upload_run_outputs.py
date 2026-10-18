#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _upload_common import (
    load_manifest,
    print_plan,
    resolve_model,
    resolve_run_directory,
    safe_slug,
    selected_files,
    upload_selection,
)

BASE_DIRECTORIES = (
    "dataset",
    "checkpoints",
    "rdms",
    "embeddings",
    "analyses",
    "plot_data",
)
REQUIRED_DIRECTORIES = ("rdms", "embeddings", "analyses", "plot_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload compact outputs for one affect-geometry run to a Hugging Face dataset"
    )
    parser.add_argument("--run-id", required=True, help="Directory name below --output-root")
    parser.add_argument("--model", help="Expected model name; defaults to the run manifest")
    parser.add_argument(
        "--repo-id", required=True, help="Hugging Face dataset repo, e.g. user/runs"
    )
    parser.add_argument("--output-root", default="outputs", help="Local output root")
    parser.add_argument(
        "--path-in-repo",
        help="Remote directory; defaults to models/<model>/runs/<run-id>",
    )
    parser.add_argument(
        "--include-plots",
        action="store_true",
        help="Include regenerable PNG plots (excluded by default)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Upload even if one or more completed-analysis directories are absent",
    )
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--private", dest="private", action="store_true")
    visibility.add_argument("--public", dest="private", action="store_false")
    parser.set_defaults(private=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate and show the upload only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_directory = resolve_run_directory(args.output_root, args.run_id)
    manifest = load_manifest(run_directory)
    model = resolve_model(manifest, args.model)

    missing = [name for name in REQUIRED_DIRECTORIES if not (run_directory / name).is_dir()]
    if missing and not args.allow_partial:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"Run is missing expected result directories: {joined}. "
            "Use --allow-partial to upload an incomplete run."
        )

    directories = [*BASE_DIRECTORIES]
    if args.include_plots:
        directories.append("plots")
    files = selected_files(run_directory, directories)
    if not files:
        raise FileNotFoundError("No run-result files were selected")

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

    allow_patterns = ["manifest.json", *(f"{name}/**" for name in directories)]
    upload_selection(
        run_directory=run_directory,
        repo_id=args.repo_id,
        path_in_repo=path_in_repo,
        allow_patterns=allow_patterns,
        private=args.private,
        commit_message=f"Upload results for {model} run {args.run_id}",
    )
    print("Run outputs uploaded successfully.")


if __name__ == "__main__":
    main()
