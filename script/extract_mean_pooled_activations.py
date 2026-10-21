#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _upload_common import (
    download_activation_run,
    load_project_environment,
    print_plan,
    safe_slug,
    selected_files,
    upload_selection,
)

from affect_geometry.config import timestamp_run_id
from affect_geometry.experiments.mean_activation_extraction import (
    extract_mean_pooled_activations,
    mean_extraction_config,
)
from affect_geometry.logging import configure_run_logging, get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "kokolamba/affect-geometry-activations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract, validate, and upload mean-pooled activations using an existing "
            "last-token run as the reproducibility reference"
        )
    )
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument(
        "--reference-run",
        help="Local root of the reference last-token activation run",
    )
    reference.add_argument(
        "--reference-path-in-repo",
        help="Reference run directory inside the Hugging Face dataset repository",
    )
    parser.add_argument(
        "--reference-repo-id",
        help="Reference Hugging Face dataset; defaults to --repo-id",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Destination Hugging Face dataset repository",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="New run ID; defaults to the current UTC timestamp",
    )
    parser.add_argument("--output-root", default="outputs", help="Local activation output root")
    parser.add_argument("--log-root", default="logs", help="Central log root")
    parser.add_argument(
        "--download-root",
        default="outputs/activation_sources",
        help="Local cache for a reference downloaded from Hugging Face",
    )
    parser.add_argument("--device", help="Model device override, such as cuda or mps")
    parser.add_argument("--batch-size", type=int, help="Extraction batch-size override")
    parser.add_argument(
        "--model-revision",
        help="Explicit model commit when the reference run did not record a revision",
    )
    parser.add_argument(
        "--path-in-repo",
        help="Destination directory; defaults to <model-slug>/<run-id>",
    )
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--private", dest="private", action="store_true")
    visibility.add_argument("--public", dest="private", action="store_false")
    parser.set_defaults(private=True)
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Extract and validate locally without uploading",
    )
    return parser.parse_args()


def _resolve_reference(args: argparse.Namespace) -> Path:
    if args.reference_run:
        return Path(args.reference_run).expanduser().resolve()
    return download_activation_run(
        repo_id=args.reference_repo_id or args.repo_id,
        path_in_repo=args.reference_path_in_repo,
        download_root=args.download_root,
        pooling=None,
    )


def main() -> None:
    args = parse_args()
    load_project_environment(PROJECT_ROOT)
    reference_run = _resolve_reference(args)
    run_id = args.run_id or timestamp_run_id()
    config = mean_extraction_config(
        reference_run,
        run_id=run_id,
        output_root=args.output_root,
        log_root=args.log_root,
        device=args.device,
        batch_size=args.batch_size,
        model_revision=args.model_revision,
    )
    configure_run_logging(config, "extract_mean_pooled_activations")
    logger = get_logger()
    try:
        result = extract_mean_pooled_activations(reference_run, config)
        files = selected_files(result.run_directory, ["dataset", "activations"])
        path_in_repo = args.path_in_repo or f"{safe_slug(result.model_name)}/{result.run_id}"
        print_plan(
            run_directory=result.run_directory,
            repo_id=args.repo_id,
            path_in_repo=path_in_repo,
            files=files,
            private=args.private,
        )
        if args.no_upload:
            print("Mean activations validated; upload was skipped.")
            return
        upload_selection(
            run_directory=result.run_directory,
            repo_id=args.repo_id,
            path_in_repo=path_in_repo,
            allow_patterns=["manifest.json", "dataset/**", "activations/**"],
            private=args.private,
            commit_message=(
                f"Upload mean-pooled activations for {result.model_name} run {result.run_id}"
            ),
        )
        logger.info("upload.complete repo=%s path=%s", args.repo_id, path_in_repo)
        print("Mean-pooled activations uploaded successfully.")
    except BaseException:
        logger.exception("command.failed command=extract_mean_pooled_activations")
        raise


if __name__ == "__main__":
    main()
