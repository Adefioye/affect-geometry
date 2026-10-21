#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _upload_common import download_activation_run, load_project_environment, safe_slug

from affect_geometry.experiments.representation_comparison import (
    RepresentationComparisonConfig,
    run_representation_comparison,
)
from affect_geometry.logging import configure_file_logging, get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "kokolamba/affect-geometry-activations"
DEFAULT_OUTPUT_ROOT = "outputs/representation_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare last-token and mean-pooled activation representations using cosine RDMs, "
            "2D classical MDS, and NRC V-A Procrustes R-squared"
        )
    )
    last_token = parser.add_mutually_exclusive_group(required=True)
    last_token.add_argument("--last-token-run", help="Local last-token activation run")
    last_token.add_argument(
        "--last-token-path-in-repo",
        help="Last-token run directory inside the Hugging Face dataset",
    )
    mean = parser.add_mutually_exclusive_group(required=True)
    mean.add_argument("--mean-run", help="Local mean-pooled activation run")
    mean.add_argument(
        "--mean-path-in-repo",
        help="Mean-pooled run directory inside the Hugging Face dataset",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset containing remotely specified runs",
    )
    parser.add_argument(
        "--download-root",
        default="outputs/activation_sources",
        help="Local cache for activation runs downloaded from Hugging Face",
    )
    parser.add_argument(
        "--va-reference",
        default="data/reference/goemotions_nrc_va.csv",
        help="NRC valence-arousal reference CSV",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Comparison artifact root (default: outputs/representation_comparison)",
    )
    parser.add_argument("--log-root", default="logs", help="Central log root")
    parser.add_argument("--neutral-label", default="neutral")
    parser.add_argument("--neutral-variance-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--minimum-vector-norm", type=float, default=1e-12)
    return parser.parse_args()


def _resolve_run(
    *,
    local_run: str | None,
    path_in_repo: str | None,
    repo_id: str,
    download_root: str,
    pooling: str,
) -> Path:
    if local_run:
        return Path(local_run).expanduser().resolve()
    if path_in_repo is None:
        raise ValueError(f"No source was provided for {pooling}")
    return download_activation_run(
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        download_root=download_root,
        pooling=pooling,
    )


def main() -> None:
    args = parse_args()
    load_project_environment(PROJECT_ROOT)
    last_token_run = _resolve_run(
        local_run=args.last_token_run,
        path_in_repo=args.last_token_path_in_repo,
        repo_id=args.repo_id,
        download_root=args.download_root,
        pooling="last_token",
    )
    mean_run = _resolve_run(
        local_run=args.mean_run,
        path_in_repo=args.mean_path_in_repo,
        repo_id=args.repo_id,
        download_root=args.download_root,
        pooling="mean",
    )
    log_identifier = safe_slug(f"representation_comparison--{last_token_run.name}--{mean_run.name}")
    configure_file_logging(
        Path(args.log_root) / log_identifier,
        "run_representation_comparison",
        last_token_run=last_token_run,
        mean_run=mean_run,
        output=args.output_root,
    )
    logger = get_logger()
    try:
        result = run_representation_comparison(
            RepresentationComparisonConfig(
                last_token_run=last_token_run,
                mean_run=mean_run,
                va_reference=args.va_reference,
                output_root=args.output_root,
                neutral_label=args.neutral_label,
                neutral_variance_threshold=args.neutral_variance_threshold,
                batch_size=args.batch_size,
                minimum_vector_norm=args.minimum_vector_norm,
            )
        )
    except BaseException:
        logger.exception("command.failed command=run_representation_comparison")
        raise

    display = result.summary[
        ["rank", "variant", "mean_va_procrustes_r2", "max_va_procrustes_r2", "max_layer"]
    ]
    print(f"Representation comparison complete: {result.output_root.resolve()}")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
