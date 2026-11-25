#!/usr/bin/env python3
"""Upload saved valence-transfer results to the corresponding Hugging Face datasets."""

import argparse
from pathlib import Path

from affect_geometry.logging import configure_file_logging, get_logger, progress
from affect_geometry.persistence.exp3_hub import hub_api, repo_path
from affect_geometry.persistence.valence_transfer import REPOSITORIES, plan_uploads, publish_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs/valence-transfer/pc1-v2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="Saved run directory")
    parser.add_argument("--model", choices=REPOSITORIES, help="Select one model (default: all)")
    parser.add_argument("--path-in-repo", help="Hub folder (default: valence-transfer/<run>)")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the upload plan")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    models = (args.model,) if args.model else tuple(REPOSITORIES)
    try:
        prefix = repo_path(args.path_in_repo or f"valence-transfer/{run_dir.name}")
        plans = plan_uploads(run_dir, models)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    for model, files in plans.items():
        print(f"{model} ({len(files)} files) → hf://datasets/{REPOSITORIES[model]}/{prefix}")
    if args.dry_run:
        return

    api = hub_api()
    for model in progress(plans, description="Upload valence-transfer results", unit="model"):
        configure_file_logging(
            PROJECT_ROOT / "logs" / f"{model}-valence-transfer-{run_dir.name}",
            "upload",
            model=model,
            repo_id=REPOSITORIES[model],
            path_in_repo=prefix,
        )
        try:
            commit = publish_model(api, model, prefix, plans[model])
        except Exception:
            get_logger().exception("upload.failed model=%s", model)
            raise
        get_logger().info("upload.completed model=%s revision=%s", model, commit.oid)
        print(f"https://huggingface.co/datasets/{REPOSITORIES[model]}/tree/{commit.oid}/{prefix}")


if __name__ == "__main__":
    main()
