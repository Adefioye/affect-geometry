"""Run and compare the two M03 pilot conditions."""

import argparse
from pathlib import Path

from affect_geometry.m03.configuration import Config

from .pipeline import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m03/gemma2-all-label-free.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bs1-run-id", help="Use a completed and identity-validated bs1 pilot")
    parser.add_argument(
        "--projection-batch-size",
        type=int,
        default=8,
        help="Projection batch size for the prefix-8 condition",
    )
    parser.add_argument("--asset-cache", type=Path, help="Reuse an existing Hugging Face cache")
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    print(
        run(
            Config.load(args.config),
            args.run_id,
            bs1_run_id=args.bs1_run_id,
            asset_cache=args.asset_cache,
            compare_only=args.compare_only,
            projection_batch_size=args.projection_batch_size,
        )
    )


if __name__ == "__main__":
    main()
