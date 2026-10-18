from __future__ import annotations

import argparse
from collections.abc import Callable

from affect_geometry.config import ExperimentConfig, load_config
from affect_geometry.pipeline import analyze, compute_rdms, embed, extract, plot, prepare, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Investigate layerwise affect geometry")
    parser.add_argument(
        "stage",
        choices=["prepare", "extract", "rdm", "embed", "analyze", "plot", "run"],
        help="Pipeline stage to execute",
    )
    parser.add_argument("--config", default="configs/default.json", help="Experiment JSON file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config: ExperimentConfig = load_config(args.config)
    stages: dict[str, Callable[[ExperimentConfig], None]] = {
        "prepare": prepare,
        "extract": extract,
        "rdm": compute_rdms,
        "embed": embed,
        "analyze": analyze,
        "plot": plot,
        "run": run,
    }
    stages[args.stage](config)


if __name__ == "__main__":
    main()
