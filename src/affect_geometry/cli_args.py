from __future__ import annotations

import argparse

from affect_geometry.config import ExperimentConfig, load_config


def config_from_argv(description: str, *, generate_run_id: bool = False) -> ExperimentConfig:
    """Parse the shared config option for a single-stage module."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/default.json", help="Experiment JSON file")
    parser.add_argument(
        "--run-id",
        help="Shared run ID, normally a UTC timestamp such as 20261018T153045Z",
    )
    args = parser.parse_args()
    try:
        return load_config(
            args.config,
            run_id=args.run_id,
            generate_run_id=generate_run_id,
        )
    except ValueError as error:
        parser.error(str(error))
        raise AssertionError("argparse.error does not return") from error
