from __future__ import annotations

import argparse
from collections.abc import Callable

from affect_geometry.config import ExperimentConfig, load_config
from affect_geometry.logging import configure_run_logging, get_logger


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


def run_command(
    description: str,
    callback: Callable[[ExperimentConfig], None],
    *,
    generate_run_id: bool = False,
) -> None:
    config = config_from_argv(description, generate_run_id=generate_run_id)
    log_directory = configure_run_logging(config, callback.__name__)
    logger = get_logger()
    try:
        callback(config)
    except BaseException:
        logger.exception("command.failed command=%s", callback.__name__)
        raise
    logger.info("command.complete command=%s logs=%s", callback.__name__, log_directory)
