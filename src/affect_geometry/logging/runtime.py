from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from tqdm.auto import tqdm

from affect_geometry.config import ExperimentConfig

T = TypeVar("T")
LOGGER_NAME = "affect_geometry"
_HANDLER_MARKER = "affect_geometry_run_handler"


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def configure_run_logging(config: ExperimentConfig, command: str) -> Path:
    """Append this process's structured events to the model/run-specific log."""
    directory = config.log_dir
    directory.mkdir(parents=True, exist_ok=True)
    logger = get_logger()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            logger.removeHandler(handler)
            handler.close()
    handler = logging.FileHandler(directory / "run.log", mode="a", encoding="utf-8")
    setattr(handler, _HANDLER_MARKER, True)
    formatter = logging.Formatter(
        "%(asctime)sZ | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    tqdm.write(f"Run log: {directory / 'run.log'}")
    logger.info(
        "command.start command=%s model=%s run_id=%s output=%s",
        command,
        config.model.name,
        config.run_id,
        config.run_dir,
    )
    return directory


def progress(
    iterable: Iterable[T],
    *,
    description: str,
    total: int | None = None,
    unit: str = "item",
    leave: bool = True,
) -> Iterator[T]:
    """Display tqdm progress and record compact milestones in the active run log."""
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None
    logger = get_logger()
    logger.info("progress.start description=%r total=%s unit=%s", description, total, unit)
    completed = 0
    next_fraction = 0.1
    finished = False
    try:
        with tqdm(
            iterable,
            desc=description,
            total=total,
            unit=unit,
            dynamic_ncols=True,
            leave=leave,
        ) as bar:
            for item in bar:
                yield item
                completed += 1
                if total:
                    fraction = completed / total
                    if fraction >= next_fraction and completed < total:
                        logger.info(
                            "progress.update description=%r completed=%d total=%d percent=%d",
                            description,
                            completed,
                            total,
                            round(fraction * 100),
                        )
                        next_fraction += 0.1
        finished = True
    finally:
        logger.info(
            "progress.%s description=%r completed=%d total=%s",
            "complete" if finished else "stopped",
            description,
            completed,
            total,
        )
