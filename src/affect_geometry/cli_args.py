from __future__ import annotations

import argparse

from affect_geometry.config import ExperimentConfig, load_config


def config_from_argv(description: str) -> ExperimentConfig:
    """Parse the shared config option for a single-stage module."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/default.json", help="Experiment JSON file")
    return load_config(parser.parse_args().config)
