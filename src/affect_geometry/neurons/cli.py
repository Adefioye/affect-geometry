"""Thin EXP5 stage entry point; scientific logic lives in domain modules."""

import argparse
from pathlib import Path

from affect_geometry.experiments.neuron_transfer import STAGES, NeuronTransferExperiment
from affect_geometry.experiments.neuron_transfer_config import NeuronTransferConfig
from affect_geometry.logging import configure_file_logging, get_logger


def main(argv=None):
    parser = argparse.ArgumentParser(description="EXP5 oracle-guided native neuron transfer")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    args = parser.parse_args(argv)
    config = NeuronTransferConfig.read(args.config)
    configure_file_logging(
        Path(config.log_root) / f"{config.model_id.replace('/', '--')}__{config.run_id}",
        f"exp5-{args.stage}",
        model=config.model_id,
        run_id=config.run_id,
    )
    try:
        root = NeuronTransferExperiment(config).run(args.stage)
    except Exception:
        get_logger().exception("exp5.failed stage=%s", args.stage)
        raise
    print(f"EXP5 {args.stage}: {root}")


if __name__ == "__main__":
    main()
