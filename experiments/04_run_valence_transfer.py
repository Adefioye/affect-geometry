"""Select native-space PC1 directions, then evaluate frozen OOD patching."""

import argparse

from affect_geometry.experiments.valence_transfer import (
    evaluate_frozen,
    load_config,
    select_directions,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "evaluate"))
    parser.add_argument("--config", default="configs/valence_transfer.json")
    args = parser.parse_args()
    config = load_config(args.config)
    (select_directions if args.stage == "select" else evaluate_frozen)(config)


if __name__ == "__main__":
    main()
