#!/usr/bin/env python
"""Run LT-NeutralProjected MDS/Isomap geometry and separate NRC VAD evaluation."""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from affect_geometry.config import load_config  # noqa: E402
from affect_geometry.experiments.neutral_projected import (  # noqa: E402
    PROJECTED_STAGES,
    run_projected_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "lt_neutral_projected.json"
    )
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID"))
    parser.add_argument(
        "--source-run",
        type=Path,
        help="Reuse an existing last-token run; skip preparation/extraction",
    )
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument(
        "--vad-reference",
        type=Path,
        help="Prepared VAD CSV (default: analysis.vad_reference in the config)",
    )
    reference.add_argument(
        "--nrc-vad-lexicon",
        type=Path,
        help="Rebuild the run reference from an official v2.1 ZIP or English TSV",
    )
    parser.add_argument("--start-at", choices=PROJECTED_STAGES, default="prepare")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--show", action="store_true", help="Open the saved plot gallery in a browser"
    )
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    source_run = args.source_run.expanduser().resolve() if args.source_run else None
    lexicon = args.nrc_vad_lexicon.expanduser().resolve() if args.nrc_vad_lexicon else None
    reference_csv = args.vad_reference.expanduser().resolve() if args.vad_reference else None
    os.chdir(PROJECT_ROOT)
    config = load_config(config_path, run_id=args.run_id, generate_run_id=True)
    if not config.run_dir.resolve().is_relative_to(PROJECT_ROOT / "outputs"):
        parser.error("output_root must be under this project's outputs/ directory")
    if not config.log_dir.resolve().is_relative_to(PROJECT_ROOT / "logs"):
        parser.error("log_root must be under this project's logs/ directory")
    config = replace(
        config,
        analysis=replace(
            config.analysis,
            va_reference=str(config.run_dir / "references" / "goemotions_nrc_vad.csv"),
            vad_reference=str(reference_csv or config.analysis.vad_reference),
        ),
    )
    gallery = run_projected_experiment(
        config,
        source_run=source_run,
        local_lexicon=lexicon,
        start_at=args.start_at,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(f"Experiment complete. Plot gallery: {gallery.resolve()}")
        if args.show and not webbrowser.open(gallery.resolve().as_uri()):
            print("No browser available; open the gallery path above on a machine with a browser.")


if __name__ == "__main__":
    main()
