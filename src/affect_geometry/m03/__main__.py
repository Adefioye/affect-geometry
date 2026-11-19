"""Thin CLI for independently resumable M03 stages."""

from __future__ import annotations

import argparse
import json

from affect_geometry.m03.configuration import Config


def parser():
    p = argparse.ArgumentParser(description="Frozen M03 label-free lens comparison")
    p.add_argument(
        "command",
        choices=(
            "prepare",
            "pilot",
            "extract",
            "calibrate",
            "analyze",
            "plot",
            "run",
            "smoke",
            "status",
            "export-scores",
        ),
    )
    p.add_argument("--config", default="configs/m03/gemma2.json")
    p.add_argument("--run-id", required=True)
    p.add_argument("--phase", choices=("reference", "test"), default="test")
    p.add_argument(
        "--variant",
        choices=("all", "canonical", "alternate_menu", "alternate_prompt"),
        default="all",
    )
    p.add_argument(
        "--export-scores",
        action="store_true",
        help="Also export the long complete-label Parquet table during analysis",
    )
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "smoke":
        from affect_geometry.m03.smoke import smoke

        print(smoke(args.run_id))
        return
    config = Config.load(args.config)
    root = config.root(args.run_id)
    from affect_geometry.m03 import pipeline

    if args.command == "prepare":
        pipeline.prepare(config, args.run_id)
    elif args.command == "pilot":
        pipeline.pilot(config, args.run_id)
    elif args.command == "extract":
        pipeline.extract(config, args.run_id, phase=args.phase, variant=args.variant)
    elif args.command == "calibrate":
        pipeline.calibrate(config, args.run_id)
    elif args.command == "analyze":
        from affect_geometry.m03.analysis import analyze

        analyze(config, args.run_id, export_complete_scores=args.export_scores)
    elif args.command == "plot":
        from affect_geometry.m03.plotting import plot

        pipeline.setup(config, args.run_id, "m03.plot")
        pipeline.protocol(root, config)
        plot(root)
    elif args.command == "export-scores":
        from affect_geometry.m03.analysis import export_scores

        pipeline.setup(config, args.run_id, "m03.export-scores")
        export_scores(root, config, "canonical" if args.variant == "all" else args.variant)
    elif args.command == "status":
        pipeline.protocol(root, config)

        def completed(name):
            path = root / name
            return path.exists() and json.loads(path.read_text()).get("complete", False)

        print(
            json.dumps(
                dict(
                    run=str(root),
                    pilot=completed("pilot_checks.json"),
                    scores=[
                        str(p.relative_to(root))
                        for p in sorted(root.glob("scores/*/*/complete.json"))
                    ],
                    analysis=completed("analysis_manifest.json"),
                ),
                indent=2,
            )
        )
    else:
        from affect_geometry.m03 import artifacts
        from affect_geometry.m03.analysis import analyze
        from affect_geometry.m03.plotting import plot

        pipeline.prepare(config, args.run_id)
        bundle = artifacts.load(config, root / "assets" / "hub")
        pipeline.pilot(config, args.run_id, bundle=bundle)
        if config.calibrated:
            pipeline.extract(config, args.run_id, phase="reference", bundle=bundle)
            pipeline.calibrate(config, args.run_id)
        pipeline.extract(config, args.run_id, bundle=bundle)
        analyze(config, args.run_id, export_complete_scores=True)
        plot(root)
    print(root)


if __name__ == "__main__":
    main()
