"""Orchestrate paired batch-size runs using the canonical M03 pilot."""

import json
from dataclasses import replace

from affect_geometry.m03 import pipeline

from . import comparison, diagnostic, plotting


def run(
    base, run_id, *, bs1_run_id=None, asset_cache=None, compare_only=False, projection_batch_size=8
):
    if base.pilot_size != 64 or base.score_atol != 0.25:
        raise ValueError("The paired diagnostic requires the 64-comment pilot and 0.25 tolerance")
    bs1 = replace(base, prefix_batch_size=1, projection_batch_size=1)
    bs8 = replace(base, prefix_batch_size=8, projection_batch_size=projection_batch_size)
    root = pipeline.setup(base, run_id, "m03.batch_pilot")
    configs = [bs1, bs8]
    condition = "bs8" if projection_batch_size == 8 else f"prefix8-proj{projection_batch_size}"
    run_ids = [bs1_run_id or f"{run_id}-bs1", f"{run_id}-{condition}-diagnostic"]
    root1, root8 = [cfg.root(rid) for cfg, rid in zip(configs, run_ids, strict=True)]
    if len({root, root1, root8}) != 3:
        raise ValueError("Use distinct run IDs for the pair and both pilot conditions")

    if not compare_only:
        if bs1_run_id:
            pipeline.cached(root1, "pilot", "canonical", bs1)
            diagnostic.read_checks(root1, diagnostic=False)
        else:
            if asset_cache:
                _link_cache(root1, asset_cache)
            pipeline.prepare(bs1, run_ids[0])
            if not (root1 / "pilot_checks.json").is_file():
                pipeline.pilot(bs1, run_ids[0])
        _link_cache(root8, asset_cache or root1 / "assets" / "hub")
        pipeline.prepare(bs8, run_ids[1])
        comparison.validate_protocols(pipeline.protocol(root1, bs1), pipeline.protocol(root8, bs8))
        if not (root8 / "diagnostic_checks.json").is_file():
            diagnostic.run(
                bs8,
                run_ids[1],
                json.loads((root1 / "freeze.json").read_text()),
                json.loads((root1 / "scores" / "pilot" / "canonical" / "index.json").read_text()),
            )

    pipeline.setup(base, run_id, "m03.batch_pilot.compare")
    comparison.compare(configs, run_ids, root)
    plotting.plot_flips(root)
    return root


def _link_cache(root, assets):
    assets = assets.resolve(strict=True)
    cache = root / "assets" / "hub"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        if cache.resolve() != assets:
            raise ValueError(f"Asset cache differs from existing run: {cache}")
    else:
        cache.symlink_to(assets, target_is_directory=True)
