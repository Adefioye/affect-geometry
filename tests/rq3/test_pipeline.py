import json

import numpy as np
import pandas as pd
from test_scientific_contracts import source_rows

from affect_geometry.analysis.cohorts import balanced_cohorts
from affect_geometry.classification.protocol import EMOTIONS
from affect_geometry.experiments.correctness_geometry import run_geometry
from affect_geometry.experiments.rq3_config import load_config
from affect_geometry.persistence.correctness_geometry import checksum


def test_end_to_end_nine_cohorts_three_methods_replay(tmp_path):
    from pathlib import Path

    project = Path(__file__).resolve().parents[2]
    config = load_config(project / "configs/rq3_geometry.json")
    config.update(
        layers=[0, 1],
        isomap_neighbors=[3],
        diffusion_neighbors=[3],
        diffusion_scale_ranks=[2],
        diffusion_times=[1],
        neighborhoods=[1],
        permutations=1,
    )
    rows = source_rows()
    labels = balanced_cohorts(rows, EMOTIONS, 42).labels
    rng = np.random.default_rng(42)
    centers = {label: rng.normal(size=10) for label in labels}
    values = np.stack([centers[label] + rng.normal(scale=0.4, size=10) for label in rows.emotion])
    arrays = [values, values + 0.05 * rng.normal(size=values.shape)]
    target = rng.normal(size=(len(labels), 2))
    roots = []
    for slug in config["models"]:
        root = tmp_path / slug
        run_geometry(
            config,
            slug,
            rows,
            arrays,
            {"fixture": True},
            {"full_population": False},
            {"fixture": True},
            target,
            {"fixture": True},
            root,
        )
        roots.append(root)
        scores = pd.read_csv(root / "metrics/scores.csv")
        assert len(scores) == 3 * 3 * 2 * 2
        assert scores.status.eq("ok").all(), scores[["method", "reason"]].to_string()
        assert scores.corrected_p.isna().all()
        assert len(pd.read_csv(root / "metrics/cka.csv")) == 18 * 3
        assert (root / "inference/1d/nulls.npz").exists()
        assert (root / "plots/cka-all-2d.png").exists()
        assert len(pd.read_csv(root / "metrics/aggregates.csv")) == 18
        before = checksum(root / "metrics/scores.csv")
        run_geometry(
            config,
            slug,
            rows,
            arrays,
            {"fixture": True},
            {"full_population": False},
            {"fixture": True},
            target,
            {"fixture": True},
            root,
        )
        assert checksum(root / "metrics/scores.csv") == before
    from affect_geometry.analysis.cross_model_geometry import compare_common_support
    from affect_geometry.reporting.correctness_geometry import compare_models

    compare_common_support(roots, tmp_path / "comparisons")
    compare_models(roots, tmp_path / "comparisons")
    manifest = json.loads((tmp_path / "comparisons/manifest.json").read_text())
    assert len(manifest["common_emotions"]) == 6
    common = pd.read_csv(tmp_path / "comparisons/common_support/aggregates.csv")
    assert len(common) == 54 and common.status.eq("ok").all()


def test_unsupported_geometry_keeps_all_cells_and_invalidates_families(tmp_path):
    from pathlib import Path

    project = Path(__file__).resolve().parents[2]
    config = load_config(project / "configs/rq3_geometry.json")
    config.update(
        layers=[0],
        isomap_neighbors=[3],
        diffusion_neighbors=[3],
        diffusion_scale_ranks=[2],
        diffusion_times=[1],
        neighborhoods=[1],
        permutations=1,
    )
    rows = source_rows()
    labels = balanced_cohorts(rows, EMOTIONS, 42).labels
    target = np.random.default_rng(8).normal(size=(len(labels), 2))
    run_geometry(
        config,
        "gemma2",
        rows,
        [np.zeros((len(rows), 4))],
        {"fixture": True},
        {"full_population": False},
        {"fixture": True},
        target,
        {"fixture": True},
        tmp_path,
    )
    scores = pd.read_csv(tmp_path / "metrics/scores.csv")
    aggregates = pd.read_csv(tmp_path / "metrics/aggregates.csv")
    assert len(scores) == 18 and scores.status.eq("unsupported").all()
    assert scores.inference_status.eq("unsupported").all()
    assert scores.cv_r2.isna().all() and aggregates.A_M.isna().all()
    assert (
        len(json.loads((tmp_path / "inference/2d/status.json").read_text())["expected_members"])
        == 9
    )
    assert (tmp_path / "plots/cka-all-2d.png").exists()
