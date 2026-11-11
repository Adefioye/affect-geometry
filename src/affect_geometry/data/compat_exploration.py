"""Pinned, lightweight EXP3 classification tables (no activation arrays)."""

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from affect_geometry.classification.environment import load_hf_token
from affect_geometry.logging import progress

SOURCES = {
    "gemma2": ("3ae38a8e481a56b3b7c24025fb86cea44259ef2c", "rq3-activations-v1"),
    "olmo2": ("a581c7d352339b00789855ae1fe13d66f44120cd", "rq3-activations-bs8-v1"),
    "apertus": ("07517cb3cbb71a2cc6298cfdfaf74362805b5cb0", "rq3-activations-v1"),
}


def load_sources(destination):
    """Reuse local files; download missing files at explicit immutable revisions."""
    result = {}
    for model, (revision, run) in progress(SOURCES.items(), description="Load EXP3 tables"):
        repo = f"kokolamba/exp3-{model}"
        prefix = f"activations/{run}"
        root = Path(destination) / model
        files = [
            "manifest.json",
            "cohorts/all/samples.csv",
            "cohorts/support.csv",
            "cohorts/unresolved.csv",
        ]
        for relative in files:
            filename = f"{prefix}/{relative}"
            if not (root / filename).exists():
                hf_hub_download(
                    repo,
                    filename,
                    repo_type="dataset",
                    revision=revision,
                    token=load_hf_token(),
                    local_dir=root,
                )
        bundle = root / prefix
        provenance = dict(
            model=model,
            repo_id=repo,
            revision=revision,
            run_id=run,
            scope="balanced_activation_cohort",
            pooling="last_nonpadding_token",
        )
        result[model] = dict(
            rows=pd.read_csv(bundle / "cohorts/all/samples.csv", keep_default_na=False),
            support=pd.read_csv(bundle / "cohorts/support.csv"),
            unresolved=pd.read_csv(bundle / "cohorts/unresolved.csv"),
            manifest=json.loads((bundle / "manifest.json").read_text()),
            provenance=provenance,
        )
    return result
