"""Explicit RQ3 panel and frozen numerical policies."""

import json
import math
from pathlib import Path

from affect_geometry.classification.protocol import MODELS

PANEL = ("gemma2", "apertus", "olmo2")


def load_config(path):
    config = json.loads(Path(path).read_text())
    required = {
        "run_id",
        "models",
        "nrc_path",
        "nrc_provenance",
        "seed",
        "permutation_seed",
        "permutations",
        "smoke",
        "layers",
        "isomap_neighbors",
        "diffusion_neighbors",
        "diffusion_scale_ranks",
        "diffusion_times",
        "diffusion_alphas",
        "selection",
        "neighborhoods",
        "filtration_bound",
        "betti_grid_size",
        "tolerance",
        "settings_frozen",
        "pilot_verified",
        "emergence",
        "extraction",
        "budgets",
    }
    if set(config) != required:
        raise ValueError(
            f"Configuration keys differ: missing={required - set(config)}, "
            f"extra={set(config) - required}"
        )
    if set(config["models"]) != set(PANEL):
        raise ValueError("RQ3 requires exactly Gemma2, Apertus and OLMo2")
    for key in ("smoke", "settings_frozen", "pilot_verified"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in ("seed", "permutation_seed", "permutations", "betti_grid_size"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if config["layers"] is not None and (
        not isinstance(config["layers"], list)
        or not config["layers"]
        or any(type(layer) is not int or layer < 0 for layer in config["layers"])
    ):
        raise ValueError("Pilot layers must be explicit nonnegative block indices")
    if (
        not config["run_id"]
        or Path(config["run_id"]).name != config["run_id"]
        or config["run_id"] in (".", "..")
    ):
        raise ValueError("run_id must be a single path component")
    if not config["smoke"] and (config["permutations"] != 2000 or config["layers"] is not None):
        raise ValueError("Full runs require 2,000 permutations and every block layer")
    if not config["smoke"] and not (config["settings_frozen"] and config["pilot_verified"]):
        raise ValueError("Freeze settings and verify pilot budgets before a full run")
    if config["selection"] != "nested":
        raise ValueError("This configuration uses nested leave-one-emotion-out selection")
    if config["permutations"] < 1 or config["tolerance"] <= 0:
        raise ValueError("Invalid permutation count or tolerance")
    if (
        not math.isfinite(config["tolerance"])
        or not math.isfinite(config["filtration_bound"])
        or config["filtration_bound"] <= 0
        or config["betti_grid_size"] < 2
    ):
        raise ValueError("Invalid numeric tolerance or filtration policy")
    if set(config["budgets"]) != {"max_seconds", "max_rss_gib"} or any(
        not math.isfinite(value) or value <= 0 for value in config["budgets"].values()
    ):
        raise ValueError("Positive runtime and memory budgets are required")
    for key in (
        "isomap_neighbors",
        "diffusion_neighbors",
        "diffusion_scale_ranks",
        "diffusion_times",
        "neighborhoods",
    ):
        if not config[key] or any(type(x) is not int or x < 1 for x in config[key]):
            raise ValueError(f"Invalid grid: {key}")
    if (
        not set(config["diffusion_alphas"]) <= {0.0, 0.5, 1.0}
        or 1.0 not in config["diffusion_alphas"]
    ):
        raise ValueError("Alpha=1 is primary; 0 and .5 are optional sensitivities")
    for slug, spec in config["models"].items():
        if (
            set(spec) != {"model_id", "classification_run", "activation_run"}
            or spec["model_id"] != MODELS[slug][0]
        ):
            raise ValueError(f"Invalid model specification: {slug}")
    return config
