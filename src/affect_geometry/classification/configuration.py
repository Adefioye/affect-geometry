"""Single-model JSON configuration; explicit command-line flags override defaults."""

import json
from pathlib import Path

from .protocol import MODELS

ALLOWED = {
    "model",
    "run_dir",
    "model_revision",
    "dataset_revision",
    "seed",
    "pilot_limit",
    "dtype",
    "device",
}


def load_config(path):
    values = json.loads(Path(path).read_text())
    if not isinstance(values, dict) or set(values) - ALLOWED:
        raise ValueError("Config must be an object containing only supported run settings")
    if values.get("model") not in MODELS:
        raise ValueError("Config must select exactly one supported model")
    if not isinstance(values.get("run_dir"), str) or not values["run_dir"].strip():
        raise ValueError("Config requires a nonempty run_dir")
    for key in ("seed", "pilot_limit"):
        if key in values and (type(values[key]) is not int or values[key] < 0):
            raise ValueError(f"{key} must be a nonnegative integer")
    if values.get("dtype", "bfloat16") not in ("float32", "float16", "bfloat16"):
        raise ValueError("Unsupported dtype")
    for key in ("device", "model_revision", "dataset_revision"):
        if key in values and (not isinstance(values[key], str) or not values[key].strip()):
            raise ValueError(f"{key} must be a nonempty string")
    values["run_dir"] = Path(values["run_dir"])
    return values
