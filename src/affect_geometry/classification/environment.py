"""Runtime environment loading for Hugging Face classification jobs."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .protocol import MODELS

GATED_MODEL_KEYS = frozenset({"llama2", "gemma2"})
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOTENV_PATH = PROJECT_ROOT / ".env"


def load_hf_token() -> str | None:
    """Load ``affect-geometry/.env`` without overriding an exported token."""
    if DOTENV_PATH.is_file():
        load_dotenv(DOTENV_PATH, override=False)
    return os.environ.get("HF_TOKEN") or None


def require_model_access(model_key: str, token: str | None) -> None:
    """Fail early when a gated model cannot be authenticated."""
    if model_key in GATED_MODEL_KEYS and not token:
        model_id = MODELS[model_key][0]
        raise RuntimeError(
            f"{model_id} is gated and requires HF_TOKEN. Accept the model's Hugging Face "
            "license with the same account, then set HF_TOKEN in the project .env file or "
            "export it in the RunPod environment."
        )
