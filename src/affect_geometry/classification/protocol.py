"""Pure prompt and scoring rules; no dependency on the Choi–Weber implementation."""

import hashlib
import random

EMOTIONS = (
    "admiration amusement anger annoyance approval caring confusion curiosity desire "
    "disappointment disapproval disgust embarrassment excitement fear gratitude grief joy "
    "love nervousness optimism pride realization relief remorse sadness surprise neutral"
).split()
MODELS = {
    "llama2": ("meta-llama/Llama-2-7b-hf", False),
    "gemma2": ("google/gemma-2-9b", False),
    "apertus": ("swiss-ai/Apertus-8B-Instruct-2509", True),
    "olmo2": ("allenai/OLMo-2-1124-7B-Instruct", True),
}
PROTOCOL = "rq2_single_label_v1"


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def prompt(row, seed=42):
    order = EMOTIONS.copy()
    random.Random(digest(f"{seed}:{row['sample_id']}")).shuffle(order)
    text = (
        "Classify this text into exactly one emotion from this list:\n"
        + ", ".join(order)
        + "\nOutput only the emotion word.\n\n"
        + f"Text: {row['text']}\nEmotion:"
    )
    return text, order


def parse_strict(raw):
    value = raw.strip().lower().strip(".,!?*").strip()
    return value if value in EMOTIONS else None


def parse_compat(raw):
    """Independently implement the documented full-label/first-word diagnostic."""
    value = raw.strip("*").strip(".,!?\n\r\t *").lower().strip()
    if value in EMOTIONS:
        return value
    first = value.split()[0].strip(".,!?*") if value.split() else ""
    return first if first in EMOTIONS else None


def score(raw, gold):
    strict, compat = parse_strict(raw), parse_compat(raw)
    return dict(
        predicted_emotion=strict,
        predicted_label_id=EMOTIONS.index(strict) if strict else None,
        correct=strict == gold,
        valid_label=strict is not None,
        format_compliant=strict is not None,
        outcome=(
            "correct" if strict == gold else "incorrect_label" if strict else "invalid_output"
        ),
        compat_predicted_emotion=compat,
        compat_valid_label=compat is not None,
        compat_correct=compat == gold,
        parser_version="strict_label_v1",
        compat_parser_version="choi_weber_first_word_v1",
    )
