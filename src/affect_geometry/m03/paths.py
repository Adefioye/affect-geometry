"""Canonical complete labels with immutable prompt-token prefixes."""

from __future__ import annotations

from dataclasses import dataclass

from affect_geometry.classification.protocol import EMOTIONS


@dataclass(frozen=True)
class CandidatePaths:
    prompt_ids: tuple[int, ...]
    labels: tuple[str, ...]
    paths: tuple[tuple[int, ...], ...]

    @property
    def nodes(self) -> list[tuple[int, ...]]:
        return sorted(
            {path[:i] for path in self.paths for i in range(len(path))}, key=lambda x: (len(x), x)
        )

    def edges(self) -> list[tuple[tuple[int, ...], int, int, bool]]:
        return [
            (path[:i], token, c, i == len(path) - 1)
            for c, path in enumerate(self.paths)
            for i, token in enumerate(path)
        ]


def canonical_paths(
    tokenizer, prompt: str, *, label_prefix: str, stop_token_id: int, max_new_tokens: int
) -> CandidatePaths:
    ids = tuple(tokenizer.encode(prompt, add_special_tokens=True))
    if not ids:
        raise ValueError("Empty prompt")
    if stop_token_id not in tokenizer.all_special_ids:
        raise ValueError("The declared stop token must be a tokenizer special token")
    paths = []
    for label in EMOTIONS:
        label_ids = tuple(tokenizer.encode(label_prefix + label, add_special_tokens=False))
        if not label_ids or any(t in tokenizer.all_special_ids for t in label_ids):
            raise ValueError(f"Invalid label tokenization: {label}")
        decoded = tokenizer.decode(label_ids, skip_special_tokens=False)
        if decoded.strip() != label:
            raise ValueError(f"Canonical path decodes incorrectly: {label}: {decoded!r}")
        # Never retokenize the joined prompt and continuation.
        full = tokenizer.decode(ids + label_ids, skip_special_tokens=True)
        if not full.rstrip().endswith(label):
            raise ValueError(f"Prompt boundary corrupts label {label}")
        path = label_ids + (stop_token_id,)
        if len(path) > max_new_tokens:
            raise ValueError(f"Canonical label exceeds output budget: {label}")
        paths.append(path)
    if len(set(paths)) != len(EMOTIONS):
        raise ValueError("Canonical paths are not unique")
    return CandidatePaths(ids, tuple(EMOTIONS), tuple(paths))
