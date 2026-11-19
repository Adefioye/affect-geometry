"""Annotation-free prediction rules, applied only to cached complete-label scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

BETA_GRID = (0.0, 0.25, 0.5, 1.0)
DYNAMIC_DEPTHS = (0.65, 0.75, 0.85, 0.95)
LENSES = ("logit", "tuned", "jacobian")


def log_prob(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Complete-label scores must all be finite")
    return scores - logsumexp(scores, axis=-1, keepdims=True)


def nearest(layers: list[int], depth: float, n_blocks: int) -> int:
    if not layers:
        raise ValueError("No common nonfinal source layer")
    return min(layers, key=lambda layer: (abs((layer + 1) / n_blocks - depth), layer))


def layer_plan(coverage: dict[str, list[int]], n_blocks: int) -> dict:
    if set(coverage) != set(LENSES):
        raise ValueError("All three lenses are required")
    for lens, layers in coverage.items():
        if not layers or layers != sorted(set(layers)):
            raise ValueError(f"Invalid source coverage for {lens}")
        if layers[0] < 0 or layers[-1] >= n_blocks:
            raise ValueError("Source layer outside backbone")
    common = sorted(set.intersection(*(set(coverage[r]) for r in LENSES)))
    nonfinal = [layer for layer in common if layer < n_blocks - 1]
    return dict(
        coverage=coverage,
        n_blocks=n_blocks,
        common_layers=common,
        headline_layer=nearest(nonfinal, 0.75, n_blocks),
        dynamic_layers=sorted({nearest(nonfinal, d, n_blocks) for d in DYNAMIC_DEPTHS}),
        upper_layer=max(nonfinal),
        beta_grid=list(BETA_GRID),
        missing_layers={r: sorted(set(range(n_blocks)) - set(v)) for r, v in coverage.items()},
    )


def contrast(final: np.ndarray, intermediate: np.ndarray, beta: float) -> np.ndarray:
    f, e = log_prob(final), log_prob(intermediate)
    return f + beta * (f - e)


def dynamic(final: np.ndarray, panel: dict[int, np.ndarray], n_blocks: int):
    layers = sorted(panel)
    f = np.exp(log_prob(final))
    p = np.stack([np.exp(log_prob(panel[layer])) for layer in layers], axis=1)
    cosine = np.einsum("nc,nlc->nl", f, p) / (
        np.linalg.norm(f, axis=-1)[:, None] * np.linalg.norm(p, axis=-1)
    )
    # Stable argmax: sorted lower index wins a distance tie.
    selected = np.argmax(1 - np.clip(cosine, -1, 1), axis=1)
    chosen = np.asarray(layers)[selected]
    beta = 2.0 / (n_blocks - 1 - chosen)
    e = np.stack([panel[layer] for layer in layers], axis=1)[np.arange(len(f)), selected]
    return contrast(final, e, beta[:, None]), chosen, beta


@dataclass
class Arm:
    name: str
    lens: str
    kind: str
    layer: int
    beta: float
    scores: np.ndarray
    headline: bool = False
    selected_layers: np.ndarray | None = None
    selected_beta: np.ndarray | None = None


def arms(scores: dict[tuple[str, int], np.ndarray], plan: dict) -> list[Arm]:
    final_layer = plan["n_blocks"] - 1
    f = scores[("native", final_layer)]
    result = [Arm("native", "native", "baseline", final_layer, 0, f, True)]
    for lens in LENSES:
        for layer in plan["coverage"][lens]:
            e = scores[(lens, layer)]
            result.append(
                Arm(
                    f"{lens}.direct.l{layer}",
                    lens,
                    "direct",
                    layer,
                    0,
                    e,
                    layer == plan["headline_layer"],
                )
            )
            if layer == final_layer:
                continue
            for beta in BETA_GRID:
                result.append(
                    Arm(
                        f"{lens}.contrast.l{layer}.b{beta:g}",
                        lens,
                        "contrast",
                        layer,
                        beta,
                        contrast(f, e, beta),
                        layer == plan["headline_layer"] and beta == 0.5,
                    )
                )
            if layer < plan["upper_layer"]:
                upper = scores[(lens, plan["upper_layer"])]
                result.append(
                    Arm(
                        f"{lens}.within.l{layer}",
                        lens,
                        "within",
                        layer,
                        0.5,
                        contrast(upper, e, 0.5),
                    )
                )
        main = contrast(f, scores[(lens, plan["headline_layer"])], 0.5)
        mask = log_prob(f) >= np.max(log_prob(f), axis=-1, keepdims=True) + np.log(0.1)
        result.append(
            Arm(
                f"{lens}.masked",
                lens,
                "masked",
                plan["headline_layer"],
                0.5,
                np.where(mask, main, -np.inf),
            )
        )
        values, selected, beta = dynamic(
            f,
            {layer: scores[(lens, layer)] for layer in plan["dynamic_layers"]},
            plan["n_blocks"],
        )
        result.append(
            Arm(
                f"{lens}.dynamic",
                lens,
                "dynamic",
                -1,
                -1,
                values,
                selected_layers=selected,
                selected_beta=beta,
            )
        )
    return result
