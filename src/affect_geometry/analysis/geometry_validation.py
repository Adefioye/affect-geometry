"""Transductive geometry, held-out ratings, and strictly nested setting selection."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr

from affect_geometry.embeddings.geometry import UnsupportedGeometry


@dataclass
class SimilarityTransform:
    source_mean: np.ndarray
    target_mean: np.ndarray
    rotation: np.ndarray
    scale: float
    rank: int

    def predict(self, values):
        return (values - self.source_mean) @ self.rotation * self.scale + self.target_mean

    def record(self):
        return dict(
            source_mean=self.source_mean.tolist(),
            target_mean=self.target_mean.tolist(),
            rotation=self.rotation.tolist(),
            scale=self.scale,
            rank=self.rank,
            reflection=bool(np.linalg.det(self.rotation) < 0),
        )


def fit_similarity(x, y, tolerance=1e-10):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise UnsupportedGeometry("invalid_alignment_inputs")
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - xm, y - ym
    rank = min(np.linalg.matrix_rank(xc, tol=tolerance), np.linalg.matrix_rank(yc, tol=tolerance))
    if rank < x.shape[1]:
        raise UnsupportedGeometry("insufficient_alignment_rank")
    u, singular, vt = np.linalg.svd(xc.T @ yc)
    return SimilarityTransform(xm, ym, u @ vt, float(singular.sum() / np.sum(xc**2)), int(rank))


def cv_predictions(x, y, indices=None):
    indices = np.arange(len(y)) if indices is None else np.asarray(indices)
    predictions, means, records = [], [], []
    for held in indices:
        train = indices[indices != held]
        transform = fit_similarity(x[train], y[train])
        predictions.append(transform.predict(x[held]))
        means.append(transform.target_mean)
        records.append(dict(held_out=int(held), training=train.tolist(), **transform.record()))
    return np.asarray(predictions), np.asarray(means), records


def pooled_r2(y, predictions, means):
    denominator = np.sum((y - means) ** 2)
    if not np.isfinite(denominator) or denominator <= 1e-10:
        raise UnsupportedGeometry("zero_target_denominator")
    return float(1 - np.sum((predictions - y) ** 2) / denominator)


def select_setting(candidates, y, train):
    scores, failures = {}, {}
    for setting, x in sorted(candidates.items()):
        try:
            predictions, means, _ = cv_predictions(x, y, train)
            scores[setting] = pooled_r2(y[train], predictions, means)
        except UnsupportedGeometry as exc:
            failures[setting] = str(exc)
    if not scores:
        raise UnsupportedGeometry("no_valid_inner_candidate")
    best = max(sorted(scores), key=scores.get)
    return best, scores, failures


def evaluate(candidates, y, *, selection="nested"):
    if not candidates:
        raise UnsupportedGeometry("no_supported_candidates")
    if selection not in ("nested", "fixed"):
        raise ValueError("Selection must be nested or fixed")
    if selection == "fixed" and len(candidates) != 1:
        raise ValueError("Fixed selection requires exactly one candidate")
    n = len(y)
    predictions, means, folds = [], [], []
    for held in range(n):
        train = np.delete(np.arange(n), held)
        if selection == "nested":
            chosen, scores, failures = select_setting(candidates, y, train)
        else:
            chosen, scores, failures = next(iter(candidates)), {}, {}
        transform = fit_similarity(candidates[chosen][train], y[train])
        prediction = transform.predict(candidates[chosen][held])
        predictions.append(prediction)
        means.append(transform.target_mean)
        folds.append(
            dict(
                held_out=held,
                training=train.tolist(),
                setting=chosen,
                inner_scores=scores,
                inner_failures=failures,
                prediction=prediction.tolist(),
                **transform.record(),
            )
        )
    predictions, means = np.asarray(predictions), np.asarray(means)
    score = pooled_r2(y, predictions, means)
    # Descriptive checkpoint settings are selected independently from outer held-out predictions.
    chosen = (
        select_setting(candidates, y, np.arange(n))[0]
        if selection == "nested"
        else next(iter(candidates))
    )
    transform = fit_similarity(candidates[chosen], y)
    aligned = transform.predict(candidates[chosen])
    return dict(
        cv_r2=score,
        full_r2=pooled_r2(y, aligned, np.broadcast_to(y.mean(0), y.shape)),
        pearson=[
            float(pearsonr(y[:, j], predictions[:, j]).statistic)
            if np.std(predictions[:, j]) > 1e-10
            else float("nan")
            for j in range(y.shape[1])
        ],
        predictions=predictions,
        training_means=means,
        errors=np.sum((predictions - y) ** 2, axis=1),
        folds=folds,
        descriptive_setting=chosen,
        full_transform=transform.record(),
        aligned=aligned,
    )


def evaluate_layers(layer_candidates, y, layers):
    """Secondary held-out evaluation selecting both layer and setting in inner folds.

    No peak from the outer scores enters selection. Equal inner scores choose the
    earliest layer, then the lexicographically first setting. All declared layers
    must have supported candidates. This secondary estimate has no permutation p-value.
    """
    if any(not layer_candidates.get(layer) for layer in layers):
        raise UnsupportedGeometry("incomplete_layer_selection_support")
    candidates, lookup = {}, {}
    for layer in sorted(layers):
        for setting, coordinates in sorted(layer_candidates[layer].items()):
            key = f"layer_{layer:06d}/{setting}"
            candidates[key] = coordinates
            lookup[key] = dict(layer=layer, setting=setting)
    result = evaluate(candidates, y, selection="nested")
    for fold in result["folds"]:
        selected = lookup[fold["setting"]]
        fold["selected_layer"] = selected["layer"]
        fold["selected_setting"] = selected["setting"]
    result["candidate_lookup"] = lookup
    return result
