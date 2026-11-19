"""Fixed-taxonomy metrics and paired duplicate-group bootstrap intervals."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix


def class_counts(gold: np.ndarray, prediction: np.ndarray, weights=None):
    weights = np.ones(len(gold)) if weights is None else weights
    support = np.bincount(gold, weights=weights, minlength=28)
    valid = prediction >= 0
    predicted = np.bincount(prediction[valid], weights=weights[valid], minlength=28)
    same = prediction == gold
    tp = np.bincount(gold[same], weights=weights[same], minlength=28)
    precision = np.divide(tp, predicted, out=np.zeros(28), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(28), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(28), where=(support + predicted) > 0)
    return dict(
        support=support, predicted=predicted, tp=tp, precision=precision, recall=recall, f1=f1
    )


def summarize(gold: np.ndarray, prediction: np.ndarray) -> dict:
    counts = class_counts(gold, prediction)
    return dict(
        n=len(gold),
        accuracy=float(np.mean(gold == prediction)),
        macro_f1=float(counts["f1"].mean()),
    )


def paired(gold, prediction, baseline) -> dict:
    correct, previous = prediction == gold, baseline == gold
    r, d = int((correct & ~previous).sum()), int((~correct & previous).sum())
    return dict(
        repairs=r,
        damages=d,
        repair_rate=r / int((~previous).sum()) if (~previous).any() else None,
        damage_rate=d / int(previous.sum()) if previous.any() else None,
        delta_accuracy_pp=100 * (r - d) / len(gold),
        wrong_to_different_wrong=int((~correct & ~previous & (prediction != baseline)).sum()),
        invalid_repairs=int((correct & (baseline < 0)).sum()),
    )


def neutral_transitions(gold, prediction, baseline) -> dict:
    """Count only legal emotion/neutral changes; invalid outputs are separate."""
    transitions = {
        "neutral_to_emotion": (baseline == 27) & (prediction >= 0) & (prediction != 27),
        "emotion_to_neutral": (baseline >= 0) & (baseline != 27) & (prediction == 27),
    }
    return {
        name: dict(
            n=int(mask.sum()),
            repairs=int((mask & (baseline != gold) & (prediction == gold)).sum()),
            damages=int((mask & (baseline == gold) & (prediction != gold)).sum()),
        )
        for name, mask in transitions.items()
    }


class GroupBootstrap:
    """The same sampled text groups and weights are shared by every arm."""

    def __init__(self, gold, groups, replicates: int, seed: int):
        self.gold = np.asarray(gold)
        unique, self.group = np.unique(groups, return_inverse=True)
        self.n_groups = len(unique)
        rng = np.random.default_rng(seed)
        self.weights = rng.multinomial(
            self.n_groups, np.full(self.n_groups, 1 / self.n_groups), size=replicates
        ).astype(np.float32)
        size = np.bincount(self.group, minlength=self.n_groups)
        self.n = self.weights @ size
        matrix = coo_matrix(
            (np.ones(len(gold)), (self.group, gold)), shape=(self.n_groups, 28)
        ).tocsr()
        self.support = self.weights @ matrix

    def distribution(self, prediction) -> dict[str, np.ndarray]:
        prediction = np.asarray(prediction)
        valid, same = prediction >= 0, prediction == self.gold
        rows = np.concatenate([self.group[valid], self.group[same]])
        cols = np.concatenate([prediction[valid], self.gold[same] + 28])
        matrix = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.n_groups, 56)).tocsr()
        counts = self.weights @ matrix
        predicted, tp = counts[:, :28], counts[:, 28:]
        denom = self.support + predicted
        f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
        return dict(accuracy=tp.sum(-1) / self.n, macro_f1=f1.mean(-1))


def intervals(distribution: dict, baselines: dict[str, dict]) -> dict:
    result = {}
    for metric, values in distribution.items():
        lo, hi = np.quantile(values, [0.025, 0.975])
        result.update({metric + "_ci_low": float(lo), metric + "_ci_high": float(hi)})
        for baseline, reference in baselines.items():
            lo, hi = np.quantile(100 * (values - reference[metric]), [0.025, 0.975])
            prefix = f"delta_{metric}_{baseline}_pp"
            result.update({prefix + "_ci_low": float(lo), prefix + "_ci_high": float(hi)})
    return result


def ranks_and_margins(scores: np.ndarray, gold: np.ndarray):
    order = np.argsort(-scores, axis=-1, kind="stable")
    ranks = np.argmax(order == gold[:, None], axis=-1) + 1
    rivals = scores.copy()
    rivals[np.arange(len(gold)), gold] = -np.inf
    margin = scores[np.arange(len(gold)), gold] - rivals.max(-1)
    return (
        ranks,
        margin,
        scores[np.arange(len(gold)), order[:, 0]] - (scores[np.arange(len(gold)), order[:, 1]]),
    )
