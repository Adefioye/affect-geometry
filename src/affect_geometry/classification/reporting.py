"""Reports derived only from persisted sample outcomes."""

from pathlib import Path

from .protocol import EMOTIONS


def rate(a, b):
    return 100 * a / b if b else None


def summarize(rows, split, emotion):
    completed = [r for r in rows if r["status"] == "completed"]
    n = len(completed)
    correct = sum(r["correct"] for r in completed)
    valid = sum(r["valid_label"] for r in completed)
    compat = sum(r["compat_correct"] for r in completed)
    # Wilson interval for a binomial proportion (descriptive, not a paired-model interval).
    if n:
        z = 1.959963984540054
        p = correct / n
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return dict(
        split=split,
        emotion=emotion,
        n_total=len(rows),
        n_completed=n,
        n_inference_errors=len(rows) - n,
        n_correct=correct,
        n_incorrect=n - correct,
        n_valid=valid,
        n_invalid=n - valid,
        n_wrong_valid_label=valid - correct,
        coverage_pct=rate(n, len(rows)),
        correct_pct=rate(correct, n),
        incorrect_pct=rate(n - correct, n),
        invalid_pct=rate(n - valid, n),
        valid_label_pct=rate(valid, n),
        valid_only_accuracy_pct=rate(correct, valid),
        compat_correct_pct=rate(compat, n),
        n_compat_correct=compat,
        n_strict_invalid_compat_correct=sum(
            not r["valid_label"] and r["compat_correct"] for r in completed
        ),
        correct_ci_low_pct=100 * (center - half) if n else None,
        correct_ci_high_pct=100 * (center + half) if n else None,
    )


def report(root, rows, model_id, run_id):
    import pandas as pd

    root = Path(root)
    for directory in ("analyses", "plot_data", "plots"):
        (root / directory).mkdir(exist_ok=True)
    summaries, confusion, scores = [], [], []
    for split in ("all", "train", "validation", "test"):
        subset = [r for r in rows if split == "all" or r["source_split"] == split]
        for emotion in ["all", *EMOTIONS]:
            group = [r for r in subset if emotion == "all" or r["emotion"] == emotion]
            summaries.append(summarize(group, split, emotion))
        completed = [r for r in subset if r["status"] == "completed"]
        for gold in EMOTIONS:
            support = sum(r["emotion"] == gold for r in completed)
            tp = sum(r["emotion"] == gold and r["predicted_emotion"] == gold for r in completed)
            pred = sum(r["predicted_emotion"] == gold for r in completed)
            precision, recall = tp / pred if pred else 0, tp / support if support else 0
            scores.append(
                dict(
                    split=split,
                    emotion=gold,
                    support=support,
                    precision=precision,
                    recall=recall,
                    f1=2 * precision * recall / (precision + recall) if precision + recall else 0,
                )
            )
            for label in [*EMOTIONS, "INVALID"]:
                count = sum(
                    r["emotion"] == gold and (r["predicted_emotion"] or "INVALID") == label
                    for r in completed
                )
                confusion.append(
                    dict(
                        split=split,
                        emotion=gold,
                        prediction=label,
                        count=count,
                        support=support,
                        pct=rate(count, support),
                    )
                )
    frames = {}
    for name, records in [
        ("summary", summaries),
        ("confusion", confusion),
        ("label_metrics", scores),
    ]:
        frame = pd.DataFrame(records)
        frame.insert(0, "run_id", run_id)
        frame.insert(0, "model_id", model_id)
        frame.to_csv(root / "analyses" / f"{name}.csv", index=False)
        frame.to_csv(root / "plot_data" / f"{name}.csv", index=False)
        frames[name] = frame
    macro = frames["label_metrics"].groupby("split")[["precision", "recall", "f1"]].mean()
    macro.to_csv(root / "analyses" / "macro_metrics.csv")
    pd.DataFrame(
        [
            {
                k: r.get(k)
                for k in [
                    "sample_id",
                    "source_split",
                    "emotion",
                    "correct",
                    "outcome",
                    "predicted_emotion",
                    "raw_response",
                    "text",
                ]
            }
            for r in rows
        ]
    ).to_csv(root / "analyses" / "sample_outcomes.csv", index=False)
    # Plot only persisted data, never an independent calculation.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved = pd.read_csv(root / "plot_data" / "summary.csv")
    view = saved[(saved.split == "all") & (saved.emotion != "all")]
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(view.emotion, view.correct_pct, label="Correct")
    ax.barh(view.emotion, view.incorrect_pct, left=view.correct_pct, label="Incorrect")
    ax.set(xlabel="Completed samples (%)", title=f"{model_id} — {run_id}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "plots" / "correctness.png")
    plt.close(fig)
    from .storage import atomic_json

    indexes = []
    for split in ("train", "validation", "test"):
        for emotion in EMOTIONS:
            for correct in (True, False):
                indexes.append(
                    dict(
                        split=split,
                        emotion=emotion,
                        correct=correct,
                        sample_ids=[
                            r["sample_id"]
                            for r in rows
                            if r["source_split"] == split
                            and r["emotion"] == emotion
                            and r.get("correct") is correct
                        ],
                    )
                )
    atomic_json(root / "analyses" / "sample_indexes.json", indexes)
    for field, filename, xlabel in [
        ("n_correct", "correct_counts.png", "Correct samples"),
        ("n_incorrect", "incorrect_counts.png", "Incorrect samples"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.barh(view.emotion, view[field])
        ax.set(xlabel=xlabel, title=f"{model_id} — {run_id}")
        fig.tight_layout()
        fig.savefig(root / "plots" / filename)
        plt.close(fig)
    matrix = pd.read_csv(root / "plot_data" / "confusion.csv")
    matrix = matrix[matrix.split == "all"].pivot(
        index="emotion", columns="prediction", values="pct"
    )
    matrix = matrix.reindex(index=EMOTIONS, columns=[*EMOTIONS, "INVALID"])
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(matrix.to_numpy(), vmin=0, vmax=100, cmap="Blues")
    ax.set_xticks(range(29), [*EMOTIONS, "INVALID"], rotation=90)
    ax.set_yticks(range(28), EMOTIONS)
    ax.set(xlabel="Predicted label", ylabel="Gold emotion", title=f"{model_id} — {run_id}")
    fig.colorbar(im, ax=ax, label="Gold-class completed samples (%)")
    fig.tight_layout()
    fig.savefig(root / "plots" / "confusion.png")
    plt.close(fig)


def compare(run_dirs, destination, seed=42, bootstrap_samples=1000):
    """Paired comparisons on identical examples; resample identical texts together."""
    import json
    from itertools import combinations

    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    sources = []
    for directory in map(Path, run_dirs):
        manifest = json.loads((directory / "manifest.json").read_text())
        rows = [
            r
            for split in ("train", "validation", "test")
            for r in pq.read_table(directory / "dataset" / f"{split}.parquet").to_pylist()
        ]
        if any(r["status"] != "completed" for r in rows):
            raise ValueError("Cross-model comparisons require completed runs")
        sources.append((manifest, {r["sample_id"]: r for r in rows}))
    summaries = []
    rng = np.random.default_rng(seed)
    for (ma, a), (mb, b) in combinations(sources, 2):
        if set(a) != set(b):
            raise ValueError("Models have different source populations")
        for key in a:
            if any(a[key][k] != b[key][k] for k in ("text", "emotion", "source_split")):
                raise ValueError("Source content differs across models")
        for emotion in ["all", *EMOTIONS]:
            keys = [k for k in a if emotion == "all" or a[k]["emotion"] == emotion]
            groups = {}
            for k in keys:
                groups.setdefault(a[k].get("text_hash", a[k]["text"]), []).append(
                    int(a[k]["correct"]) - int(b[k]["correct"])
                )
            values = list(groups.values())
            if values:
                totals = np.array([sum(x) for x in values])
                sizes = np.array([len(x) for x in values])
                boots = []
                for _ in range(bootstrap_samples):
                    draw = rng.integers(0, len(values), len(values))
                    boots.append(100 * totals[draw].sum() / sizes[draw].sum())
                low, high = np.quantile(boots, [0.025, 0.975])
                delta = 100 * totals.sum() / sizes.sum()
            else:
                low = high = delta = None
            summaries.append(
                dict(
                    model_a=ma["model_id"],
                    model_b=mb["model_id"],
                    emotion=emotion,
                    n=len(keys),
                    delta_correct_percentage_points=delta,
                    ci_low=low,
                    ci_high=high,
                    seed=seed,
                    bootstrap_samples=bootstrap_samples,
                )
            )
    pd.DataFrame(summaries).to_csv(destination / "paired_model_comparisons.csv", index=False)
    frames = [pd.read_csv(Path(d) / "analyses" / "summary.csv") for d in run_dirs]
    pd.concat(frames, ignore_index=True).to_csv(
        destination / "all_model_summaries.csv", index=False
    )
