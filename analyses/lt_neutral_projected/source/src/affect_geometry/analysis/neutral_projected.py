"""Exploratory geometry and separate three-axis reference recovery."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from affect_geometry.analysis.va import align_similarity
from affect_geometry.config import ExperimentConfig
from affect_geometry.logging import progress

# Published GoEmotions taxonomy; this is independent of NRC numerical ratings.
POSITIVE = frozenset(
    {
        "admiration",
        "amusement",
        "approval",
        "caring",
        "desire",
        "excitement",
        "gratitude",
        "joy",
        "love",
        "optimism",
        "pride",
        "relief",
    }
)
NEGATIVE = frozenset(
    {
        "anger",
        "annoyance",
        "disappointment",
        "disapproval",
        "disgust",
        "embarrassment",
        "fear",
        "grief",
        "nervousness",
        "remorse",
        "sadness",
    }
)


def emotion_group(label: str) -> str:
    if label == "neutral":
        return "neutral"
    if label in POSITIVE:
        return "positive"
    if label in NEGATIVE:
        return "negative"
    return "ambiguous"


def neutral_relative_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    """Translate each embedding to neutral, preserving its unaligned orientation."""
    keys = ["pooling", "layer", "distance", "method", "dimensions"]
    columns = [column for column in frame if column.startswith("coordinate_")]
    required = [*keys, "emotion", *columns]
    if "neighbors_used" in frame:
        required.append("neighbors_used")
    result = frame[required].copy()
    result["emotion_group"] = result.emotion.map(emotion_group)
    result["quadrant"] = "not_applicable"
    for _, group in result.groupby(keys):
        neutral = group.loc[group.emotion == "neutral"]
        if len(neutral) != 1:
            raise ValueError("Every exploratory embedding must contain exactly one neutral node")
        dimensions = int(group.dimensions.iloc[0])
        for axis in range(1, dimensions + 1):
            column = f"coordinate_{axis}"
            result.loc[group.index, f"neutral_original_{axis}"] = neutral[column].iloc[0]
            result.loc[group.index, column] = group[column] - neutral[column].iloc[0]
        if dimensions == 2:
            x = result.loc[group.index, "coordinate_1"].to_numpy()
            y = result.loc[group.index, "coordinate_2"].to_numpy()
            quadrant = np.select(
                [(x > 0) & (y > 0), (x < 0) & (y > 0), (x < 0) & (y < 0), (x > 0) & (y < 0)],
                ["I", "II", "III", "IV"],
                default="axis",
            )
            result.loc[group.index, "quadrant"] = quadrant
            result.loc[neutral.index, "quadrant"] = "neutral"
    return result


def analyze_projected(config: ExperimentConfig) -> None:
    """Write self-contained plot tables; keep reference targets out of exploration."""
    from affect_geometry.pipeline import analyze

    analyze(config)
    root = config.run_dir
    reference_path = Path(config.analysis.va_reference)
    context = {
        "model": config.model.name,
        "run_id": config.run_id,
        "seed": config.seed,
        "split": config.rdm.split,
        "variant": "LT-NeutralProjected",
        "neutral_variance_threshold": config.representation.neutral_variance_threshold,
        "post_projection_pca": False,
        "rdm_pca_components_used": 0,
    }
    # Add the experimental context to the standard summaries as well as the new plots.
    for path in (root / "plot_data").glob("*.csv"):
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        for column, value in context.items():
            frame[column] = value
        frame.to_csv(path, index=False)

    coordinates = pd.read_csv(root / "plot_data" / "coordinates.csv")
    exploration = neutral_relative_coordinates(coordinates)
    for column, value in context.items():
        exploration[column] = value
    exploration["coordinate_origin"] = "neutral"
    exploration["orientation"] = "unaligned eigenvectors; signs and quadrants are descriptive"
    exploration["partition_neighbors"] = config.analysis.knn_neighbors
    projection = pd.read_csv(root / "analyses" / "neutral_projection.csv")
    exploration = exploration.merge(
        projection[
            [
                "layer",
                "neutral_components_removed",
                "neutral_variance_explained",
                "neutral_samples",
                "source_run",
                "activation_dimensions",
            ]
        ],
        on="layer",
        validate="many_to_one",
    )
    samples = pd.read_csv(root / "dataset" / "samples.csv")
    if config.rdm.split != "all":
        samples = samples.loc[samples.split == config.rdm.split]
    counts = samples.groupby("emotion").size().rename("sample_count")
    exploration = exploration.merge(counts, on="emotion", validate="many_to_one")
    exploration.to_csv(root / "plot_data" / "exploration_coordinates.csv", index=False)
    partition = (
        exploration.loc[exploration.dimensions == 2]
        .groupby(["pooling", "layer", "distance", "method", "quadrant", "emotion_group"])
        .size()
        .reset_index(name="emotion_count")
    )
    for column, value in context.items():
        partition[column] = value
    partition.to_csv(root / "analyses" / "neutral_quadrant_counts.csv", index=False)

    reference = pd.read_csv(reference_path).set_index("emotion")
    axes = ["valence", "arousal", "dominance"]
    metrics, aligned_rows = [], []
    grouped = coordinates.loc[coordinates.dimensions == 3].groupby(
        ["pooling", "layer", "distance", "method"], sort=True
    )
    for key, frame in progress(
        grouped,
        description="Evaluating separate VAD recovery",
        total=grouped.ngroups,
        unit="embedding",
    ):
        target = reference.loc[frame.emotion, axes].to_numpy(float)
        x = frame[[f"coordinate_{axis}" for axis in range(1, 4)]].to_numpy(float)
        aligned, score = align_similarity(x, target)
        rng = np.random.default_rng(config.seed)
        exceeded = 0
        for _ in progress(
            range(config.analysis.va_permutations),
            description="Testing VAD alignment",
            unit="permutation",
            leave=False,
        ):
            _, null_score = align_similarity(x, target[rng.permutation(len(target))])
            exceeded += null_score >= score
        record = {
            **context,
            **dict(zip(["pooling", "layer", "distance", "method"], key, strict=True)),
            "dimensions": 3,
            "reference": str(reference_path),
            "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "vad_procrustes_r2": score,
            "permutations": config.analysis.va_permutations,
            "vad_permutation_p": (
                (1 + exceeded) / (config.analysis.va_permutations + 1)
                if config.analysis.va_permutations
                else np.nan
            ),
            **{
                f"{axis}_pearson": float(pearsonr(aligned[:, i], target[:, i]).statistic)
                for i, axis in enumerate(axes)
            },
        }
        metrics.append(record)
        for i, label in enumerate(frame.emotion):
            aligned_rows.append(
                {
                    **record,
                    "emotion": label,
                    **{f"aligned_{axis}": aligned[i, j] for j, axis in enumerate(axes)},
                    **{axis: target[i, j] for j, axis in enumerate(axes)},
                }
            )
    pd.DataFrame(metrics).to_csv(root / "analyses" / "vad_recovery.csv", index=False)
    pd.DataFrame(metrics).to_csv(root / "plot_data" / "vad_recovery.csv", index=False)
    pd.DataFrame(aligned_rows).to_csv(root / "analyses" / "vad_alignment.csv", index=False)
