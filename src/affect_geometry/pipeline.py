from __future__ import annotations

import importlib.metadata
import platform
import random
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from affect_geometry.activations.extract import extract_activations
from affect_geometry.activations.store import load_layer
from affect_geometry.analysis.agreement import (
    embedding_procrustes_r2,
    neighborhood_overlap,
    rdm_spearman,
)
from affect_geometry.analysis.intrinsic import (
    eigenvalue_summary,
    elbow_dimension,
    residual_variance,
)
from affect_geometry.analysis.quality import embedding_quality, geodesic_ratio_summary
from affect_geometry.analysis.va import va_recovery
from affect_geometry.config import ExperimentConfig
from affect_geometry.data.goemotions import load_samples, prepare_goemotions
from affect_geometry.distances.rdm import RDMResult, compute_rdm, load_rdm, save_rdm
from affect_geometry.embeddings.isomap import geodesic_distances, isomap
from affect_geometry.embeddings.mds import classical_mds
from affect_geometry.logging import get_logger, progress
from affect_geometry.persistence.layout import RunLayout


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def initialize_run(config: ExperimentConfig) -> RunLayout:
    layout = RunLayout.create(config.run_dir)
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "transformers", "datasets"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    layout.write_json(
        "manifest.json",
        {
            "config": config.to_dict(),
            "git_commit": _revision(),
            "python": platform.python_version(),
            "packages": packages,
        },
    )
    _seed_everything(config.seed)
    return layout


def prepare(config: ExperimentConfig) -> None:
    logger = get_logger()
    logger.info("stage.start stage=prepare")
    layout = initialize_run(config)
    prepare_goemotions(config.dataset, layout.dataset)
    logger.info("stage.complete stage=prepare")


def extract(config: ExperimentConfig) -> None:
    logger = get_logger()
    logger.info("stage.start stage=extract")
    layout = initialize_run(config)
    samples_path = layout.dataset / "samples.csv"
    if not samples_path.exists():
        prepare_goemotions(config.dataset, layout.dataset)
    samples = load_samples(samples_path)
    metadata = extract_activations(samples, layout.activations, config.model, config.activations)
    layout.write_json("activations/metadata.json", metadata)
    logger.info("stage.complete stage=extract")


def _group_indices(emotions: np.ndarray, labels: list[str]) -> list[np.ndarray]:
    return [np.flatnonzero(emotions == label) for label in labels]


def _cap_indices(indices: list[np.ndarray], maximum: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        np.sort(rng.choice(values, maximum, replace=False)) if len(values) > maximum else values
        for values in indices
    ]


def compute_rdms(config: ExperimentConfig) -> None:
    if config.representation.transformation == "neutral_projected":
        from affect_geometry.experiments.neutral_projected import compute_projected_rdms

        compute_projected_rdms(config, config.run_dir)
        return
    logger = get_logger()
    logger.info("stage.start stage=rdm")
    layout = initialize_run(config)
    all_samples = load_samples(layout.dataset / "samples.csv")
    selected_rows = np.arange(len(all_samples))
    if config.rdm.split != "all":
        selected_rows = np.flatnonzero(all_samples["split"].to_numpy() == config.rdm.split)
    samples = all_samples.iloc[selected_rows].reset_index(drop=True)
    labels = sorted(samples["emotion"].unique().tolist())
    local_indices = _group_indices(samples["emotion"].to_numpy(), labels)
    capped_local = _cap_indices(local_indices, config.rdm.max_samples_per_emotion, config.seed)
    sampling_rows = []
    for label, rows in progress(
        zip(labels, capped_local, strict=True),
        description="Recording distribution samples",
        total=len(labels),
        unit="emotion",
        leave=False,
    ):
        for row in rows:
            sampling_rows.append(
                {
                    "emotion": label,
                    "sample_id": samples.iloc[row]["sample_id"],
                    "activation_row": int(selected_rows[row]),
                }
            )
    pd.DataFrame(sampling_rows).to_csv(layout.rdms / "distribution_sample.csv", index=False)

    layer_files = sorted((layout.activations / config.activations.poolings[0]).glob("layer_*.npy"))
    if not layer_files:
        raise FileNotFoundError("No activations found; run the extract stage first")
    for pooling in progress(
        config.activations.poolings,
        description="Computing RDM pooling modes",
        unit="pooling",
    ):
        for layer, _ in progress(
            enumerate(layer_files),
            description=f"Processing {pooling} layers",
            total=len(layer_files),
            unit="layer",
            leave=False,
        ):
            activations = np.asarray(
                load_layer(layout.activations, pooling, layer)[selected_rows], dtype=np.float32
            )
            fit_rows = np.concatenate(capped_local)
            components = (
                activations.shape[1]
                if config.rdm.pca_components is None
                else min(config.rdm.pca_components, activations.shape[1], len(fit_rows) - 1)
            )
            pca: PCA | None = None
            if components < activations.shape[1]:
                pca = PCA(
                    n_components=components, svd_solver="randomized", random_state=config.seed
                )
                pca.fit(activations[fit_rows])
                reduced = pca.transform(activations)
                checkpoint = layout.checkpoints / pooling / f"layer_{layer:03d}_pca.npz"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    checkpoint,
                    components=pca.components_,
                    mean=pca.mean_,
                    explained_variance=pca.explained_variance_,
                    explained_variance_ratio=pca.explained_variance_ratio_,
                )
            else:
                reduced = activations
            full_groups = [reduced[rows] for rows in local_indices]
            capped_groups = [reduced[rows] for rows in capped_local]
            for distance_name in progress(
                config.rdm.distances,
                description=f"Layer {layer}: constructing RDMs",
                unit="distance",
                leave=False,
            ):
                groups = (
                    capped_groups
                    if distance_name in {"rbf_mmd", "energy", "sliced_wasserstein"}
                    else full_groups
                )
                result = compute_rdm(
                    distance_name,
                    labels,
                    groups,
                    seed=config.seed,
                    folds=config.rdm.crossnobis_folds,
                    bandwidth_multipliers=config.rdm.mmd_bandwidth_multipliers,
                    projections=config.rdm.sliced_wasserstein_projections,
                )
                destination = layout.rdms / pooling / f"layer_{layer:03d}" / distance_name
                save_rdm(
                    result,
                    destination,
                    {
                        "model": config.model.name,
                        "split": config.rdm.split,
                        "pooling": pooling,
                        "layer": layer,
                        "distance": distance_name,
                        "pca_components": reduced.shape[1],
                        "distribution_cap": config.rdm.max_samples_per_emotion,
                    },
                )
    logger.info("stage.complete stage=rdm")


def _embedding_context(path: Path, layout: RunLayout) -> tuple[str, int, str]:
    relative = path.relative_to(layout.rdms)
    return relative.parts[0], int(relative.parts[1].split("_")[-1]), path.stem


def embed(config: ExperimentConfig) -> None:
    logger = get_logger()
    logger.info("stage.start stage=embed")
    layout = initialize_run(config)
    requested_dimensions = sorted(set(config.embedding.dimensions))
    rdm_paths = sorted(layout.rdms.glob("*/layer_*/*.npz"))
    for rdm_path in progress(
        rdm_paths,
        description="Embedding layerwise RDMs",
        unit="RDM",
    ):
        pooling, layer, distance_name = _embedding_context(rdm_path, layout)
        rdm = load_rdm(rdm_path)
        max_dimension = min(
            max(max(requested_dimensions), config.embedding.intrinsic_max_dimensions),
            len(rdm.labels) - 1,
        )
        mds = classical_mds(rdm.distance, max_dimension)
        iso = isomap(rdm.distance, max_dimension, config.embedding.isomap_neighbors)
        destination = layout.embeddings / pooling / f"layer_{layer:03d}" / distance_name
        destination.mkdir(parents=True, exist_ok=True)
        coordinate_rows = []
        for method, coordinates, neighbors_used in (
            ("mds", mds.coordinates, np.nan),
            ("isomap", iso.coordinates, iso.neighbors_used),
        ):
            for dimensions in requested_dimensions:
                for row, emotion in enumerate(rdm.labels):
                    record = {
                        "model": config.model.name,
                        "split": config.rdm.split,
                        "pooling": pooling,
                        "layer": layer,
                        "distance": distance_name,
                        "method": method,
                        "dimensions": dimensions,
                        "emotion": emotion,
                        "neighbors_used": neighbors_used,
                    }
                    for axis in range(3):
                        record[f"coordinate_{axis + 1}"] = (
                            coordinates[row, axis] if axis < dimensions else np.nan
                        )
                    coordinate_rows.append(record)
        pd.DataFrame(coordinate_rows).to_csv(destination / "coordinates.csv", index=False)
        eigen_rows = []
        for method, values in (("mds", mds.eigenvalues), ("isomap", iso.eigenvalues)):
            for index, value in enumerate(values, start=1):
                eigen_rows.append(
                    {
                        "model": config.model.name,
                        "pooling": pooling,
                        "layer": layer,
                        "distance": distance_name,
                        "method": method,
                        "component": index,
                        "eigenvalue": value,
                    }
                )
        pd.DataFrame(eigen_rows).to_csv(destination / "eigenvalues.csv", index=False)
        np.savez_compressed(
            destination / "geodesic_distances.npz", distances=iso.geodesic_distances
        )
        residual_rows = []
        for dimensions in range(1, config.embedding.intrinsic_max_dimensions + 1):
            if dimensions > iso.coordinates.shape[1]:
                break
            residual_rows.append(
                {
                    "model": config.model.name,
                    "pooling": pooling,
                    "layer": layer,
                    "distance": distance_name,
                    "dimensions": dimensions,
                    "residual_variance": residual_variance(
                        iso.geodesic_distances, iso.coordinates[:, :dimensions]
                    ),
                }
            )
        pd.DataFrame(residual_rows).to_csv(
            destination / "isomap_residual_variance.csv", index=False
        )
        sweep_rows = []
        for requested_k in progress(
            config.embedding.isomap_neighbor_grid,
            description=f"Layer {layer}: sweeping Isomap neighbors",
            unit="setting",
            leave=False,
        ):
            geodesic, used_k = geodesic_distances(rdm.distance, requested_k)
            sweep_embedding = classical_mds(geodesic, min(2, max_dimension)).coordinates
            sweep_rows.append(
                {
                    "requested_neighbors": requested_k,
                    "neighbors_used": used_k,
                    "stress_to_direct": embedding_quality(
                        rdm.distance, sweep_embedding, config.embedding.quality_neighbors
                    )["stress"],
                    "residual_variance_2d": residual_variance(geodesic, sweep_embedding),
                }
            )
        pd.DataFrame(sweep_rows).to_csv(destination / "isomap_neighbor_sweep.csv", index=False)
    logger.info("stage.complete stage=embed")


def _coordinates(frame: pd.DataFrame, method: str, dimensions: int) -> tuple[list[str], np.ndarray]:
    part = frame.loc[(frame["method"] == method) & (frame["dimensions"] == dimensions)]
    columns = [f"coordinate_{axis}" for axis in range(1, dimensions + 1)]
    return part["emotion"].tolist(), part[columns].to_numpy(float)


def analyze(config: ExperimentConfig) -> None:
    logger = get_logger()
    logger.info("stage.start stage=analyze")
    layout = initialize_run(config)
    reference = pd.read_csv(config.analysis.va_reference)
    quality_rows: list[dict[str, object]] = []
    va_rows: list[dict[str, object]] = []
    va_alignment_rows: list[dict[str, object]] = []
    intrinsic_rows: list[dict[str, object]] = []
    spectrum_frames: list[pd.DataFrame] = []
    residual_frames: list[pd.DataFrame] = []
    geodesic_rows: list[dict[str, object]] = []
    geodesic_pair_rows: list[dict[str, object]] = []
    all_coordinates: list[pd.DataFrame] = []
    coordinates_cache: dict[tuple[str, int, str, str, int], tuple[list[str], np.ndarray]] = {}
    rdm_cache: dict[tuple[str, int, str], RDMResult] = {}

    coordinate_paths = sorted(layout.embeddings.glob("*/layer_*/*/coordinates.csv"))
    for coordinate_path in progress(
        coordinate_paths,
        description="Analyzing embedded RDMs",
        unit="embedding",
    ):
        relative = coordinate_path.relative_to(layout.embeddings)
        pooling, layer_part, distance_name = relative.parts[:3]
        layer = int(layer_part.split("_")[-1])
        context = {"pooling": pooling, "layer": layer, "distance": distance_name}
        rdm = load_rdm(layout.rdms / pooling / layer_part / f"{distance_name}.npz")
        rdm_cache[(pooling, layer, distance_name)] = rdm
        coordinate_frame = pd.read_csv(coordinate_path)
        all_coordinates.append(coordinate_frame)
        for method in ("mds", "isomap"):
            for dimensions in config.embedding.dimensions:
                labels, coordinates = _coordinates(coordinate_frame, method, dimensions)
                coordinates_cache[(pooling, layer, distance_name, method, dimensions)] = (
                    labels,
                    coordinates,
                )
                metrics = embedding_quality(
                    rdm.distance, coordinates, config.embedding.quality_neighbors
                )
                quality_rows.append(
                    {**context, "method": method, "dimensions": dimensions, **metrics}
                )
                if dimensions == 2:
                    va_metrics, aligned = va_recovery(
                        labels,
                        coordinates,
                        reference,
                        permutations=config.analysis.va_permutations,
                        seed=config.seed,
                    )
                    va_rows.append({**context, "method": method, **va_metrics})
                    for key, value in context.items():
                        aligned[key] = value
                    aligned["method"] = method
                    va_alignment_rows.extend(aligned.to_dict("records"))
        eigen_frame = pd.read_csv(coordinate_path.parent / "eigenvalues.csv")
        spectrum_frames.append(eigen_frame)
        for method in ("mds", "isomap"):
            values = eigen_frame.loc[eigen_frame["method"] == method, "eigenvalue"].to_numpy()
            intrinsic_rows.append({**context, "method": method, **eigenvalue_summary(values)})
        residual_frame = pd.read_csv(coordinate_path.parent / "isomap_residual_variance.csv")
        residual_frames.append(residual_frame)
        intrinsic_rows[-1]["isomap_residual_elbow"] = elbow_dimension(
            residual_frame["dimensions"].astype(int).tolist(),
            residual_frame["residual_variance"].tolist(),
        )
        geodesic = np.load(coordinate_path.parent / "geodesic_distances.npz")["distances"]
        geodesic_rows.append({**context, **geodesic_ratio_summary(rdm.distance, geodesic)})
        for i, emotion_i in enumerate(rdm.labels):
            for j in range(i + 1, len(rdm.labels)):
                direct = rdm.distance[i, j]
                geodesic_pair_rows.append(
                    {
                        **context,
                        "emotion_i": emotion_i,
                        "emotion_j": rdm.labels[j],
                        "direct_distance": direct,
                        "geodesic_distance": geodesic[i, j],
                        "geodesic_direct_ratio": (
                            geodesic[i, j] / direct if direct > np.finfo(float).eps else np.nan
                        ),
                    }
                )

    quality = pd.DataFrame(quality_rows)
    va_metrics = pd.DataFrame(va_rows)
    intrinsic = pd.DataFrame(intrinsic_rows)
    coordinates_all = pd.concat(all_coordinates, ignore_index=True).merge(
        reference[["emotion", "valence", "arousal"]], on="emotion", how="left"
    )
    spectrum = pd.concat(spectrum_frames, ignore_index=True)
    residuals = pd.concat(residual_frames, ignore_index=True)

    rdm_agreement_rows = []
    embedding_agreement_rows = []
    neighborhood_rows = []
    group_keys = sorted({(key[0], key[1]) for key in rdm_cache})
    for pooling, layer in progress(
        group_keys,
        description="Comparing distance notions",
        unit="layer",
    ):
        names = sorted(key[2] for key in rdm_cache if key[:2] == (pooling, layer))
        for left, right in combinations(names, 2):
            first, second = rdm_cache[(pooling, layer, left)], rdm_cache[(pooling, layer, right)]
            rdm_agreement_rows.append(
                {
                    "pooling": pooling,
                    "layer": layer,
                    "distance_i": left,
                    "distance_j": right,
                    "spearman": rdm_spearman(first.distance, second.distance),
                }
            )
            overlap, changes = neighborhood_overlap(
                first.labels,
                first.distance,
                second.distance,
                config.analysis.knn_neighbors,
            )
            for column, value in {
                "pooling": pooling,
                "layer": layer,
                "distance_i": left,
                "distance_j": right,
            }.items():
                changes[column] = value
            changes["mean_pair_overlap"] = overlap
            neighborhood_rows.extend(changes.to_dict("records"))
            for method in ("mds", "isomap"):
                for dimensions in config.embedding.dimensions:
                    first_coordinates = coordinates_cache[
                        (pooling, layer, left, method, dimensions)
                    ][1]
                    second_coordinates = coordinates_cache[
                        (pooling, layer, right, method, dimensions)
                    ][1]
                    embedding_agreement_rows.append(
                        {
                            "pooling": pooling,
                            "layer": layer,
                            "distance_i": left,
                            "distance_j": right,
                            "method": method,
                            "dimensions": dimensions,
                            "procrustes_r2": embedding_procrustes_r2(
                                first_coordinates, second_coordinates
                            ),
                        }
                    )

    cross_layer_rows = []
    layerwise_contexts = [
        (pooling, distance_name, method)
        for pooling in config.activations.poolings
        for distance_name in config.rdm.distances
        for method in ("mds", "isomap")
    ]
    for pooling, distance_name, method in progress(
        layerwise_contexts,
        description="Comparing representations across layers",
        unit="geometry",
    ):
        keys = sorted(
            key
            for key in coordinates_cache
            if key[0] == pooling and key[2] == distance_name and key[3] == method and key[4] == 2
        )
        for left_key, right_key in combinations(keys, 2):
            cross_layer_rows.append(
                {
                    "pooling": pooling,
                    "distance": distance_name,
                    "method": method,
                    "layer_i": left_key[1],
                    "layer_j": right_key[1],
                    "procrustes_r2": embedding_procrustes_r2(
                        coordinates_cache[left_key][1], coordinates_cache[right_key][1]
                    ),
                }
            )

    comparison = quality.pivot_table(
        index=["pooling", "layer", "distance", "dimensions"],
        columns="method",
        values=["stress", "trustworthiness", "continuity", "shepard_spearman"],
    )
    comparison.columns = [f"{metric}_{method}" for metric, method in comparison.columns]
    comparison = comparison.reset_index()
    for metric in ("stress", "trustworthiness", "continuity", "shepard_spearman"):
        comparison[f"{metric}_isomap_minus_mds"] = (
            comparison[f"{metric}_isomap"] - comparison[f"{metric}_mds"]
        )

    outputs = {
        "embedding_quality.csv": quality,
        "va_recovery.csv": va_metrics,
        "intrinsic_dimension.csv": intrinsic,
        "rdm_agreement.csv": pd.DataFrame(rdm_agreement_rows),
        "embedding_agreement.csv": pd.DataFrame(embedding_agreement_rows),
        "neighborhood_change.csv": pd.DataFrame(neighborhood_rows),
        "cross_layer_similarity.csv": pd.DataFrame(cross_layer_rows),
        "mds_isomap_comparison.csv": comparison,
        "geodesic_ratios.csv": pd.DataFrame(geodesic_rows),
        "geodesic_pair_ratios.csv": pd.DataFrame(geodesic_pair_rows),
    }
    for name, frame in outputs.items():
        frame.to_csv(layout.analyses / name, index=False)

    plot_tables = {
        "coordinates.csv": coordinates_all,
        "layerwise_metrics.csv": quality.merge(
            va_metrics, on=["pooling", "layer", "distance", "method"], how="left"
        ).merge(intrinsic, on=["pooling", "layer", "distance", "method"], how="left"),
        "va_alignment.csv": pd.DataFrame(va_alignment_rows),
        "intrinsic_spectrum.csv": spectrum,
        "isomap_residual_variance.csv": residuals,
        "distance_agreement.csv": pd.DataFrame(rdm_agreement_rows),
        "neighborhood_change.csv": pd.DataFrame(neighborhood_rows),
        "mds_isomap_comparison.csv": comparison,
        "geodesic_ratios.csv": pd.DataFrame(geodesic_rows),
        "geodesic_pair_ratios.csv": pd.DataFrame(geodesic_pair_rows),
        "cross_layer_similarity.csv": pd.DataFrame(cross_layer_rows),
    }
    for name, frame in plot_tables.items():
        frame.to_csv(layout.plot_data / name, index=False)
    logger.info("stage.complete stage=analyze")


def plot(config: ExperimentConfig) -> None:
    from affect_geometry.reporting.plots import create_all_plots

    logger = get_logger()
    logger.info("stage.start stage=plot")
    layout = initialize_run(config)
    create_all_plots(layout.plot_data, layout.plots)
    logger.info("stage.complete stage=plot")


def run(config: ExperimentConfig) -> None:
    logger = get_logger()
    logger.info("pipeline.start")
    prepare(config)
    extract(config)
    compute_rdms(config)
    embed(config)
    analyze(config)
    plot(config)
    logger.info("pipeline.complete")
