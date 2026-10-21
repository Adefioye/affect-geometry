"""Orchestration of LT-NeutralProjected geometry from stored sample activations."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from affect_geometry.activations.source import ActivationSource
from affect_geometry.analysis.representation_comparison import cosine_rdm
from affect_geometry.analysis.representations import build_pooling_representations
from affect_geometry.config import ExperimentConfig
from affect_geometry.distances.rdm import RDMResult, compute_rdm, save_rdm
from affect_geometry.logging import get_logger, progress


def compute_projected_rdms(config: ExperimentConfig, source_root: Path) -> None:
    """Compute all six distances in the full neutral-projected hidden space.

    PCA only identifies the neutral subspace to remove. No PCA reduction, rotation,
    or further centering is applied after projection, regardless of rdm.pca_components.
    """
    from affect_geometry.pipeline import initialize_run

    if source_root.resolve() == config.run_dir.resolve():
        preserved_source = config.run_dir / "activation_source"
        if not (preserved_source / "manifest.json").is_file():
            ActivationSource.open(source_root, "last_token")
            preserved_source.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_root / "manifest.json", preserved_source / "manifest.json")
            shutil.copytree(
                source_root / "dataset", preserved_source / "dataset", dirs_exist_ok=True
            )
            shutil.move(str(source_root / "activations"), str(preserved_source / "activations"))
        source_root = preserved_source
    source = ActivationSource.open(source_root, "last_token")
    if source.model_name != config.model.name:
        raise ValueError(f"Source model {source.model_name!r} differs from {config.model.name!r}")
    if config.model.revision is not None and source.model_revision != config.model.revision:
        raise ValueError("Source model revision differs from the configured revision")
    layout = initialize_run(config)
    get_logger().info("stage.start stage=rdm variant=LT-NeutralProjected source=%s", source_root)
    if source_root.resolve() != config.run_dir.resolve():
        for path in (source_root / "dataset").iterdir():
            if path.is_file():
                shutil.copy2(path, layout.dataset / path.name)
    all_samples = source.samples
    rows = np.arange(len(all_samples))
    if config.rdm.split != "all":
        rows = np.flatnonzero(all_samples["split"].to_numpy() == config.rdm.split)
    samples = all_samples.iloc[rows].reset_index(drop=True)
    if samples.empty:
        raise ValueError(f"No samples in split {config.rdm.split!r}")
    samples.to_csv(layout.dataset / "samples.csv", index=False)
    samples.groupby(["split", "emotion"]).size().reset_index(name="size").to_csv(
        layout.dataset / "class_counts.csv", index=False
    )
    projected_directory = layout.activations / "last_token"
    projected_directory.mkdir(parents=True, exist_ok=True)
    # Publish completion metadata only after every projected layer has been written.
    (layout.activations / "metadata.json").unlink(missing_ok=True)
    labels = sorted(samples.emotion.unique())
    indices = [np.flatnonzero(samples.emotion.to_numpy() == label) for label in labels]
    rng = np.random.default_rng(config.seed)
    capped = [
        np.sort(rng.choice(group, config.rdm.max_samples_per_emotion, replace=False))
        if len(group) > config.rdm.max_samples_per_emotion
        else group
        for group in indices
    ]
    pd.DataFrame(
        [
            {
                "emotion": label,
                "sample_id": samples.iloc[row].sample_id,
                "activation_row": int(rows[row]),
            }
            for label, group in zip(labels, capped, strict=True)
            for row in group
        ]
    ).to_csv(layout.rdms / "distribution_sample.csv", index=False)
    diagnostics = []
    for layer in progress(
        range(source.n_layers), description="Projecting last-token layers", unit="layer"
    ):
        activations = source.load_layer(layer)
        # Avoid copying the complete memmap when the complete dataset is selected.
        if config.rdm.split != "all":
            activations = activations[rows]
        representation = build_pooling_representations(
            "last_token",
            activations,
            samples,
            neutral_variance_threshold=config.representation.neutral_variance_threshold,
            batch_size=config.representation.batch_size,
        )
        subspace = representation.neutral_subspace
        checkpoint = layout.checkpoints / "neutral_projection" / f"layer_{layer:03d}.npz"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            checkpoint,
            grand_mean=representation.grand_mean,
            neutral_mean=subspace.mean,
            components=subspace.components,
            explained_variance=subspace.explained_variance,
            explained_variance_ratio=subspace.explained_variance_ratio,
            labels=representation.centroids.labels,
            sample_counts=representation.centroids.sample_counts,
            projected_centroids=representation.neutral_projected,
        )
        context = {
            "model": config.model.name,
            "run_id": config.run_id,
            "split": config.rdm.split,
            "pooling": "last_token",
            "variant": "LT-NeutralProjected",
            "layer": layer,
            "neutral_variance_threshold": subspace.variance_threshold,
            "neutral_components_removed": subspace.n_components,
            "neutral_variance_explained": subspace.cumulative_explained_variance,
            "source_run": str(source_root.resolve()),
            "activation_dimensions": activations.shape[1],
            "post_projection_pca": False,
        }
        diagnostics.append({**context, "neutral_samples": subspace.neutral_samples})
        # Remove obsolete checkpoints when recomputing a run created by the older workflow.
        checkpoint.with_name(f"layer_{layer:03d}_auxiliary_pca.npz").unlink(missing_ok=True)
        projected = np.lib.format.open_memmap(
            projected_directory / f"layer_{layer:03d}.npy",
            mode="w+",
            dtype=np.float32,
            shape=activations.shape,
        )
        for start in progress(
            range(0, len(activations), config.representation.batch_size),
            description=f"Layer {layer}: projecting sample batches",
            unit="batch",
            leave=False,
        ):
            stop = start + config.representation.batch_size
            projected[start:stop] = subspace.project_out(
                np.asarray(activations[start:stop], dtype=float) - representation.grand_mean
            )
        projected.flush()
        for name in progress(
            config.rdm.distances,
            description=f"Layer {layer}: projected RDMs",
            unit="distance",
            leave=False,
        ):
            if name == "cosine":
                distance = cosine_rdm(representation.neutral_projected, labels=labels)
                result = RDMResult(labels, distance, distance.copy())
            else:
                if name == "euclidean":
                    groups = [center[None, :] for center in representation.neutral_projected]
                else:
                    selected = indices if name == "crossnobis" else capped
                    groups = [projected[group] for group in selected]
                result = compute_rdm(
                    name,
                    labels,
                    groups,
                    seed=config.seed,
                    folds=config.rdm.crossnobis_folds,
                    bandwidth_multipliers=config.rdm.mmd_bandwidth_multipliers,
                    projections=config.rdm.sliced_wasserstein_projections,
                )
            save_rdm(
                result,
                layout.rdms / "last_token" / f"layer_{layer:03d}" / name,
                {
                    **context,
                    "distance": name,
                    "pca_components": 0,
                    "distribution_cap": config.rdm.max_samples_per_emotion,
                },
            )
    pd.DataFrame(diagnostics).to_csv(layout.analyses / "neutral_projection.csv", index=False)
    layout.write_json(
        "activations/metadata.json",
        {
            **source.metadata,
            "n_samples": len(samples),
            "pooling": "last_token",
            "variant": "LT-NeutralProjected",
            "transformation": "neutral_projected",
            "storage_dtype": "float32",
            "source_run": str(source_root.resolve()),
            "source_split": config.rdm.split,
            "post_projection_pca": False,
            "neutral_variance_threshold": config.representation.neutral_variance_threshold,
            "complete": True,
        },
    )
    layout.write_json(
        "representation.json",
        {
            "variant": "LT-NeutralProjected",
            "source_run": str(source_root.resolve()),
            "source_manifest": source.manifest,
            "source_metadata": source.metadata,
            "transformation": (
                "(activation - equal_weight_non_neutral_centroid_mean) @ (I - U.T @ U)"
            ),
            "neutral_variance_threshold": config.representation.neutral_variance_threshold,
            "centroid_metrics": "full hidden space after neutral projection",
            "sample_metrics": "full hidden space after neutral projection",
            "post_projection_pca": False,
            "rdm_pca_components_used": 0,
        },
    )
    get_logger().info("stage.complete stage=rdm variant=LT-NeutralProjected")


PROJECTED_STAGES = ("prepare", "reference", "extract", "rdm", "embed", "analyze", "plot")


def run_projected_experiment(
    config: ExperimentConfig,
    *,
    source_run: Path | None = None,
    local_lexicon: Path | None = None,
    start_at: str = "prepare",
    dry_run: bool = False,
) -> Path:
    """Run or resume a complete experiment, optionally reusing immutable activation files."""
    from affect_geometry.analysis.neutral_projected import (
        NEGATIVE,
        POSITIVE,
        analyze_projected,
    )
    from affect_geometry.data.nrc_vad import prepare_nrc_vad
    from affect_geometry.logging import configure_run_logging
    from affect_geometry.pipeline import embed, extract, prepare
    from affect_geometry.reporting.neutral_projected import render_projected_plots

    if config.activations.poolings != ["last_token"]:
        raise ValueError('activations.poolings must be exactly ["last_token"]')
    if config.representation.transformation != "neutral_projected":
        raise ValueError("representation.transformation must be neutral_projected")
    if not config.dataset.include_neutral:
        raise ValueError("The experiment requires dataset.include_neutral=true")
    if sorted(config.embedding.dimensions) != [1, 2, 3]:
        raise ValueError("The experiment requires embedding.dimensions=[1, 2, 3]")
    if start_at not in PROJECTED_STAGES:
        raise ValueError(f"Unknown stage {start_at!r}")
    if not {"cosine", "euclidean"}.issubset(config.rdm.distances):
        raise ValueError("Include both cosine and euclidean for centroid geometry comparison")
    if source_run is not None:
        if source_run.resolve() == config.run_dir.resolve():
            raise ValueError("Use a new output run when reusing a source activation run")
        source = ActivationSource.open(source_run, "last_token")
        if source.model_name != config.model.name:
            raise ValueError("Source model differs from the experiment model")
    print(f"LT-NeutralProjected · {config.model.name} · {config.run_id}")
    print(f"Outputs: {config.run_dir.resolve()}\nLogs: {config.log_dir.resolve()}")
    stages = PROJECTED_STAGES[PROJECTED_STAGES.index(start_at) :]
    if dry_run:
        for stage in stages:
            detail = (
                " (reuse --source-run)" if source_run and stage in {"prepare", "extract"} else ""
            )
            print(f"{stage}{detail}")
        return config.run_dir / "exploration.html"
    configure_run_logging(config, "lt_neutral_projected")
    for stage in progress(
        stages, description="Running LT-NeutralProjected experiment", unit="stage"
    ):
        logger = get_logger()
        logger.info("stage.start stage=%s", stage)
        try:
            if stage in {"prepare", "extract"} and source_run is not None:
                logger.info("stage.reused stage=%s source=%s", stage, source_run)
            elif stage == "prepare":
                prepare(config)
            elif stage == "reference":
                samples = pd.read_csv((source_run or config.run_dir) / "dataset" / "samples.csv")
                expected = (
                    POSITIVE
                    | NEGATIVE
                    | {"confusion", "curiosity", "realization", "surprise", "neutral"}
                )
                labels = set(samples.emotion)
                if labels != expected:
                    raise ValueError(
                        f"Expected all 28 GoEmotions labels; missing={expected - labels}, "
                        f"extra={labels - expected}"
                    )
                prepare_nrc_vad(
                    sorted(labels),
                    config.run_dir / "references",
                    local_lexicon=local_lexicon,
                    local_reference=(
                        Path(config.analysis.vad_reference) if local_lexicon is None else None
                    ),
                )
            elif stage == "extract":
                extract(config)
            elif stage == "rdm":
                compute_projected_rdms(config, source_run or config.run_dir)
            elif stage == "embed":
                embed(config)
            elif stage == "analyze":
                analyze_projected(config)
            elif stage == "plot":
                render_projected_plots(config.run_dir)
        except BaseException:
            logger.exception("stage.failed stage=%s", stage)
            print(f"Resume with the same options plus --run-id {config.run_id} --start-at {stage}")
            raise
        logger.info("stage.complete stage=%s", stage)
    return config.run_dir / "exploration.html"
