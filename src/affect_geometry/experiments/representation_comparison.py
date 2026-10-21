from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from affect_geometry.activations.source import ActivationSource, align_activation_sources
from affect_geometry.analysis.representation_comparison import (
    GeometryEvaluationConfig,
    RepresentationComparator,
)
from affect_geometry.analysis.representations import (
    DEFAULT_REPRESENTATION_VARIANTS,
    RepresentationVariant,
)
from affect_geometry.logging import get_logger, progress
from affect_geometry.persistence.representation_comparison import (
    RepresentationComparisonArtifactStore,
)
from affect_geometry.reporting.representation_comparison import (
    create_representation_comparison_plots,
)


@dataclass(frozen=True)
class RepresentationComparisonConfig:
    last_token_run: str | Path
    mean_run: str | Path
    va_reference: str | Path
    output_root: str | Path
    neutral_label: str = "neutral"
    neutral_variance_threshold: float = 0.5
    batch_size: int = 2048
    minimum_vector_norm: float = 1e-12
    variants: tuple[RepresentationVariant, ...] = DEFAULT_REPRESENTATION_VARIANTS


@dataclass(frozen=True)
class RepresentationComparisonResult:
    output_root: Path
    layerwise: pd.DataFrame
    summary: pd.DataFrame
    coordinates: pd.DataFrame
    plot_paths: tuple[Path, ...]


def _run_id(source: ActivationSource) -> str:
    return str(source.manifest.get("config", {}).get("run_id", source.root.name))


def run_representation_comparison(
    config: RepresentationComparisonConfig,
) -> RepresentationComparisonResult:
    """Compare configured representations from two compatible activation stores."""

    logger = get_logger()
    logger.info("stage.start stage=representation_comparison")
    last_token = ActivationSource.open(config.last_token_run, "last_token")
    mean = ActivationSource.open(config.mean_run, "mean")
    alignment = align_activation_sources(last_token, mean)
    reference = pd.read_csv(config.va_reference)
    analysis_config = GeometryEvaluationConfig(
        neutral_label=config.neutral_label,
        neutral_variance_threshold=config.neutral_variance_threshold,
        batch_size=config.batch_size,
        minimum_vector_norm=config.minimum_vector_norm,
        variants=config.variants,
    )
    comparator = RepresentationComparator(alignment.samples, reference, analysis_config)
    store = RepresentationComparisonArtifactStore.create(config.output_root)
    identity_order = np.arange(mean.n_samples, dtype=np.int64)
    mean_row_order = (
        None
        if np.array_equal(alignment.candidate_row_order, identity_order)
        else alignment.candidate_row_order
    )

    store.write_provenance(
        {
            "experiment": "representation_comparison_cosine_mds_procrustes",
            "last_token_run": str(last_token.root.resolve()),
            "last_token_run_id": _run_id(last_token),
            "mean_run": str(mean.root.resolve()),
            "mean_run_id": _run_id(mean),
            "model": last_token.model_name,
            "last_token_model_revision": last_token.model_revision,
            "mean_model_revision": mean.model_revision,
            "samples": last_token.n_samples,
            "layers": last_token.n_layers,
            "hidden_size": last_token.hidden_size,
            "va_reference": str(Path(config.va_reference).resolve()),
            "analysis_config": asdict(analysis_config),
        }
    )

    for layer in progress(
        range(last_token.n_layers),
        description="Comparing activation representations",
        total=last_token.n_layers,
        unit="layer",
    ):
        comparison = comparator.analyze_layer(
            layer,
            last_token.load_layer(layer),
            mean.load_layer(layer, row_order=mean_row_order),
        )
        store.write_layer(comparison)

    layerwise, summary, coordinates = store.finalize()
    output_root = Path(config.output_root)
    plot_paths = tuple(
        create_representation_comparison_plots(output_root / "plot_data", output_root / "plots")
    )
    figure_index = pd.DataFrame(
        [
            {
                "figure_png": "plots/representation_comparison_layerwise.png",
                "figure_pdf": "plots/representation_comparison_layerwise.pdf",
                "plot_data": "plot_data/representation_comparison_layerwise.csv",
                "description": "Layerwise NRC V-A Procrustes R2 by representation variant",
            },
            {
                "figure_png": "plots/representation_comparison_mean.png",
                "figure_pdf": "plots/representation_comparison_mean.pdf",
                "plot_data": "plot_data/representation_comparison_summary.csv",
                "description": "Mean NRC V-A Procrustes R2 across layers",
            },
        ]
    )
    figure_index.to_csv(output_root / "analyses" / "figure_index.csv", index=False)
    logger.info("stage.complete stage=representation_comparison output=%s", output_root)
    return RepresentationComparisonResult(
        output_root=output_root,
        layerwise=layerwise,
        summary=summary,
        coordinates=coordinates,
        plot_paths=plot_paths,
    )
