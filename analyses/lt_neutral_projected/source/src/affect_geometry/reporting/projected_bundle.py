"""Export a portable analysis directory from the canonical projected-run artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from affect_geometry.logging import get_logger, progress


def bundle_projected_report(run: Path, destination: Path) -> None:
    """Copy report assets and the original gallery alongside the analysis README."""
    root = run / "report"
    destination.mkdir(parents=True, exist_ok=True)
    directories = [
        (root / "figures", destination / "figures"),
        (root / "tables", destination / "tables"),
        (root / "plot_data", destination / "plot_data"),
        (root / "source", destination / "source"),
        (run / "plots", destination / "plots"),
        (run / "plot_data", destination / "plot_data" / "original"),
    ]
    for source, target in progress(
        directories, description="Bundling analysis figures and data", unit="directory"
    ):
        shutil.copytree(source, target, dirs_exist_ok=True)

    metadata = destination / "metadata"
    metadata.mkdir(exist_ok=True)
    for path in root.iterdir():
        if path.is_file() and path.suffix in {".json", ".csv"}:
            shutil.copy2(path, metadata / path.name)

    snapshots = [
        "manifest.json",
        "representation.json",
        "activations/metadata.json",
        "references/goemotions_nrc_vad.csv",
        "references/nrc_vad_provenance.json",
        "analyses/run_validation.json",
        "plot_manifest.json",
        "orchestration/status.json",
        "orchestration/upload_verification.json",
    ]
    for relative in snapshots:
        source = run / relative
        if source.exists():
            target = metadata / "run" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    gallery = (run / "exploration.html").read_text(encoding="utf-8")
    gallery = gallery.replace('href="plot_data/"', 'href="plot_data/original/"')
    (destination / "exploration.html").write_text(gallery, encoding="utf-8")
    plot_manifest = json.loads((run / "plot_manifest.json").read_text())
    for entry in plot_manifest.values():
        if isinstance(entry, dict) and "csv" in entry:
            entry["csv"] = entry["csv"].replace("plot_data/", "plot_data/original/", 1)
    (metadata / "plot_manifest.json").write_text(json.dumps(plot_manifest, indent=2) + "\n")
    get_logger().info("report.bundle.complete destination=%s", destination)
