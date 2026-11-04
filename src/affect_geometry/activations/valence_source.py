"""Resolve local or immutable Hub activation inputs for valence transfer."""

import re
from pathlib import Path

from affect_geometry.activations.rq3_bundle import PreparedActivations
from affect_geometry.persistence.rq3_hub import download_bundle, repo_path


def validate_source(settings):
    """Require exactly one explicit source; never silently substitute local data."""
    if bool(settings.get("bundle")) == bool(settings.get("activation_source")):
        raise ValueError("Specify exactly one of bundle or activation_source")
    if source := settings.get("activation_source"):
        if not re.fullmatch(r"[0-9a-f]{40}", source.get("revision", "")):
            raise ValueError("Activation source revision must be an immutable commit SHA")
        repo_path(source["path_in_repo"])
        if not source.get("repo_id"):
            raise ValueError("Activation source requires repo_id")


def open_source(settings, model, project_root):
    """Download a pinned complete bundle, then validate all recorded checksums."""
    validate_source(settings)
    remote = None
    if source := settings.get("activation_source"):
        path, remote = download_bundle(
            source, Path(project_root) / "outputs/valence-transfer/activation-cache" / model
        )
        if remote["revision"] != source["revision"]:
            raise ValueError("Resolved activation revision differs from configured commit")
    else:
        path = settings["bundle"]
    return PreparedActivations.open(path, expected_model=model), remote
