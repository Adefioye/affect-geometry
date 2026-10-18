from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def load_manifest(run_directory: Path) -> dict[str, Any]:
    manifest_path = run_directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def manifest_model(manifest: dict[str, Any]) -> str:
    try:
        model = manifest["config"]["model"]["name"]
    except (KeyError, TypeError) as error:
        raise ValueError("manifest.json does not contain config.model.name") from error
    if not isinstance(model, str) or not model.strip():
        raise ValueError("manifest config.model.name must be a non-empty string")
    return model


def resolve_model(manifest: dict[str, Any], requested_model: str | None) -> str:
    recorded_model = manifest_model(manifest)
    if requested_model is not None and requested_model != recorded_model:
        raise ValueError(
            f"Requested model {requested_model!r} does not match manifest model "
            f"{recorded_model!r}"
        )
    return recorded_model


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "--", value.strip()).strip("-.")
    if not slug:
        raise ValueError(f"Cannot construct a repository path from {value!r}")
    return slug.lower()


def resolve_run_directory(output_root: str | Path, run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("--run-id must be a single output-directory name")
    root = Path(output_root).expanduser().resolve()
    run_directory = (root / run_id).resolve()
    if run_directory.parent != root:
        raise ValueError("Resolved run directory falls outside --output-root")
    if not run_directory.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_directory}")
    return run_directory


def selected_files(
    run_directory: Path,
    directories: Iterable[str],
    root_files: Iterable[str] = ("manifest.json",),
) -> list[Path]:
    files = [run_directory / name for name in root_files if (run_directory / name).is_file()]
    for directory in directories:
        base = run_directory / directory
        if base.is_dir():
            files.extend(path for path in base.rglob("*") if path.is_file())
    return sorted(set(files))


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def print_plan(
    *,
    run_directory: Path,
    repo_id: str,
    path_in_repo: str,
    files: list[Path],
    private: bool,
) -> None:
    total_size = sum(path.stat().st_size for path in files)
    print(f"Local run:       {run_directory}")
    print(f"Destination:     hf://datasets/{repo_id}/{path_in_repo}")
    print(f"New repo access: {'private' if private else 'public'}")
    print(f"Selected files:  {len(files)}")
    print(f"Selected size:   {human_size(total_size)}")


def upload_selection(
    *,
    run_directory: Path,
    repo_id: str,
    path_in_repo: str,
    allow_patterns: list[str],
    private: bool,
    commit_message: str,
) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required; install it with "
            "`python -m pip install -U huggingface_hub`"
        ) from error

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=run_directory,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        commit_message=commit_message,
    )
