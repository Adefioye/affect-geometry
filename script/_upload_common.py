from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
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


def manifest_run_id(manifest: dict[str, Any]) -> str:
    try:
        run_id = manifest["config"]["run_id"]
    except (KeyError, TypeError) as error:
        raise ValueError("manifest.json does not contain config.run_id") from error
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("manifest config.run_id must be a non-empty string")
    return run_id


def resolve_model(manifest: dict[str, Any], requested_model: str | None) -> str:
    recorded_model = manifest_model(manifest)
    if requested_model is not None and requested_model != recorded_model:
        raise ValueError(
            f"Requested model {requested_model!r} does not match manifest model {recorded_model!r}"
        )
    return recorded_model


def verify_run_id(manifest: dict[str, Any], requested_run_id: str) -> None:
    recorded_run_id = manifest_run_id(manifest)
    if requested_run_id != recorded_run_id:
        raise ValueError(
            f"Requested run ID {requested_run_id!r} does not match manifest run ID "
            f"{recorded_run_id!r}"
        )


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "--", value.strip()).strip("-.")
    if not slug:
        raise ValueError(f"Cannot construct a repository path from {value!r}")
    return slug.lower()


def load_project_environment(project_root: str | Path) -> None:
    """Load repository-local credentials without overriding exported environment values."""
    environment_path = Path(project_root) / ".env"
    if not environment_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise RuntimeError("python-dotenv is required to load the project's .env file") from error
    load_dotenv(environment_path, override=False)


def _validated_repo_path(path_in_repo: str) -> PurePosixPath:
    path = PurePosixPath(path_in_repo)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("A Hugging Face run path must be a safe relative repository path")
    return path


def download_activation_run(
    *,
    repo_id: str,
    path_in_repo: str,
    download_root: str | Path,
    pooling: str | None,
) -> Path:
    """Download one activation run, optionally including one pooling's layer arrays."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required; install it with "
            "`python -m pip install -U huggingface_hub`"
        ) from error

    remote_path = _validated_repo_path(path_in_repo)
    prefix = remote_path.as_posix()
    patterns = [
        f"{prefix}/manifest.json",
        f"{prefix}/dataset/**",
        f"{prefix}/activations/metadata.json",
    ]
    if pooling is not None:
        if not re.fullmatch(r"[a-z0-9_]+", pooling):
            raise ValueError(f"Unsafe activation pooling name: {pooling!r}")
        patterns.append(f"{prefix}/activations/{pooling}/**")

    local_repository = Path(download_root).expanduser().resolve() / safe_slug(repo_id)
    local_repository.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_repository,
        allow_patterns=patterns,
    )
    run_directory = local_repository.joinpath(*remote_path.parts)
    for required in (
        run_directory / "manifest.json",
        run_directory / "dataset" / "samples.csv",
        run_directory / "activations" / "metadata.json",
    ):
        if not required.is_file():
            raise FileNotFoundError(f"Downloaded activation run is missing: {required}")
    return run_directory


def resolve_run_directory(output_root: str | Path, model: str, run_id: str) -> Path:
    if (
        not run_id
        or run_id == "auto"
        or Path(run_id).name != run_id
        or run_id in {".", ".."}
        or safe_slug(run_id) != run_id.lower()
    ):
        raise ValueError(
            "--run-id must be one filesystem-safe name, normally a UTC timestamp such as "
            "20261018T153045Z"
        )
    root = Path(output_root).expanduser().resolve()
    model_directory = (root / safe_slug(model)).resolve()
    if model_directory.parent != root:
        raise ValueError("Resolved model directory falls outside --output-root")
    run_directory = (model_directory / run_id).resolve()
    if run_directory.parent != model_directory:
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
