#!/usr/bin/env python3
"""Archive saved EXP5 neuron-community runs in a Hugging Face dataset folder."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from _upload_common import human_size, load_manifest, load_project_environment

from affect_geometry.logging import configure_file_logging, get_logger, progress
from affect_geometry.persistence.correctness_geometry import checksum

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "allenai/OLMo-2-1124-7B-Instruct"
DEFAULT_REPO = "kokolamba/exp3-olmo2"
REMOTE_FOLDER = "neuron-communities"


def run_files(run: Path, model: str) -> dict[str, Path]:
    manifest = load_manifest(run)
    config = manifest["config"]
    if (config["model_id"], config["run_id"]) != (model, run.name):
        raise ValueError(f"EXP5 manifest identity does not match the run path: {run}")

    files = {}
    for path in sorted(run.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"EXP5 run contains a symbolic link: {path}")
        if path.is_file():
            files[path.relative_to(run).as_posix()] = path
    return files


def verify_upload(api, repo_id: str, prefix: str, revision: str, files: dict[str, Path]) -> None:
    from huggingface_hub import RepoFile

    remote_files = {
        entry.path: entry
        for entry in api.list_repo_tree(
            repo_id, prefix, recursive=True, revision=revision, repo_type="dataset"
        )
        if isinstance(entry, RepoFile)
    }
    for name, local in files.items():
        remote_path = f"{prefix}/{name}"
        remote = remote_files.get(remote_path)
        if remote is None:
            raise RuntimeError(f"Missing Hub artifact: {remote_path}")
        if remote.size != local.stat().st_size:
            raise RuntimeError(f"Hub artifact size mismatch: {remote_path}")
        if remote.lfs is not None:
            matches = remote.lfs.sha256 == checksum(local)
        else:
            data = local.read_bytes()
            blob = f"blob {len(data)}\0".encode() + data
            matches = remote.blob_id == hashlib.sha1(blob, usedforsecurity=False).hexdigest()
        if not matches:
            raise RuntimeError(f"Hub artifact checksum mismatch: {remote_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO, help="Existing Hugging Face dataset")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID in the EXP5 manifest")
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "outputs/neuron-transfer"
    )
    parser.add_argument("--run-id", action="append", help="Repeat to select runs; defaults to all")
    parser.add_argument("--dry-run", action="store_true", help="List runs and sizes locally")
    args = parser.parse_args()

    model_slug = args.model.replace("/", "--")
    model_root = args.output_root / model_slug
    if not model_root.is_dir():
        parser.error(f"EXP5 model output directory does not exist: {model_root}")
    runs = {
        path.name: path
        for path in sorted(model_root.iterdir())
        if path.is_dir() and not path.is_symlink() and (path / "manifest.json").is_file()
    }
    run_ids = list(dict.fromkeys(args.run_id or runs))
    if not run_ids:
        parser.error(f"No EXP5 runs with manifests found in {model_root}")
    missing = set(run_ids) - runs.keys()
    if missing:
        names = ", ".join(sorted(missing))
        parser.error(f"EXP5 run directories missing from {model_root}: {names}")

    selections = {run_id: run_files(runs[run_id], args.model) for run_id in run_ids}
    print(f"Destination: hf://datasets/{args.repo_id}/{REMOTE_FOLDER}/")
    for run_id, files in selections.items():
        status = "complete" if "checkpoints/run.json" in files else "partial"
        size = human_size(sum(path.stat().st_size for path in files.values()))
        print(f"{run_id}: {len(files)} files, {size} ({status})")
    if args.dry_run:
        return

    load_project_environment(PROJECT_ROOT)
    from huggingface_hub import HfApi

    api = HfApi(token=True)
    configure_file_logging(
        PROJECT_ROOT / "logs" / f"{model_slug}__{REMOTE_FOLDER}__hub-upload",
        "upload_neuron_communities",
        repo=args.repo_id,
        path=REMOTE_FOLDER,
        runs=",".join(run_ids),
    )
    api.dataset_info(args.repo_id)
    logger = get_logger()
    for run_id in progress(run_ids, description="Uploading and verifying EXP5 runs", unit="run"):
        prefix = f"{REMOTE_FOLDER}/{run_id}"
        files = selections[run_id]
        logger.info(
            "upload.start run_id=%s files=%d status=%s",
            run_id,
            len(files),
            "complete" if "checkpoints/run.json" in files else "partial",
        )
        try:
            commit = api.upload_folder(
                repo_id=args.repo_id,
                repo_type="dataset",
                folder_path=runs[run_id],
                path_in_repo=prefix,
                allow_patterns=list(files),
                commit_message=f"Archive EXP5 neuron-community artifacts for {run_id}",
            )
            revision = commit.oid
            verify_upload(api, args.repo_id, prefix, revision, files)
        except Exception:
            logger.exception("upload.failed run_id=%s", run_id)
            raise
        url = f"{api.endpoint}/datasets/{args.repo_id}/tree/{revision}/{prefix}"
        logger.info("upload.verified run_id=%s files=%d url=%s", run_id, len(files), url)
        print(f"Verified {len(files)} files: {url}")


if __name__ == "__main__":
    main()
