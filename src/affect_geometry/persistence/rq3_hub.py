"""Publish and retrieve complete RQ3 stages in model-specific Hugging Face datasets."""

import json
from pathlib import Path, PurePosixPath

from affect_geometry.classification.environment import load_hf_token
from affect_geometry.persistence.correctness_geometry import GeometryStore, checksum, fingerprint


def repo_path(value):
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or any(p in (".", "..", "") for p in value.split("/"))
        or any(c in value for c in ("*", "?", "[", "\\"))
    ):
        raise ValueError("Expected a safe relative Hugging Face directory")
    return path.as_posix()


def resolve_repository(template, api):
    if not isinstance(template, str) or template.count("/") != 1:
        raise ValueError("Hugging Face repo_id must be username/dataset")
    if template.startswith("{username}/"):
        template = template.replace("{username}", api.whoami()["name"], 1)
    if "{" in template or "}" in template:
        raise ValueError("Only the {username} repository placeholder is supported")
    from huggingface_hub.utils import validate_repo_id

    validate_repo_id(template)
    return template


def hub_api():
    from huggingface_hub import HfApi

    return HfApi(token=load_hf_token())


def publish_stage(root, *, repo_id, path_in_repo, private=True, api=None):
    """Publish only checksum-covered files plus their atomic completion marker.

    Dataset folders are append-only by convention: an existing different manifest at
    the same path is rejected. Replaying an identical completed upload is harmless.
    """
    from huggingface_hub import hf_hub_download

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    store = GeometryStore(root, {k: v for k, v in manifest.items() if k != "schema_version"})
    if not store.valid("run", store.manifest):
        raise ValueError("Cannot publish an incomplete or modified RQ3 stage")
    record = json.loads((root / "checkpoints/run.json").read_text())
    files = sorted(record["artifacts"]) + ["checkpoints/run.json"]
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != set(files):
        raise ValueError("Run contains uncheckpointed files; cannot silently omit outputs")
    api = api or hub_api()
    repository = resolve_repository(repo_id, api)
    prefix = repo_path(path_in_repo)
    api.create_repo(repo_id=repository, repo_type="dataset", private=private, exist_ok=True)
    # Guard an existing run namespace using only the manifest (no downloads of arrays).
    if api.file_exists(repo_id=repository, filename=f"{prefix}/manifest.json", repo_type="dataset"):
        remote = hf_hub_download(
            repo_id=repository,
            filename=f"{prefix}/manifest.json",
            repo_type="dataset",
            token=api.token,
            local_dir=root.parent / "hub-manifest-cache",
        )
        if json.loads(Path(remote).read_text()) != manifest:
            raise ValueError("Remote run manifest differs; choose a new run_id/path")
    commit = api.upload_folder(
        repo_id=repository,
        repo_type="dataset",
        folder_path=root,
        path_in_repo=prefix,
        allow_patterns=files,
        commit_message=f"RQ3 {manifest.get('model', 'model')} {prefix}",
    )
    return dict(
        repo_id=repository,
        path_in_repo=prefix,
        revision=commit.oid,
        url=f"https://huggingface.co/datasets/{repository}/tree/{commit.oid}/{prefix}",
        manifest_sha256=checksum(root / "manifest.json"),
        files=len(files),
    )


def download_bundle(source, destination, api=None):
    """Resolve a moving revision once, then download the entire bundle at that commit."""
    from huggingface_hub import snapshot_download

    api = api or hub_api()
    repository = resolve_repository(source["repo_id"], api)
    prefix = repo_path(source["path_in_repo"])
    revision = api.repo_info(
        repo_id=repository, repo_type="dataset", revision=source["revision"]
    ).sha
    destination = Path(destination) / fingerprint(dict(repo=repository, revision=revision))[:20]
    snapshot_download(
        repo_id=repository,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"{prefix}/**"],
        local_dir=destination,
        token=api.token,
    )
    return destination.joinpath(*PurePosixPath(prefix).parts), dict(
        repo_id=repository, revision=revision, path_in_repo=prefix
    )
