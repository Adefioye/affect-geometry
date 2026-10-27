"""Validate and publish source-split-aligned model datasets."""

import json
from pathlib import Path

from .data import EXPECTED
from .storage import atomic_json


def publish(root, repo_id, private=True):
    from datasets import DatasetDict, load_dataset
    from huggingface_hub import DatasetCard, HfApi, hf_hub_download

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["pilot_limit"]:
        raise ValueError("Pilot runs cannot be published as complete datasets")
    required = [
        "summary.csv",
        "confusion.csv",
        "label_metrics.csv",
        "macro_metrics.csv",
        "sample_outcomes.csv",
        "sample_indexes.json",
    ]
    for name in required:
        if not (root / "analyses" / name).is_file():
            raise ValueError(f"Missing local metric artifact: {name}; run report first")
    parts = DatasetDict(
        {
            s: load_dataset(
                "parquet", data_files=str(root / "dataset" / f"{s}.parquet"), split="train"
            )
            for s in EXPECTED
        }
    )
    for split, part in parts.items():
        if len(part) != EXPECTED[split] or len(set(part["sample_id"])) != len(part):
            raise ValueError("Incomplete or duplicate split")
        if any(r["status"] != "completed" or r["correct"] is None for r in part):
            raise ValueError("Unresolved inference errors")
    api = HfApi()
    if api.repo_exists(repo_id, repo_type="dataset"):
        # Fail closed if this is not a prior release of this exact run/protocol.
        previous_path = hf_hub_download(repo_id, "manifest.json", repo_type="dataset")
        previous = json.loads(Path(previous_path).read_text())
        if previous != manifest:
            raise ValueError("Destination contains a different run; use a new repository")
        actual_private = api.repo_info(repo_id, repo_type="dataset").private
        if actual_private != private:
            raise ValueError("Existing repository visibility differs from requested visibility")
    parts.push_to_hub(repo_id, private=private)
    card = (
        "---\nlicense: apache-2.0\n---\n# Single-label GoEmotions classifications\n\n"
        f"Model: `{manifest['model_id']}`.\n\n"
        "Derived from google-research-datasets/go_emotions (simplified). "
        "Original train/validation/test splits and neutral are retained; multilabel rows "
        "are excluded. Strict correctness requires a single canonical emotion matching the "
        "gold label; invalid outputs count as incorrect. Compatibility first-word results "
        "are diagnostic only. No model training or activation extraction is performed. "
        "These Reddit comments and model predictions may contain offensive content. "
        "Predictions are research annotations, not reliable judgments about individuals.\n\n"
        "See manifest.json for pinned source/model revisions and rendering settings.\n"
    )
    api = HfApi()
    existing = DatasetCard.load(repo_id)
    existing.text = card.split("---\n", 2)[-1]
    existing.data.license = "apache-2.0"
    existing.push_to_hub(repo_id)
    api.upload_file(
        path_or_fileobj=str(root / "manifest.json"),
        path_in_repo="manifest.json",
        repo_id=repo_id,
        repo_type="dataset",
    )
    api.upload_folder(
        folder_path=str(root / "analyses"),
        path_in_repo="analyses",
        repo_id=repo_id,
        repo_type="dataset",
    )
    for directory in ("plot_data", "plots"):
        if (root / directory).is_dir():
            api.upload_folder(
                folder_path=str(root / directory),
                path_in_repo=directory,
                repo_id=repo_id,
                repo_type="dataset",
            )
    for name in ("source_audit.json", "duplicate_audit.json"):
        if (root / name).is_file():
            api.upload_file(
                path_or_fileobj=str(root / name),
                path_in_repo=name,
                repo_id=repo_id,
                repo_type="dataset",
            )
    revision = api.repo_info(repo_id, repo_type="dataset").sha
    remote = load_dataset(repo_id, revision=revision)
    for split, part in parts.items():
        if part.features != remote[split].features or part.to_list() != remote[split].to_list():
            raise ValueError("Published round-trip mismatch")
    atomic_json(
        root / "publication.json", dict(repo_id=repo_id, revision=revision, private=private)
    )
