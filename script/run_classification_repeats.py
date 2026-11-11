"""Run, compare and publish two full EXP2 repeats using the original classifier."""

import argparse
import hashlib
import importlib.metadata
import json
import shlex
import subprocess
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from affect_geometry.classification.data import EXPECTED as SPLITS
from affect_geometry.classification.environment import load_hf_token

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/exp2_classification"
SOURCE = ROOT / "src/affect_geometry/classification"
BASELINES = {
    "gemma2": "cbe0b8e3dd8df548fac272246f70a291a2c74e1a",
    "olmo2": "04c26868cb2b16ab127d9182e8d06264e13dd698",
    "apertus": "d61972e561c168e9be9fb55b29208999acf2c00f",
}
RUNS = ("repeat-01", "repeat-02")
REPEAT_BATCH_SIZE = 16
MATCH_COLUMNS = ["sample_id", "source_split", "source_row", "text_hash", "emotion", "prompt_hash"]
OUTCOME_COLUMNS = ["status", "correct", "compat_correct", "inference_batch_size"]


def command(model, run, manifest):
    return [
        sys.executable,
        "-m",
        "affect_geometry.classification",
        "run",
        "--model",
        model,
        "--run-dir",
        f"outputs/exp2_classification/{model}/{run}",
        "--model-revision",
        manifest["model_revision"],
        "--dataset-revision",
        manifest["dataset_revision"],
        "--seed",
        str(manifest["seed"]),
        "--batch-size",
        str(REPEAT_BATCH_SIZE),
        "--dtype",
        manifest["dtype"],
        "--device",
        manifest["device"],
    ]


def check_manifest(baseline, actual, run):
    expected = dict(baseline, run_id=run, batch_size=REPEAT_BATCH_SIZE)
    differences = [key for key, value in expected.items() if actual.get(key) != value]
    if differences:
        raise ValueError(f"{run}: manifest mismatch: {differences}")


def load_split(files, split):
    import pandas as pd

    frame = pd.concat(
        [pd.read_parquet(file, columns=MATCH_COLUMNS + OUTCOME_COLUMNS) for file in files],
        ignore_index=True,
    )
    if (
        len(frame) != SPLITS[split]
        or frame[MATCH_COLUMNS].isna().any().any()
        or frame.sample_id.duplicated().any()
        or not frame.source_split.eq(split).all()
        or not frame.status.eq("completed").all()
        or not frame[["correct", "compat_correct"]].isin([True, False]).all().all()
    ):
        raise ValueError(f"Incomplete or invalid {split} data")
    return frame.sort_values("sample_id").reset_index(drop=True)


def check_batch_sizes(splits, run):
    # The classifier batches the full dataset in source order, across split boundaries.
    sizes = [
        size for split in splits for size in split.sort_values("source_row").inference_batch_size
    ]
    full_batches, tail = divmod(len(sizes), REPEAT_BATCH_SIZE)
    full_rows = full_batches * REPEAT_BATCH_SIZE
    if any(size != REPEAT_BATCH_SIZE for size in sizes[:full_rows]) or any(
        size != tail for size in sizes[full_rows:]
    ):
        raise ValueError(f"{run}: unexpected actual batch sizes for batch size {REPEAT_BATCH_SIZE}")


def compare_runs(model, frames):
    import pandas as pd

    original = frames["full"]
    pairs = original[["sample_id", "source_split", "emotion", "prompt_hash"]].copy()
    metrics = ("correct", "compat_correct")
    for run, frame in frames.items():
        for metric in metrics:
            pairs[f"{run}_{metric}"] = frame[metric]
    for metric in metrics:
        pairs[f"n_{metric}"] = pairs[[f"{run}_{metric}" for run in frames]].sum(axis=1)
    destination = OUTPUT / model / "repeat-comparison"
    destination.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(destination / "sample_outcomes.csv", index=False)

    summary = [
        {"run_id": run, "n": len(frame), **{metric: frame[metric].mean() for metric in metrics}}
        for run, frame in frames.items()
    ]
    pd.DataFrame(summary).to_csv(destination / "accuracy.csv", index=False)
    counts = {
        metric: pairs[f"n_{metric}"].value_counts().reindex(range(4), fill_value=0)
        for metric in metrics
    }
    pd.DataFrame(counts).rename_axis("n_correct_out_of_three").to_csv(
        destination / "correctness_counts.csv"
    )


def publish_repeats(api, repo, model, parent_commit):
    operations = []
    patterns = [
        "dataset/*.parquet",
        "manifest.json",
        "analyses/*",
        "plot_data/*",
        "plots/*",
        "source_audit.json",
        "duplicate_audit.json",
    ]
    for run in RUNS:
        local = OUTPUT / model / run
        paths = sorted(
            {path for pattern in patterns for path in local.glob(pattern) if path.is_file()}
        )
        required = {local / "dataset" / f"{split}.parquet" for split in SPLITS}
        required |= {local / "manifest.json", local / "analyses/summary.csv"}
        if not required <= set(paths):
            raise ValueError(f"Missing publication artifacts for {model}/{run}")
        operations.extend(
            CommitOperationAdd(
                path_in_repo=f"runs/{run}/{path.relative_to(local).as_posix()}",
                path_or_fileobj=path,
            )
            for path in paths
        )
    commit = api.create_commit(
        repo_id=repo,
        repo_type="dataset",
        parent_commit=parent_commit,
        operations=operations,
        commit_message="Add both EXP2 classification repeats",
    )
    uploaded = set(api.list_repo_files(repo, repo_type="dataset", revision=commit.oid))
    if not {operation.path_in_repo for operation in operations} <= uploaded:
        raise ValueError(f"Incomplete Hub commit: {commit.oid}")
    print(
        f"Verified upload: https://huggingface.co/datasets/{repo}/tree/{commit.oid}/runs",
        flush=True,
    )


def run_model(api, model, repo, revision, cache, baseline):
    import pandas as pd

    versions = {name: importlib.metadata.version(name) for name in baseline["versions"]}
    if versions != baseline["versions"]:
        raise ValueError(
            f"Package versions differ. Expected {baseline['versions']}; got {versions}"
        )
    code_hash = hashlib.sha256(
        "".join(path.read_text() for path in sorted(SOURCE.glob("*.py"))).encode()
    ).hexdigest()
    if code_hash != baseline["code_hash"]:
        raise ValueError("Classifier code differs from the published original")

    if not api.token:
        raise RuntimeError("Set HF_TOKEN for model access and dataset uploads")
    baseline_files = api.list_repo_files(repo, repo_type="dataset", revision=revision)
    current = api.repo_info(repo, repo_type="dataset")
    for run in RUNS:
        if any(file.rfilename.startswith(f"runs/{run}/") for file in current.siblings):
            raise ValueError(f"Destination already contains {repo}/runs/{run}")
    original = {}
    for split in SPLITS:
        files = [
            name
            for name in baseline_files
            if name.startswith(f"data/{split}-") and name.endswith(".parquet")
        ]
        original[split] = load_split(
            [
                hf_hub_download(
                    repo,
                    name,
                    revision=revision,
                    repo_type="dataset",
                    token=api.token,
                    local_dir=cache,
                )
                for name in files
            ],
            split,
        )

    frames = {"full": pd.concat(original.values(), ignore_index=True)}
    for run in RUNS:
        local = OUTPUT / model / run
        subprocess.run(command(model, run, baseline), cwd=ROOT, check=True)
        actual = json.loads((local / "manifest.json").read_text())
        check_manifest(baseline, actual, run)
        paired = []
        for split, reference in original.items():
            repeat = load_split([local / "dataset" / f"{split}.parquet"], split)
            pd.testing.assert_frame_equal(repeat[MATCH_COLUMNS], reference[MATCH_COLUMNS])
            paired.append(repeat)
        check_batch_sizes(paired, f"{model}/{run}")
        rows = pd.concat(paired, ignore_index=True)
        frames[run] = rows
        print(
            f"{model}/{run}: {len(rows)} verified prompts, strict {rows.correct.mean():.2%}, "
            f"relaxed {rows.compat_correct.mean():.2%}",
            flush=True,
        )

    compare_runs(model, frames)
    publish_repeats(api, repo, model, current.sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=BASELINES, action="append", help="Default: all three")
    parser.add_argument(
        "--plan", action="store_true", help="Print commands using baseline manifests"
    )
    args = parser.parse_args()
    api = HfApi(token=load_hf_token())
    for model in args.model or BASELINES:
        repo = f"kokolamba/exp2-goemotions-single-label-{model}"
        revision = BASELINES[model]
        cache = OUTPUT / model / "full/hf_snapshot" / revision
        manifest_path = cache / "manifest.json"
        if not manifest_path.is_file():
            hf_hub_download(
                repo,
                "manifest.json",
                revision=revision,
                repo_type="dataset",
                token=api.token,
                local_dir=cache,
            )
        baseline = json.loads(manifest_path.read_text())
        if args.plan:
            for run in RUNS:
                print(shlex.join(command(model, run, baseline)))
        else:
            run_model(api, model, repo, revision, cache, baseline)


if __name__ == "__main__":
    main()
