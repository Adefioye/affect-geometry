import argparse
import importlib.metadata
import json
from pathlib import Path

from .data import EXPECTED, SOURCE, prepare
from .protocol import MODELS, PROTOCOL, digest
from .storage import atomic_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Independent single-label emotion classification")
    parser.add_argument("command", choices=["run", "report", "publish", "compare"])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--config", type=Path, help="Single-model JSON config; CLI flags override")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--compare-runs", nargs="+", type=Path)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot-limit", type=int, default=0, help="Training-only pilot size")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--repo-id")
    parser.add_argument("--public", action="store_true")
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        from .configuration import load_config

        try:
            parser.set_defaults(**load_config(preliminary.config))
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    args = parser.parse_args(argv)
    if args.run_dir is None:
        parser.error("--run-dir or a config containing run_dir is required")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def main():
    args = parse_args()
    parser = argparse.ArgumentParser()
    if args.command == "compare":
        if not args.compare_runs or len(args.compare_runs) < 2:
            parser.error("--compare-runs requires at least two completed run directories")
        from .reporting import compare

        compare(args.compare_runs, args.run_dir, args.seed)
        return
    if args.command == "publish":
        if not args.repo_id:
            parser.error("--repo-id is required")
        from .publish import publish

        publish(args.run_dir, args.repo_id, private=not args.public)
        return
    from .reporting import report

    if args.command == "report":
        import pyarrow.parquet as pq

        manifest = json.loads((args.run_dir / "manifest.json").read_text())
        rows = [
            r
            for s in EXPECTED
            for r in pq.read_table(args.run_dir / "dataset" / f"{s}.parquet").to_pylist()
        ]
        report(args.run_dir, rows, manifest["model_id"], manifest["run_id"])
        return
    if not args.model or args.pilot_limit < 0:
        parser.error("--model is required and pilot limit must be nonnegative")
    from .environment import load_hf_token, require_model_access

    hf_token = load_hf_token()
    require_model_access(args.model, hf_token)
    from datasets import load_dataset
    from huggingface_hub import HfApi

    from affect_geometry.logging import configure_file_logging

    from .inference import Generator
    from .pipeline import classify

    model_id, chat = MODELS[args.model]
    api = HfApi(token=hf_token)
    previous_path = args.run_dir / "manifest.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        if args.model_revision == "main":
            args.model_revision = previous["model_revision"]
        if args.dataset_revision == "main":
            args.dataset_revision = previous["dataset_revision"]
    model_revision = api.model_info(model_id, revision=args.model_revision).sha
    dataset_id = "google-research-datasets/go_emotions"
    dataset_revision = api.dataset_info(dataset_id, revision=args.dataset_revision).sha
    dataset = load_dataset(dataset_id, "simplified", revision=dataset_revision, token=hf_token)
    rows, audit = prepare(dataset, f"{dataset_id}@{dataset_revision}")
    for item in audit:
        if item["source"] != SOURCE[item["split"]] or item["retained"] != EXPECTED[item["split"]]:
            raise ValueError("Pinned dataset counts differ from protocol")
    if args.pilot_limit:
        rows = [r for r in rows if r["source_split"] == "train"][: args.pilot_limit]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    configure_file_logging(Path("logs") / f"{args.model}-{args.run_dir.name}", "classify")
    generator = Generator(
        model_id,
        model_revision,
        chat,
        args.dtype,
        args.device,
        token=hf_token,
    )
    source_dir = Path(__file__).parent
    code_hash = digest("".join(p.read_text() for p in sorted(source_dir.glob("*.py"))))
    manifest = dict(
        model_id=model_id,
        model_revision=model_revision,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        seed=args.seed,
        batch_size=args.batch_size,
        run_id=args.run_dir.name,
        pilot_limit=args.pilot_limit,
        dtype=args.dtype,
        device=args.device,
        protocol=PROTOCOL,
        chat=chat,
        chat_template=generator.template,
        max_new_tokens=10,
        do_sample=False,
        num_beams=1,
        code_hash=code_hash,
        sample_ids_hash=digest("\n".join(r["sample_id"] for r in rows)),
        versions={
            p: importlib.metadata.version(p)
            for p in ["torch", "transformers", "datasets", "pyarrow", "huggingface-hub"]
        },
    )
    atomic_json(args.run_dir / "source_audit.json", audit)
    from collections import Counter

    duplicates = Counter(r["text_hash"] for r in rows)
    atomic_json(
        args.run_dir / "duplicate_audit.json", {h: n for h, n in duplicates.items() if n > 1}
    )
    result = classify(rows, generator, args.run_dir, manifest)
    report(args.run_dir, result, model_id, args.run_dir.name)
    if any(r["status"] != "completed" for r in result):
        raise SystemExit("Partial run: unresolved errors retained; rerun to retry")


if __name__ == "__main__":
    main()
