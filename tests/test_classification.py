import pytest

from affect_geometry.classification.pipeline import classify
from affect_geometry.classification.protocol import parse_compat, parse_strict, prompt, score
from affect_geometry.classification.reporting import report, summarize
from affect_geometry.classification.storage import Ledger


@pytest.mark.parametrize(
    "raw,strict,compat",
    [
        ("joy", "joy", "joy"),
        ("Joy!", "joy", "joy"),
        ("**gratitude**", "gratitude", "gratitude"),
        ("neutral", "neutral", "neutral"),
        ("joy because happy", None, "joy"),
        ("joy or excitement", None, "joy"),
        ("joy, sadness", None, "joy"),
        ("The emotion is joy", None, None),
        ("Emotion: joy", None, None),
        ('"joy"', None, None),
        ('{"emotion":"joy"}', None, None),
        ("joy/sadness", None, None),
        ("joy:", None, None),
        ("happy", None, None),
        ("grat", None, None),
        (" \n", None, None),
    ],
)
def test_parsers(raw, strict, compat):
    assert parse_strict(raw) == strict
    assert parse_compat(raw) == compat


def test_gold_and_prompt():
    assert score("joy", "sadness")["outcome"] == "incorrect_label"
    row = {"sample_id": "x", "text": "Thank you"}
    assert prompt(row) == prompt(row)
    assert len(set(prompt(row)[1])) == 28


def test_denominators():
    rows = [
        dict(status="completed", **score(x, "joy"))
        for x in ["joy", "sadness", "The emotion is joy"]
    ]
    rows.append(dict(status="error"))
    result = summarize(rows, "all", "joy")
    assert result["coverage_pct"] == 75
    assert result["n_correct"] == 1 and result["n_incorrect"] == 2
    assert result["valid_only_accuracy_pct"] == 50
    assert summarize([], "all", "grief")["correct_pct"] is None


def test_resume_errors_snapshot_and_reports(tmp_path):
    import pyarrow.parquet as pq

    rows = [
        dict(
            sample_id=str(i),
            id=str(i),
            text="example",
            labels=[17],
            label_id=17,
            emotion="joy",
            source_split="train",
        )
        for i in range(3)
    ]
    manifest = dict(model_id="fake", model_revision="sha", run_id="test", seed=42)

    def generate(row, seed):
        if row["sample_id"] == "0":
            raise RuntimeError("temporary failure")
        return dict(
            raw_response="joy" if row["sample_id"] == "1" else "joy because happy",
            generated_token_ids=[1],
        )

    result = classify(rows, generate, tmp_path, manifest, retries=0)
    assert result[0]["correct"] is None
    table = pq.read_table(tmp_path / "dataset/train.parquet")
    assert table["correct"].to_pylist() == [None, True, False]
    assert table["raw_response"].to_pylist()[1] == "joy"
    report(tmp_path, result, "fake", "test")
    assert (tmp_path / "plots/correctness.png").exists()
    calls = []

    def recovery(row, seed):
        calls.append(row["sample_id"])
        return dict(raw_response="neutral", generated_token_ids=[2])

    result = classify(rows, recovery, tmp_path, manifest)
    assert calls == ["0"]
    assert result[0]["outcome"] == "incorrect_label"
    assert result[0]["attempt_count"] == 2
    assert len(result) == 3
    with pytest.raises(ValueError, match="configuration differs"):
        Ledger(tmp_path, dict(manifest, seed=99))


def test_source_preservation_and_filter():
    from types import SimpleNamespace

    from affect_geometry.classification.data import prepare
    from affect_geometry.classification.protocol import EMOTIONS

    class Part(list):
        features = {"labels": SimpleNamespace(feature=SimpleNamespace(names=EMOTIONS))}

    data = {
        split: Part(
            [
                dict(id="a", text="x", labels=[27], extra="keep"),
                dict(id="b", text="y", labels=[1, 2]),
            ]
        )
        for split in ["train", "validation", "test"]
    }
    rows, audit = prepare(data, "fixture")
    assert len(rows) == 3 and len({r["sample_id"] for r in rows}) == 3
    assert all(r["emotion"] == "neutral" and r["extra"] == "keep" for r in rows)
    assert all(a["excluded"] == 1 for a in audit)


def test_all_errors_have_nullable_boolean_snapshot(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        dict(sample_id="a", id="a", text="x", labels=[27], emotion="neutral", source_split="test")
    ]

    def failure(row, seed):
        raise RuntimeError("offline")

    classify(
        rows,
        failure,
        tmp_path,
        dict(model_id="fake", model_revision="x", run_id="x", seed=1),
        retries=0,
    )
    table = pq.read_table(tmp_path / "dataset/test.parquet")
    assert table.schema.field("correct").type == pa.bool_()
    assert table["correct"].to_pylist() == [None]


def test_paired_comparison_checks_population(tmp_path):
    import pandas as pd

    from affect_geometry.classification.reporting import compare

    rows = [
        dict(sample_id="a", id="a", text="x", labels=[27], emotion="neutral", source_split="test")
    ]
    dirs = []
    for i, answer in enumerate(["neutral", "joy"]):
        root = tmp_path / str(i)
        manifest = dict(model_id=str(i), model_revision="x", run_id=str(i), seed=1)
        result = classify(
            rows,
            lambda row, seed: dict(raw_response=answer, generated_token_ids=[1]),
            root,
            manifest,
        )
        report(root, result, str(i), str(i))
        dirs.append(root)
    compare(dirs, tmp_path / "comparison", bootstrap_samples=10)
    frame = pd.read_csv(tmp_path / "comparison/paired_model_comparisons.csv")
    assert frame.iloc[0].delta_correct_percentage_points == 100
    import pyarrow as pa
    import pyarrow.parquet as pq

    record = pq.read_table(dirs[1] / "dataset/test.parquet").to_pylist()[0]
    record["sample_id"] = "different"
    pq.write_table(pa.Table.from_pylist([record]), dirs[1] / "dataset/test.parquet")
    with pytest.raises(ValueError, match="different source populations"):
        compare(dirs, tmp_path / "bad")


def test_single_model_configs_and_cli_overrides():
    from pathlib import Path

    from affect_geometry.classification.__main__ import parse_args

    root = Path(__file__).resolve().parents[1]
    for model in ["llama2", "gemma2", "apertus", "olmo2"]:
        config = root / "configs" / "classification" / f"{model}.json"
        args = parse_args(["run", "--config", str(config)])
        assert args.model == model
        assert str(args.run_dir) == f"outputs/rq2_classification/{model}/full"
        assert args.pilot_limit == 0
        override = parse_args(
            ["run", "--config", str(config), "--pilot-limit", "8", "--run-dir", "outputs/pilot"]
        )
        assert override.pilot_limit == 8 and str(override.run_dir) == "outputs/pilot"


def test_reject_multi_model_config(tmp_path):
    import json

    from affect_geometry.classification.configuration import load_config

    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"models": ["gemma2", "llama2"], "run_dir": "outputs/bad"}))
    with pytest.raises(ValueError):
        load_config(path)
