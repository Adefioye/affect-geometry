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
        assert args.batch_size == 8
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


def test_hf_token_loads_from_dotenv_without_overriding_environment(tmp_path, monkeypatch):
    from affect_geometry.classification import environment

    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / ".env").write_text("HF_TOKEN=from-dotenv\n")
    monkeypatch.setattr(environment, "DOTENV_PATH", tmp_path / ".env")
    assert environment.load_hf_token() == "from-dotenv"

    monkeypatch.setenv("HF_TOKEN", "from-runpod")
    assert environment.load_hf_token() == "from-runpod"


def test_dotenv_path_is_repository_root():
    from affect_geometry.classification.environment import DOTENV_PATH

    assert DOTENV_PATH.parent.name == "affect-geometry"
    assert (DOTENV_PATH.parent / "pyproject.toml").is_file()


def test_gated_models_require_hf_token():
    from affect_geometry.classification.environment import require_model_access

    for model in ("llama2", "gemma2"):
        with pytest.raises(RuntimeError, match="requires HF_TOKEN"):
            require_model_access(model, None)
        require_model_access(model, "secret")
    for model in ("apertus", "olmo2"):
        require_model_access(model, None)


def test_generator_passes_hf_token_to_model_and_tokenizer(monkeypatch):
    import sys
    from types import SimpleNamespace

    calls = {}

    class Tokenizer:
        chat_template = None
        pad_token_id = 0
        eos_token_id = 2

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["tokenizer"] = (model_id, kwargs)
            return Tokenizer()

    class Model:
        def eval(self):
            calls["evaluated"] = True

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["model"] = (model_id, kwargs)
            return Model()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=AutoModelForCausalLM,
            AutoTokenizer=AutoTokenizer,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))

    from affect_geometry.classification.inference import Generator

    Generator("example/model", "revision", False, token="secret")
    assert calls["tokenizer"][1]["token"] == "secret"
    assert calls["model"][1]["token"] == "secret"
    assert calls["evaluated"] is True


def test_batches_tail_resume_and_fallback(tmp_path):
    rows = [
        dict(
            sample_id=str(i),
            id=str(i),
            text="x",
            labels=[27],
            emotion="neutral",
            source_split="train",
        )
        for i in range(10)
    ]
    manifest = dict(model_id="fake", model_revision="x", run_id="x", seed=1, batch_size=8)

    class Batched:
        calls = []
        singles = []

        def batch(self, batch, seed):
            self.calls.append([r["sample_id"] for r in batch])
            return [dict(raw_response="neutral", generated_token_ids=[1]) for r in batch]

        def __call__(self, row, seed):
            self.singles.append(row["sample_id"])
            return dict(raw_response="neutral", generated_token_ids=[1])

    generator = Batched()
    result = classify(rows, generator, tmp_path / "normal", manifest)
    assert [len(x) for x in generator.calls] == [8, 2]
    assert all(r["correct"] for r in result)
    generator.calls.clear()
    classify(rows, generator, tmp_path / "normal", manifest)
    assert generator.calls == []

    class Failing(Batched):
        def batch(self, batch, seed):
            raise RuntimeError("simulated batch memory failure")

    failing = Failing()
    result = classify(rows, failing, tmp_path / "fallback", manifest)
    assert len(failing.singles) == 10
    assert all(r["status"] == "completed" for r in result)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_batch_config(tmp_path, value):
    import json

    from affect_geometry.classification.configuration import load_config

    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(model="gemma2", run_dir="outputs/test", batch_size=value)))
    with pytest.raises(ValueError, match="batch_size"):
        load_config(path)


def test_batched_decode_removes_eos_padding_and_counts_unpadded_input(monkeypatch):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace

    import numpy as np

    from affect_geometry.classification.inference import Generator

    class Tensor:
        def __init__(self, data):
            self.data = np.array(data)
            self.shape = self.data.shape

        def to(self, device):
            return self

        def sum(self, dim):
            return Tensor(self.data.sum(axis=dim))

        def tolist(self):
            return self.data.tolist()

        def __getitem__(self, key):
            return Tensor(self.data[key])

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            assert len(text) == 2 and kwargs["padding"] is True
            return dict(
                input_ids=Tensor([[0, 0, 1], [1, 1, 1]]),
                attention_mask=Tensor([[0, 0, 1], [1, 1, 1]]),
            )

        def decode(self, ids, **kwargs):
            return str(ids)

    def generate(**kwargs):
        assert kwargs["attention_mask"].tolist() == [[0, 0, 1], [1, 1, 1]]
        return Tensor([[0, 0, 1, 7, 2, 0], [1, 1, 1, 8, 9, 10]])

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    gen = Generator.__new__(Generator)
    gen.tokenizer = Tokenizer()
    gen.model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=100),
        device="cpu",
        generation_config=SimpleNamespace(eos_token_id=2),
        generate=generate,
    )
    gen.chat, gen.limit = False, 3
    result = gen.batch([dict(sample_id=str(i), text="x") for i in range(2)], 42)
    assert result[0]["generated_token_ids"] == [7, 2]
    assert result[0]["generation_finish_reason"] == "eos"
    assert result[0]["input_token_count"] == 1
    assert result[1]["generated_token_ids"] == [8, 9, 10]
    assert result[1]["generation_finish_reason"] == "max_new_tokens"
    assert result[1]["input_token_count"] == 3
