"""Scientific contracts and CPU integration checks; no checkpoint downloads."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from affect_geometry.analysis.valence_direction import fit_centroid_pc1
from affect_geometry.data.sentiment_transfer import make_pairs
from affect_geometry.evaluation.directional_patching import evaluate_pairs, summarize
from affect_geometry.interventions.directional import directional_replace
from affect_geometry.models.activation_patching import LastTokenPatcher


class Tokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True):
        if text in (" Negative", " Positive"):
            return [4 if text == " Negative" else 5]
        extra = [7, 8] if "long" in text else []
        return [1, *extra, 2 if "good" in text else 6, 3]

    def __call__(self, text, **kwargs):
        ids = torch.tensor([self.encode(text)])
        return dict(input_ids=ids, attention_mask=torch.ones_like(ids))


def test_centroid_pc1_native_dimensions_equal_weight_neutral_and_translation():
    labels = ["sad", "neutral", "joy"]
    x = np.array([[-2.0, 0, 1, 4], [0, 0, 1, 4], [2, 0, 1, 4]])
    fit = fit_centroid_pc1(x, labels, labels, [-1, -0.062, 1])
    assert fit.vector.shape == (4,) and fit.scores.shape == (3,)
    np.testing.assert_allclose(fit.scores, (x - x.mean(axis=0)) @ fit.vector)
    np.testing.assert_allclose(fit.scores, [-2, 0, 2])
    assert fit.vector[0] > 0.99 and fit.pearson_r > 0.99
    duplicate = fit_centroid_pc1(
        np.concatenate([x, x[2:]]), labels + ["joy"], labels, [-1, -0.062, 1]
    )
    translated = fit_centroid_pc1(x + [4, 3, 9, 7], labels, labels, [-1, -0.062, 1])
    np.testing.assert_allclose(fit.vector, duplicate.vector)
    np.testing.assert_allclose(fit.scores, translated.scores)
    with pytest.raises(ValueError):
        fit_centroid_pc1(x, labels, labels, [0, 0, 0])


def test_direction_replaces_coordinate_preserves_orthogonal_and_sign_invariance():
    torch.manual_seed(3)
    recipient, donor = torch.randn(2, 5), torch.randn(2, 5)
    d = torch.randn(5)
    d /= d.norm()
    edited = directional_replace(recipient, donor, d)
    torch.testing.assert_close(edited @ d, donor @ d)
    projection = torch.eye(5) - torch.outer(d, d)
    torch.testing.assert_close(edited @ projection, recipient @ projection)
    torch.testing.assert_close(edited, directional_replace(recipient, donor, -d))
    with pytest.raises(ValueError):
        directional_replace(recipient, donor, torch.zeros(5))


def candidate_frame():
    return pd.DataFrame(
        [
            dict(example_id="p", label=1, text="good", prompt="good"),
            dict(example_id="n", label=0, text="bad", prompt="bad"),
        ]
    )


def test_pairing_is_bidirectional_deterministic_and_shared():
    pairs, audit = make_pairs(candidate_frame(), Tokenizer())
    assert len(pairs) == 2 and len(audit) == 2
    assert set(pairs.recipient_label) == {0, 1}
    assert (pairs.recipient_label != pairs.donor_label).all()
    pd.testing.assert_frame_equal(pairs, make_pairs(candidate_frame(), Tokenizer())[0])
    with pytest.raises(ValueError):
        make_pairs(candidate_frame(), Tokenizer(), max_tokens=1)


@pytest.mark.parametrize("architecture", ["gemma2", "apertus", "olmo2"])
def test_real_tiny_architectures_post_block_capture_and_full_replacement(architecture):
    import transformers

    torch.manual_seed(4)
    classes = {
        "gemma2": ("Gemma2Config", "Gemma2ForCausalLM"),
        "apertus": ("ApertusConfig", "ApertusForCausalLM"),
        "olmo2": ("Olmo2Config", "Olmo2ForCausalLM"),
    }
    config_name, model_name = classes[architecture]
    config = getattr(transformers, config_name)(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    model = getattr(transformers, model_name)(config).eval()
    patcher = LastTokenPatcher(model, Tokenizer(), "cpu")
    donor_delta, donor = patcher.run("good", 1)
    recipient_delta, recipient = patcher.run("bad", 1)
    d = (donor - recipient).squeeze(0)
    patched, _ = patcher.run("bad", 1, donor=donor, direction=d)
    full, _ = patcher.run("bad", 1, donor=donor, full=True)
    assert full == pytest.approx(donor_delta, abs=1e-6)
    assert patched == pytest.approx(donor_delta, abs=1e-6)
    noop, _ = patcher.run("bad", 1, donor=recipient, direction=d)
    assert noop == pytest.approx(recipient_delta, abs=1e-6)
    assert not model.model.layers[1]._forward_hooks

    candidates = pd.concat(
        [
            candidate_frame(),
            candidate_frame().assign(
                example_id=lambda frame: "long-" + frame.example_id,
                prompt=lambda frame: "long " + frame.prompt,
            ),
        ],
        ignore_index=True,
    )
    pairs, _ = make_pairs(candidates, Tokenizer())
    single = evaluate_pairs(patcher, pairs, d.numpy(), 1, random_count=1, batch_size=1)
    forward_batch_sizes = []
    handle = model.register_forward_hook(
        lambda _model, _args, output: forward_batch_sizes.append(output.logits.shape[0])
    )
    try:
        batched = evaluate_pairs(patcher, pairs, d.numpy(), 1, random_count=1, batch_size=3)
    finally:
        handle.remove()
    assert forward_batch_sizes == [3] * 5 + [1] * 5
    pd.testing.assert_frame_equal(single, batched, atol=1e-5, rtol=1e-4)
    pd.testing.assert_frame_equal(summarize(single), summarize(batched), atol=1e-5, rtol=1e-4)


class PerfectPatcher:
    def encode_batch(self, prompts):
        return torch.tensor([2.0 if "good" in prompt else -2.0 for prompt in prompts])

    def run_batch(self, prompts, layer, donor=None, direction=None, full=False):
        if donor is not None:
            return donor[:, 0], donor
        return prompts, prompts[:, None]


def test_recovery_does_not_cancel_opposite_directions():
    pairs, _ = make_pairs(candidate_frame(), Tokenizer())
    records = evaluate_pairs(PerfectPatcher(), pairs, np.array([1.0, 0.0]), 0, random_count=1)
    metrics = summarize(records)
    pc1 = metrics.query("control == 'pc1' and subset == 'all_pairs'").iloc[0]
    noop = metrics.query("control == 'no_op' and subset == 'all_pairs'").iloc[0]
    assert pc1.recovery_percent == pytest.approx(100)
    assert pc1.sign_flip_percent == 100
    assert pc1.calibrated_logit_flip_percent == 100
    assert noop.recovery_percent == 0 and noop.sign_flip_percent == 0


def test_end_to_end_selection_and_evaluation(tmp_path, monkeypatch):
    from affect_geometry.activations import valence_source
    from affect_geometry.activations.exp3_bundle import COHORT_NAMES as INPUT_COHORTS
    from affect_geometry.experiments import valence_transfer as experiment

    labels = ["sadness", "neutral", "joy"]
    nrc = tmp_path / "nrc.csv"
    pd.DataFrame(dict(emotion=labels, valence=[-1, -0.062, 1])).to_csv(nrc, index=False)
    cfg = dict(
        project_root=str(tmp_path),
        output_dir=str(tmp_path / "outputs"),
        nrc_path=str(nrc),
        seed=42,
        max_tokens=20,
        random_directions=1,
        models={
            m: {
                "activation_source": dict(
                    repo_id=f"kokolamba/exp3-{m}",
                    revision="a" * 40,
                    path_in_repo="activations/fixture",
                )
            }
            for m in experiment.MODELS
        },
        datasets=[dict(name=name, split="test") for name in sorted(experiment.DATASETS)],
    )
    expected_layers = {"correct": 0, "incorrect": 1}
    frames = pd.DataFrame([dict(cohort=c, emotion=e) for c in INPUT_COHORTS for e in labels])

    def bundle_open(path, expected_model):
        root = tmp_path / expected_model
        root.mkdir()
        (root / "manifest.json").write_text("{}")

        def projected(cohort, layer):
            states = np.zeros((3, 16))
            axis = 2 * list(experiment.MODELS).index(expected_model) + layer
            # Both correct layers tie. Only layer 1 aligns for the incorrect cohort.
            states[:, axis] = [-1, -0.062, 1] if cohort == "correct" or layer == 1 else [1, -2, 1]
            return states

        return SimpleNamespace(
            root=root,
            cohorts=SimpleNamespace(samples=frames),
            manifest=dict(
                model_id=experiment.MODELS[expected_model],
                projected_states_are_unit_normalized=False,
                prompt_metadata=dict(layer_indexing="zero_based_post_block"),
                emotion_order=labels,
                source=dict(model_revision="fixture"),
                n_layers=2,
            ),
            projected=projected,
        )

    from affect_geometry.activations.valence_source import PreparedActivations

    monkeypatch.setattr(PreparedActivations, "open", bundle_open)
    monkeypatch.setattr(
        valence_source, "download_bundle", lambda source, destination: (destination, dict(source))
    )
    experiment.select_directions(cfg)
    selection = pd.read_csv(tmp_path / "outputs/directions/selection.csv")
    assert len(selection) == 6
    for _, selected in selection.groupby("model"):
        assert selected.set_index("cohort").layer.to_dict() == expected_layers
    assert not list((tmp_path / "outputs/directions").glob("*/all"))
    manifest = json.loads((tmp_path / "outputs/directions/manifest.json").read_text())
    for name in experiment.MODELS:
        assert manifest["sources"][name]["activation_repo_id"] == f"kokolamba/exp3-{name}"
        assert manifest["sources"][name]["activation_revision"] == "a" * 40

    patched = []
    model_index = None

    def loader(config):
        from transformers import Olmo2Config, Olmo2ForCausalLM

        nonlocal model_index
        model_index = list(experiment.MODELS.values()).index(config.name)
        tiny = Olmo2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        tiny._attn_implementation = "eager"
        return Olmo2ForCausalLM(tiny).eval(), Tokenizer(), "cpu"

    def check_patching_direction(patcher, pairs, vector, layer, **kwargs):
        expected_vector = np.eye(16)[2 * model_index + layer]
        np.testing.assert_allclose(vector, expected_vector)
        assert kwargs["batch_size"] == 32
        patched.append((model_index, layer))
        return evaluate_pairs(patcher, pairs, vector, layer, **kwargs)

    monkeypatch.setattr(experiment, "evaluate_pairs", check_patching_direction)
    eval_cfg = dict(cfg, batch_size=32)
    experiment.evaluate_frozen(
        eval_cfg, model_loader=loader, candidate_loader=lambda spec: candidate_frame()
    )
    result = tmp_path / "outputs/evaluation"
    evaluation = json.loads((result / "manifest.json").read_text())
    assert evaluation["status"] == "completed" and evaluation["batch_size"] == 32
    assert sorted(patched) == [
        (model, layer) for model in range(3) for layer in (0, 1) for _ in experiment.DATASETS
    ]
    assert len(list(result.glob("*/*/*_metrics.csv"))) == 24
    for path in result.glob("*/*/*_metrics.csv"):
        cohort = path.stem.removesuffix("_metrics")
        assert set(pd.read_csv(path).layer) == {expected_layers[cohort]}
    assert len(list(result.glob("*/*/pairs.csv"))) == 12
    with pytest.raises(FileExistsError):
        experiment.evaluate_frozen(cfg, model_loader=loader)


def test_selection_reads_actual_checksummed_exp3_bundle(tmp_path):
    from affect_geometry.activations.exp3_bundle import export_cohort_bundle
    from affect_geometry.analysis.cohorts import balanced_cohorts
    from affect_geometry.experiments.valence_transfer import MODELS, select_directions
    from affect_geometry.persistence.correctness_geometry import checksum

    rows = pd.DataFrame(
        [
            dict(
                sample_id=f"{emotion}-{correct}-{i:03d}",
                emotion=emotion,
                compat_correct=correct,
                status="completed",
                source_split="train",
                prompt_hash="fixture",
                text="example",
                text_length=7,
                compat_valid_label=True,
                correct=correct,
                valid_label=True,
            )
            for emotion in ("sadness", "neutral", "joy")
            for correct, count in ((True, 102), (False, 10))
            for i in range(count)
        ]
    )
    cohorts = balanced_cohorts(rows, ("sadness", "neutral", "joy"), 42)
    rng = np.random.default_rng(4)
    states = rng.normal(size=(len(rows), 16))
    states[:, 0] += rows.emotion.map({"sadness": -5, "neutral": 0, "joy": 5}).to_numpy()
    layer = tmp_path / "layer.npy"
    np.save(layer, states)
    nrc = tmp_path / "nrc.csv"
    pd.DataFrame(dict(emotion=["sadness", "neutral", "joy"], valence=[-1, -0.062, 1])).to_csv(
        nrc, index=False
    )
    models = {}
    for name, model_id in MODELS.items():
        root = tmp_path / name
        export_cohort_bundle(
            root,
            model=name,
            run_id="fixture",
            cohorts=cohorts,
            source_manifest=dict(model_id=model_id, model_revision="fixture"),
            source_audit={},
            prompt_samples=rows,
            layer_paths=[layer],
            prompt_metadata=dict(hidden_size=16, layer_indexing="zero_based_post_block"),
            prompt_checksums={layer.name: checksum(layer)},
            seed=42,
        )
        models[name] = dict(bundle=str(root))
    cfg = dict(
        models=models,
        output_dir=str(tmp_path / "outputs"),
        project_root=str(tmp_path),
        nrc_path=str(nrc),
    )
    select_directions(cfg)
    selection = pd.read_csv(tmp_path / "outputs/directions/selection.csv")
    assert len(selection) == 6 and set(selection.status) == {"ok"}
    assert set(selection.cohort) == {"correct", "incorrect"}
    assert (selection.n_emotions == 3).all()


def test_pinned_activation_download_and_provenance(tmp_path, monkeypatch):
    from affect_geometry.activations import valence_source as source

    remote = dict(
        repo_id="kokolamba/rq3-olmo2",
        revision="a" * 40,
        path_in_repo="activations/rq3-activations-bs8-v1",
    )
    settings = dict(activation_source=remote)
    bundle = SimpleNamespace(root=tmp_path / "bundle")
    calls = []

    def download(spec, destination):
        calls.append((spec, destination))
        return bundle.root, dict(remote)

    def validate(path, expected_model):
        assert path == bundle.root and expected_model == "olmo2"
        return bundle

    monkeypatch.setattr(source, "download_bundle", download)
    monkeypatch.setattr(source.PreparedActivations, "open", validate)
    actual, provenance = source.open_source(settings, "olmo2", tmp_path)
    assert actual is bundle and provenance == remote
    assert calls == [(remote, tmp_path / "outputs/valence-transfer/activation-cache/olmo2")]
    monkeypatch.setattr(
        source, "download_bundle", lambda *args: (bundle.root, dict(remote, revision="b" * 40))
    )
    with pytest.raises(ValueError, match="differs"):
        source.open_source(settings, "olmo2", tmp_path)


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"bundle": "local", "activation_source": {"repo_id": "repo"}},
        {"activation_source": {"revision": "main"}},
    ],
)
def test_activation_source_rejects_ambiguous_or_unpinned_settings(settings):
    from affect_geometry.activations.valence_source import validate_source

    with pytest.raises(ValueError):
        validate_source(settings)


def test_invalid_local_activation_bundle_does_not_download(tmp_path, monkeypatch):
    from affect_geometry.activations import valence_source as source

    def unexpected(*args, **kwargs):
        pytest.fail("Local source must not fall back to a remote bundle")

    monkeypatch.setattr(source, "download_bundle", unexpected)
    with pytest.raises((FileNotFoundError, ValueError)):
        source.open_source(dict(bundle=str(tmp_path / "missing")), "gemma2", tmp_path)


def test_three_metrics_distinguish_calibrated_flips_from_literal_flips():
    # Biased logits: condition-wise centering yields 50% calibrated recovery,
    # and two of four cases literally cross zero toward the donor.
    donor = np.array([5.0, 7.0, -1.0, -3.0])
    recipient = np.array([-1.0, -3.0, 5.0, 7.0])
    patched = np.array([3.0, 1.0, 1.0, 3.0])
    sign = np.array([1.0, 1.0, -1.0, -1.0])
    denominator = sign * (donor - recipient)
    records = pd.DataFrame(
        dict(
            control=["pc1"] * 4,
            pair_id=[0, 1, 0, 1],
            donor_label=[1, 1, 0, 0],
            donor_logit_diff=donor,
            recipient_logit_diff=recipient,
            patched_logit_diff=patched,
            donor_margin=sign * donor,
            recipient_margin=sign * recipient,
            patched_margin=sign * patched,
            denominator=denominator,
            recovery_percent=100 * sign * (patched - recipient) / denominator,
            target_sign_flip=(recipient * patched < 0) & (sign * patched > 0),
            both_baselines_correct=[True] * 4,
            recipient_correct=[True] * 4,
            donor_correct=[True] * 4,
            patched_target_correct=sign * patched > 0,
        )
    )
    row = summarize(records).query("subset == 'all_pairs'").iloc[0]
    assert row.recovery_percent == pytest.approx(50)
    assert row.calibrated_logit_flip_percent == pytest.approx(50)
    assert row.sign_flip_percent == pytest.approx(50)
    # Translation preserves both centered metrics but changes literal flips.
    for column in ("donor_logit_diff", "recipient_logit_diff", "patched_logit_diff"):
        records[column] += 10
    records["target_sign_flip"] = False
    row = summarize(records).query("subset == 'all_pairs'").iloc[0]
    assert row.calibrated_logit_flip_percent == pytest.approx(50)
    assert row.sign_flip_percent == 0


def test_model_and_tokenizer_load_token_in_fresh_process(monkeypatch):
    import transformers

    from affect_geometry.config import ModelConfig
    from affect_geometry.models import huggingface

    calls = []
    tokenizer = SimpleNamespace(pad_token_id=0)
    model = SimpleNamespace(to=lambda device: None, eval=lambda: None)

    def load_tokenizer(name, **kwargs):
        calls.append(kwargs)
        return tokenizer

    def load_model(name, **kwargs):
        calls.append(kwargs)
        return model

    monkeypatch.setattr(huggingface, "load_hf_token", lambda: "test-token")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load_model)
    huggingface.load_causal_lm(ModelConfig(name="fixture", revision="pinned", device="cpu"))
    assert len(calls) == 2
    assert all(call["token"] == "test-token" and call["revision"] == "pinned" for call in calls)
