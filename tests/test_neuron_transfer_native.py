"""Scientific hook, gradient, score, and generation contracts on tiny native architectures."""

from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Olmo2Config, Olmo2ForCausalLM

from affect_geometry.classification.protocol import EMOTIONS, parse_strict
from affect_geometry.evaluation.emotion_labels import EmotionLabelScorer
from affect_geometry.evaluation.neuron_transfer import NativeNeuronRunner
from affect_geometry.interventions.neuron_patching import NeuronPatch
from affect_geometry.models.mlp_neurons import NativeMLPNeurons
from affect_geometry.neurons.attribution import attribute


class TinyTokenizer:
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2
    all_special_ids = [0, 1, 2]

    def encode(self, text, add_special_tokens=False):
        if text.strip() in EMOTIONS:
            label = text.strip()
            if label == "disapproval":
                return [60, 61]
            if label == "disappointment":
                return [60, 62]
            return [10 + EMOTIONS.index(label)]
        return ([1] if add_special_tokens else []) + [80 + ord(c) % 30 for c in text]

    def decode(self, ids, skip_special_tokens=True, **_):
        ids = [int(v) for v in ids if not skip_special_tokens or v not in self.all_special_ids]
        words = {10 + i: " " + label for i, label in enumerate(EMOTIONS)}
        words.update(
            {0: "<pad>", 1: "<s>", 2: "</s>", 60: " dis", 61: "approval", 62: "appointment"}
        )
        return "".join(words.get(v, chr(65 + v % 26)) for v in ids)


@pytest.fixture(params=["llama", "olmo2"])
def native(request):
    torch.manual_seed(42)
    config_class, model_class = (
        (LlamaConfig, LlamaForCausalLM)
        if request.param == "llama"
        else (Olmo2Config, Olmo2ForCausalLM)
    )
    model = model_class(
        config_class(
            vocab_size=128,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=1024,
            eos_token_id=2,
            pad_token_id=0,
            bos_token_id=1,
        )
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = TinyTokenizer()
    scorer = EmotionLabelScorer(model, tokenizer, batch_size=4)
    return model, tokenizer, scorer, NativeMLPNeurons(model)


def test_complete_paths_match_native_log_probabilities_and_batching(native):
    model, tokenizer, scorer, _ = native
    prompt = [1, 80, 81]
    scores, _ = scorer.score(prompt)
    assert len(scores) == 28
    for label in ("joy", "disapproval", "disappointment"):
        path = list(scorer.paths[label].ids)
        ids = torch.tensor([prompt + path[:-1]])
        with torch.no_grad():
            logits = model(ids, use_cache=False).logits[0].float().log_softmax(-1)
        values = [float(logits[len(prompt) - 1 + i, token]) for i, token in enumerate(path)]
        assert scores[label]["total"] == pytest.approx(sum(values), abs=2e-6)
        assert scores[label]["stop_score"] == pytest.approx(values[-1])
        assert parse_strict(tokenizer.decode(scorer.paths[label].label_ids)) == label
    scorer.batch_size = 1
    sequential, _ = scorer.score(prompt)
    assert all(
        scores[c]["total"] == pytest.approx(sequential[c]["total"], abs=2e-6) for c in EMOTIONS
    )


def test_true_gradient_and_signed_finite_difference(native):
    _, _, scorer, adapter = native
    prompt = [1, 80, 81, 82]
    totals = {}
    rows = attribute(adapter, scorer, prompt, "disapproval", top_k=3, totals=totals)
    assert len(rows) == 6 and set(totals) == {0, 1}
    assert all(len(v) == 24 for v in totals.values())
    chosen = max(rows, key=lambda r: r["importance"])
    key = chosen["layer"], chosen["channel"]
    epsilon = 0.0001
    scores = []
    for sign in (-1, 1):

        def factory(pos):
            return NeuronPatch(adapter, [key], {}, pos, offsets={key: sign * epsilon})

        result, _ = scorer.score(prompt, ["disapproval"], factory)
        scores.append(result["disapproval"]["total"])
    measured = (scores[1] - scores[0]) / (2 * epsilon)
    assert measured == pytest.approx(chosen["gradient"], rel=0.1, abs=1e-3)
    assert chosen["signed_contribution"] == pytest.approx(
        chosen["activation"] * chosen["gradient"], rel=1e-5
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_runner_finite_difference_resolves_native_precision(native, dtype):
    model, tokenizer, _, _ = native
    model.to(dtype)
    runner = NativeNeuronRunner(
        SimpleNamespace(stop_token_id=None, label_prefix=" ", score_batch_size=4),
        dict(chat=False, max_new_tokens=10),
        model=model,
        tokenizer=tokenizer,
    )
    info = dict(ids=[1, 80, 81, 82])
    rows = attribute(runner.adapter, runner.scorer, info["ids"], "disapproval", top_k=3)
    chosen = max(rows, key=lambda row: row["importance"])
    result = runner.finite_difference(info, "disapproval", chosen)
    assert len(result["trials"]) == 4
    assert result["gradient"] * result["finite_difference"] > 0
    assert abs(result["gradient"] - result["finite_difference"]) <= result["tolerance"]


def test_noop_self_donor_and_scoring_branches(native):
    model, _, scorer, adapter = native
    prompt = [1, 80, 81]
    members = [(0, 2), (1, 3)]
    with torch.no_grad(), adapter.capture([2], members) as captured:
        model(torch.tensor([prompt]), use_cache=False)
    donor = {(0, 2): float(captured[0][0, 0]), (1, 3): float(captured[1][0, 0])}
    baseline, _ = scorer.score(prompt)
    for alpha in (0, 1):
        patched, audit = scorer.score(
            prompt, patch_factory=lambda pos: NeuronPatch(adapter, members, donor, pos, alpha)
        )
        assert max(abs(patched[c]["total"] - baseline[c]["total"]) for c in EMOTIONS) < 2e-6
        assert len(audit) == 28 * len(members)
        assert all(r["position"] == 2 and r["norm"] < 1e-7 for r in audit)
    # Every candidate branch gets the prompt edit, including shared multi-token labels.
    zero = {(0, 2): 0, (1, 3): 0}
    edited, _ = scorer.score(
        prompt, patch_factory=lambda pos: NeuronPatch(adapter, members, zero, pos, alpha=1)
    )
    assert any(abs(edited[c]["total"] - baseline[c]["total"]) > 1e-7 for c in EMOTIONS)


def test_padding_one_token_prefill_decode_and_exception_cleanup(native):
    model, _, _, adapter = native
    members = [(0, 2), (1, 3)]
    donor = {(0, 2): 1.0, (1, 3): -1.0}
    # Explicit first-forward tracking, including a one-token prefill.
    with torch.no_grad(), NeuronPatch(adapter, members, donor, [0], 1) as patch:
        first = model(torch.tensor([[1]]), use_cache=True)
        model(torch.tensor([[80]]), past_key_values=first.past_key_values, use_cache=True)
    assert patch.forward_count == 2
    assert len(patch.audit) == 2
    with torch.no_grad(), NeuronPatch(adapter, members, donor, [2, 2], 1) as padded:
        model(
            torch.tensor([[0, 0, 1], [1, 80, 81]]),
            attention_mask=torch.tensor([[0, 0, 1], [1, 1, 1]]),
            use_cache=False,
        )
    assert len(padded.audit) == 4
    before = [len(m._forward_pre_hooks) for m in adapter.modules.values()]
    with pytest.raises(RuntimeError), NeuronPatch(adapter, members, donor, [0]):
        raise RuntimeError("deliberate failure")
    assert [len(m._forward_pre_hooks) for m in adapter.modules.values()] == before
    assert not model._forward_pre_hooks


def test_multilayer_patch_uses_current_recipient_and_preserves_other_coordinates(native):
    model, _, _, adapter = native
    ids = torch.tensor([[1, 80, 81]])
    early, later = (0, 2), (1, 3)
    donor, alpha = {early: 10.0, later: -2.0}, 0.25
    with torch.no_grad():
        with adapter.capture() as baseline:
            model(ids, use_cache=False)
        with NeuronPatch(adapter, [early], donor, [2], alpha), adapter.capture() as upstream:
            model(ids, use_cache=False)
        with NeuronPatch(adapter, [early, later], donor, [2], alpha), adapter.capture() as both:
            model(ids, use_cache=False)
    assert abs(float(upstream[1][0, 2, 3] - baseline[1][0, 2, 3])) > 1e-7
    expected = upstream[1].clone()
    expected[0, 2, 3] = (1 - alpha) * upstream[1][0, 2, 3] + alpha * donor[later]
    assert torch.equal(both[1], expected)
    expected_early = baseline[0].clone()
    expected_early[0, 2, 2] = (1 - alpha) * baseline[0][0, 2, 2] + alpha * donor[early]
    assert torch.equal(both[0], expected_early)


def test_generation_uses_edited_prefill_and_hooks_do_not_leak(native):
    model, _, _, adapter = native
    ids = torch.tensor([[1, 80, 81]])
    kwargs = dict(
        attention_mask=torch.ones_like(ids),
        do_sample=False,
        num_beams=1,
        max_new_tokens=3,
        pad_token_id=0,
    )
    with torch.no_grad():
        baseline = model.generate(ids, **kwargs)
        with NeuronPatch(adapter, [(0, 2)], {(0, 2): 100.0}, [2], 1):
            expected_logits = model(
                ids, attention_mask=kwargs["attention_mask"], use_cache=True
            ).logits[:, -1]
        with NeuronPatch(adapter, [(0, 2)], {(0, 2): 100.0}, [2], 1) as patch:
            edited = model.generate(ids, **kwargs, return_dict_in_generate=True, output_scores=True)
        restored = model.generate(ids, **kwargs)
    assert torch.equal(baseline, restored)
    assert torch.equal(edited.scores[0], expected_logits)
    assert torch.equal(edited.sequences[:, ids.shape[1]], expected_logits.argmax(-1))
    assert edited.sequences.shape[0] == 1 and len(patch.audit) == 1
    assert patch.forward_count >= 1


def test_adapter_rejects_unknown_architecture(native):
    model, _, _, _ = native
    model.config.model_type = "unsupported"
    with pytest.raises(ValueError, match="No validated"):
        NativeMLPNeurons(model)
