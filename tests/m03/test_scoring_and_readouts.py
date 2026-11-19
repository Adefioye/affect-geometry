import hashlib

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from affect_geometry.m03.extraction import Scorer
from affect_geometry.m03.paths import canonical_paths
from affect_geometry.m03.readouts import Readouts, unembedding_hash


def paths(tokenizer, text="A comment. Emotion:"):
    return canonical_paths(tokenizer, text, label_prefix=" ", stop_token_id=1, max_new_tokens=10)


def test_paths_preserve_prompt_ids_and_complete_multitoken_labels(bundle):
    _, tokenizer, *_ = bundle
    req = paths(tokenizer)
    assert req.prompt_ids == tuple(tokenizer.encode("A comment. Emotion:"))
    assert len(req.paths) == len(set(req.paths)) == 28
    for label, path in zip(req.labels, req.paths, strict=True):
        assert tokenizer.decode(path[:-1]).strip() == label
        assert path[-1] == 1
        assert len(path) > 2
    assert len(req.nodes) < sum(map(len, req.paths))
    with pytest.raises(ValueError, match="budget"):
        canonical_paths(tokenizer, "prompt", label_prefix=" ", stop_token_id=1, max_new_tokens=1)
    with pytest.raises(ValueError, match="special"):
        canonical_paths(tokenizer, "prompt", label_prefix=" ", stop_token_id=123, max_new_tokens=10)


def test_actual_gemma_native_logits_candidate_alignment_and_shared_forward_count(
    bundle, tiny_config
):
    model, tokenizer, readouts, *_ = bundle
    req = paths(tokenizer)
    scorer = Scorer(model, tokenizer, readouts, tiny_config)
    values = scorer.score([req], native_check=True)[0]
    assert scorer.calls == int(np.ceil(len(req.nodes) / tiny_config.prefix_batch_size))
    assert scorer.native_max_error < 1e-5
    native = scorer.keys.index(("native", 2))
    final_logit = scorer.keys.index(("logit", 2))
    np.testing.assert_allclose(values["label_scores"][native], values["label_scores"][final_logit])
    for label in (0, 9, 27):
        continuation = req.paths[label]
        seq = torch.tensor([req.prompt_ids + continuation[:-1]])
        with torch.inference_mode():
            logits = model(input_ids=seq, use_cache=False).logits[0].float()
        positions = np.arange(len(continuation)) + len(req.prompt_ids) - 1
        gold_scores = logits.log_softmax(-1)[positions, list(continuation)].double().numpy()
        np.testing.assert_allclose(
            values["label_scores"][native, label], gold_scores[:-1].sum(), atol=1e-5
        )
        np.testing.assert_allclose(values["stop_scores"][native, label], gold_scores[-1], atol=1e-5)
    assert all(not block._forward_hooks for block in readouts.blocks)


def test_batch_single_independent_prefix_equivalence(bundle, tiny_config):
    model, tokenizer, readouts, *_ = bundle
    reqs = [paths(tokenizer), paths(tokenizer, "A substantially longer second comment. Emotion:")]
    result = Scorer(model, tokenizer, readouts, tiny_config).technical_checks(reqs)
    assert result["individual_batch_equivalence"]
    assert result["shared_prefix_equivalence"]


def test_tuned_index_mapping_embedding_exclusion_and_readout_math(bundle):
    model, _, readouts, *_ = bundle
    assert readouts.plan["coverage"] == dict(logit=[0, 1, 2], tuned=[0, 1], jacobian=[0, 1])
    assert readouts.metadata["embedding_translators"] == [0]
    assert readouts.metadata["tuned_mapping"] == {"0": 1, "1": 2}
    h = torch.randn(2, 16)
    w, b, _ = readouts.tuned[0]
    raw_tuned = model.lm_head(model.model.norm(h + F.linear(h, w, b))).float()
    torch.testing.assert_close(readouts.project(h, "tuned", 0), raw_tuned)
    j = readouts.jacobian[0]
    raw = model.lm_head(model.model.norm(h @ j.T))
    torch.testing.assert_close(readouts.project(h, "jacobian", 0), 3 * torch.tanh(raw / 3))
    native = model.lm_head(model.model.norm(h))
    torch.testing.assert_close(readouts.project(h, "native", 2), 3 * torch.tanh(native / 3))


@pytest.mark.parametrize("shape", [(16, 16), (32, 64), (33, 33), (100, 2000)])
def test_tuned_compatible_hash_matches_upstream_numpy_algorithm(shape):
    weight = torch.randn(shape)
    expected = hashlib.sha256(np.array_str(weight.numpy(), precision=1).encode()).hexdigest()
    assert unembedding_hash(weight) == expected


@pytest.mark.parametrize(
    "field,value",
    [("d_model", 99), ("base_model_name_or_path", "wrong"), ("unembed_hash", "wrong")],
)
def test_artifact_incompatibility_fails(bundle, tiny_config, field, value):
    model, _, readouts, *_ = bundle
    tuned_config = dict(readouts.metadata["tuned_config"], **{field: value})
    state = {f"{i}.weight": torch.zeros(16, 16) for i in range(3)}
    state.update({f"{i}.bias": torch.zeros(16) for i in range(3)})
    jstate = dict(J={0: torch.eye(16)}, source_layers=[0], d_model=16)
    with pytest.raises(ValueError, match="mismatch"):
        Readouts(model, tiny_config, tuned_config, state, jstate)


def test_capture_removes_hooks_when_model_raises(bundle, tiny_config, monkeypatch):
    model, tokenizer, readouts, *_ = bundle
    scorer = Scorer(model, tokenizer, readouts, tiny_config)

    def fail(**kwargs):
        raise RuntimeError("injected backbone failure")

    monkeypatch.setattr(model.model, "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        scorer.capture([(0, 4, 5)], native_check=False)
    assert all(not block._forward_hooks for block in readouts.blocks)
