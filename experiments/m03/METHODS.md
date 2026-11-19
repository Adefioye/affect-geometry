# M03 implementation conventions

The authoritative experiment specification is [m-03-docs.md](../../m-03-docs.md).
The reusable implementation is in `src/affect_geometry/m03/`; the CLI only coordinates stages.

## Frozen Gemma artifacts

| Component | Frozen choice |
| --- | --- |
| Backbone/tokenizer | `google/gemma-2-9b`, `33c193028431c2fde6c6e51f29e6f17b60cbfac6` |
| Tuned lens | `chutommy/tuned-lenses`, `5667bafe74f3e55e9f1be1f9b1a2f699a8383fdd`, `tuned-lens--gemma-2-9b--fineweb-edu` |
| Jacobian lens | `neuronpedia/jacobian-lens`, `d2527c3cdbaa60dd4b65cbca1604921f044e989a`, `gemma-2-9b/jlens/Salesforce-wikitext/gemma-2-9b_jacobian_lens.pt` |
| Dataset | `google-research-datasets/go_emotions`, `add492243ff905527e67aeb8b80c082af02207c3` |
| Greedy baseline | `kokolamba/rq2-goemotions-single-label-gemma2`, `cbe0b8e3dd8df548fac272246f70a291a2c74e1a` |

FineWeb-Edu is the predeclared tuned-lens corpus, selected without GoEmotions accuracy.
The other released tuned-lens corpora are not searched for a better test result.
The tuned config does not pin its base-model revision: the loader verifies its model ID,
width, block count, declared base revision when present, and output-head hash.
`freeze.json` preserves artifact configuration, checksums, and the resulting mapping.

The [tuned-lens training loop](https://github.com/AlignmentResearch/tuned-lens/blob/main/tuned_lens/scripts/train_loop.py)
uses `output.hidden_states[:-1]`, which includes the embedding and excludes the final normalized
state. With standard HF indexing, translator `i=0` reads embeddings; translator `i=l+1`
reads raw post-block `l`. The default explicitly declares this indexing, excludes the embedding
translator from block curves, and records its presence. It does not infer 42 fitted post-block
sources from 42 matrices. Unexpected checkpoint layouts fail rather than receiving guessed maps.

The [tuned-lens transform](https://github.com/AlignmentResearch/tuned-lens/blob/main/tuned_lens/nn/lenses.py)
is `h + linear(h)`. Its [unembedding wrapper](https://github.com/AlignmentResearch/tuned-lens/blob/main/tuned_lens/nn/unembed.py)
applies the final norm and LM head without Gemma's final softcap; that convention is preserved
by default (`tuned_softcap: false`). The native and raw logit arms use norm, head, and native
softcap exactly once. The [Jacobian readout](https://github.com/anthropics/jacobian-lens/blob/main/jlens/lens.py)
transports float32 states with `h @ J.T` and uses the
[native HF wrapper](https://github.com/anthropics/jacobian-lens/blob/main/jlens/hf.py), including softcap.

The released Jacobian [fit configuration](https://huggingface.co/neuronpedia/jacobian-lens/blob/d2527c3cdbaa60dd4b65cbca1604921f044e989a/gemma-2-9b/jlens/Salesforce-wikitext/config.yaml)
declares `target_layer: null`; the upstream fitter resolves that to the final block, index 41.
This is the fit target, distinct from M03's intermediate contrast comparison layer. Actual
matrix keys determine supported Jacobian sources. The config reports 462 prompts fitted;
its environment metadata was backfilled and does not pin the original jlens commit. These
provenance limitations remain visible in the saved fit config.

The runtime reads frozen PyTorch tensors directly with `weights_only=True`; it does not need
the upstream lens packages, refit weights, duplicate the vocabulary head, or change tokenizer
BOS behavior through a third-party wrapper. Independent tests check the readout formulas.

## Candidate scores and sharing

Canonical label paths append token IDs for a leading space and each complete label, then
Gemma EOS token `1`, to the unchanged EXP2 prompt token IDs. The exact paths and decoded
boundaries are verified. Every probability uses a stable full-vocabulary log-softmax.
The state predicting continuation token `t` is the last state of `prompt + continuation[:t]`.
Stop and label contributions are saved separately.

The implementation deduplicates candidate prefixes within each comment and batches them
across comments. Each prefix backbone pass supplies all layers and all lenses. Small
prediction-position residuals are captured using temporary block hooks; vocabulary projections
are chunked. Full prefixes are replayed; this version does not implement KV caching or early
exit, and makes no compute-saving claim for either. Cached contrast requires no backbone pass.

Pilot tests compare native CausalLM logits with final raw-state readout, separate candidate
paths with shared prefixes, and individual examples with padded batches. Full-vocabulary
diagnostics use the empty candidate prefix (the final prompt token) for a deterministic
sample and record `KL(native || lens)`. Optional float16 log-probability arrays retain the
sampled distributions; the KL itself is computed before that storage conversion.

## Predictions and evaluation

- Direct: rank complete-label scores at every supported layer, breaking ties in canonical order.
- Native contrast: `F + beta*(F-E)`, at every supported nonfinal layer and beta `0, .25, .5, 1`.
- Headlines: common nonfinal layer nearest 75% depth; contrast beta `.5`. Exact depth ties
  choose the lower index. With the standard full Gemma common panel, the headline is block 30.
- Dynamic: greatest cosine distance over common layers nearest `.65, .75, .85, .95` depth;
  ties choose the lower index; beta is `2/(41-selected_layer)`.
- Mask: the fixed headline contrast with native label probability at least `.1` times the maximum.
- Within-lens: use the highest common fitted nonfinal source as the upper reference and beta `.5`
  for each lower supported source. Compare against its direct upper-layer readout.

Configuration and runtime scoring rules do not consume gold annotations. Labels enter
evaluation, retrospective examples, bootstrap intervals, exclusion diagnostics, and offline
opportunity analysis after prediction settings are frozen. Macro-F1 always averages the fixed
28 classes, zero-dividing to zero; an invalid greedy output is a false negative for its gold
class and remains in the denominator. Duplicate text hashes define paired bootstrap groups.

Contrast tables include paired comparisons with the same-layer and predeclared direct rules;
dynamic same-layer direct uses the per-comment selected layer. Within-lens contrast also has
an upper-direct comparison. All have the same resampled groups as the two main baselines.
Sensitivity tables count prediction changes from removing EOS, optional length normalization,
and alternate prompt/menu scoring. Alternate-prompt rows identify the canonical greedy
reference as a different prompt (`greedy_prompt_match: false`); native candidate comparisons
remain matched to that arm's prompt.

The six primary configurations are never selected by test peaks. The implementation reports
pointwise confidence intervals and effect sizes; it does not report significance tests or
familywise significance for full depth scans. No validation-tuned selector is implemented.
