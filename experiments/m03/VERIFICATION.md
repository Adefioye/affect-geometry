# Implementation verification

The implementation was checked locally on 2026-11-18. The Python environment has PyTorch
2.13.0 and Transformers 5.14.1; CUDA and MPS both report unavailable.

## Automated checks

```bash
python -m pytest tests/m03 -q
python -m pytest -q
python -m ruff check src/affect_geometry/m03 tests/m03
git diff --check
```

The M03 suite contains 47 passing tests. The repository suite contains 171 passing tests.
Lint and whitespace checks pass. Tests cover:

- Canonical multi-token paths, immutable prompt IDs, EOS contributions, output budgets,
  prediction-position alignment, full-vocabulary normalization, and deterministic ties.
- Actual tiny `Gemma2ForCausalLM` logits and candidate-score agreement, shared versus independent
  candidate-prefix replay, padded batches versus single examples, and hook cleanup on failure.
- Tuned embedding/post-block mapping, residual translator/bias semantics, Jacobian transport,
  final normalization, softcap conventions, upstream-compatible output-head hashes, and
  incompatible-artifact rejection.
- All-layer coverage and beta grids, the six fixed headline settings, cosine-selected dynamic
  layers/beta, relative probability masks, and within-lens contrast.
- Fixed 28-class macro-F1, invalid greedy denominators, ranks, repairs/damage decomposition,
  paired duplicate-text-group bootstrap intervals, and undefined conditional rates.
- Exact baseline provenance and prompts, duplicate audits, gold-independent reference sampling,
  immutable configs/caches, complete 28-label shards, complete-batch reuse and partial-batch
  resume with the original batch context,
  and refusal to extract test scores before optional reference thresholds are frozen.
- Full stage integration, optional calibration, EOS/length/prompt sensitivity, paired direct
  comparisons, long Parquet exports, plot CSVs, and generated figures. Validation-tuning
  configuration fields and validation extraction are rejected.

## Executed command-line workflow

```bash
python -m affect_geometry.m03 smoke --run-id implementation-validation-03
python -m affect_geometry.m03 prepare --config configs/m03/gemma2-all-label-free.json --run-id implementation-preflight-03
python -m affect_geometry.m03 status --config configs/m03/gemma2-all-label-free.json --run-id implementation-preflight-03
```

The CPU smoke workflow completed every stage, including all optional label-free analyses,
and produced eight headline rows and 19 figures. Its saved manifests identify the random
tiny model and invented data as synthetic. Representative curves and heatmaps were visually
inspected. No smoke accuracy is treated as experimental evidence.

The real EXP2 preflight verified all **45,446** canonical single-label records and original
split counts. It froze **64** training pilot IDs, **2,048** reference IDs, and **4,590** test
IDs, together with exact prompt/baseline provenance and a duplicate audit. The prepared run is:

```text
outputs/m03_lens_comparison/google--gemma-2-9b/implementation-preflight-03/
```

## GPU work still pending

The actual Gemma-2-9B pilot command was attempted and stopped at the explicit device check:

```text
CUDA is unavailable. Use a CUDA host for Gemma-2-9B; the offline smoke test runs on CPU.
```

Consequently, the actual 9B backbone/lens tensors have not been loaded, GPU numerical
compatibility and throughput have not been established, and the 4,590-comment experiment has
not been executed. There are no real M03 accuracy results in this verification. Run the GPU
pilot and extraction commands in [README.md](README.md) on the target CUDA environment.
