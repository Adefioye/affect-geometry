# M03 files

Actual model outputs live under:

```text
outputs/m03_lens_comparison/google--gemma-2-9b/<run-id>/
```

Operational logs use `logs/google--gemma-2-9b__m03__<run-id>/run.log` and the shared structured
logger/progress utilities. Model/lens/baseline downloads use `assets/hub/` inside the run.
Do not commit or publish the private EXP2 source snapshot through this workflow.

| File or directory | Contents |
| --- | --- |
| `protocol.json` | Frozen config, source provenance, source/split checksums, selected IDs, implementation hash, and test history |
| `prepared/*.parquet` | Canonical pilot, reference, and test source/baseline records |
| `prepared/duplicate_audit.csv` | Source IDs/text hashes, original splits, duplicate counts, cross-split overlap |
| `freeze.json` | Actual layer coverage/intersection, translator mappings, artifact hashes, fit config, conventions, versions, headline/dynamic layers |
| `pilot_checks.json` | Native agreement, shared-prefix/batch equivalence, tolerances, 64-comment extraction completion and cost; no accuracy selection |
| `scores/<phase>/<variant>/` | Atomic per-comment `.npz` score shards, frozen `context.json` and `index.json`, completion marker |
| `thresholds.json` | Optional training-reference quantiles, exact reference IDs and frozen context |
| `runtime/*.json` | Extraction invocation timings, reused/new examples, prefix forwards, batches, GPU and peak memory |
| `tables/complete_label_scores_<variant>.parquet` | Long score table, exported by `run`, `analyze --export-scores`, or `export-scores` |
| `tables/predictions.parquet` | Per-comment outcomes for every evaluated arm, settings, gold ranks/margins, both baselines, repairs/damage |
| `tables/metrics.csv` | All-layer/grid metrics, paired intervals/effects, primary flags, deduplicated sensitivity |
| `tables/main_results.csv` | Two baselines and the six predeclared primary configurations, canonical total scores |
| `tables/common_layer_panel.csv` | Metrics restricted to the shared source-layer intersection |
| `tables/class_metrics.csv` | Fixed-taxonomy precision, recall, F1, support, predicted counts, true positives |
| `tables/availability.csv` | Offline correct-label availability across depth and gaps to predeclared direct rules |
| `tables/trajectories.csv` | First correct source, correctness transitions, prediction changes per comment/lens |
| `tables/stability.csv` | Adjacent available-layer prediction-change rates and actual layer-index gaps |
| `tables/agreement.csv` | Cross-lens agreement and repair intersections/unions at common layers |
| `tables/confusion.csv` | 28 gold classes by 29 prediction outcomes, including invalid greedy responses |
| `tables/mask_exclusion.csv` | Per-class gold exclusion under the fixed native plausibility mask |
| `tables/dynamic_selection.csv` | Selected layer/beta frequencies for each dynamic lens arm |
| `tables/token_length.csv` | Performance stratified by gold canonical-label token count |
| `tables/scoring_and_prompt_sensitivity.csv` | Prediction changes, repairs/damage relative to canonical total scoring |
| `tables/neutral_transitions.csv` | Neutral-to-emotion/emotion-to-neutral changes and repair/damage counts |
| `tables/examples.csv` | Deterministic retrospective repairs, damages, and unchanged errors |
| `tables/calibration.csv` | Optional training-threshold statistic, per-class exceedance/support and multi-label exceedance rates |
| `tables/vocabulary_kl.csv` | Full-vocabulary KL at declared diagnostic prompt positions |
| `tables/runtime.csv` | Collected extraction invocation records |
| `plot_data/*.csv` | Self-contained values and experimental context used to recreate every essential figure |
| `plots/*.png` | Direct/contrast curves, beta heatmaps, class recall, confusion, repairs/damage, ranks, and vocabulary KL |
| `analysis_manifest.json` | Analysis completion, intervals/selection conventions, context hash, cached analysis time |

Each score shard contains arrays `[readout, 28]` named `label_scores`, `stop_scores`, and
`token_counts`. The first readout is native final scoring; `context.json` maps every other
row to its lens and post-block index. JSON metadata binds sample, prompt, token paths,
artifact/layer/config/software freeze, phase and prompt variant. Diagnostic shards may also
contain `[readout, vocabulary]` float16 log probabilities and a float64 KL vector.

The long score export names every sample, split, lens, layer, label, token count, label-token
score, EOS score, total score, and matched native reference score. The compact shards can
reconstruct all arm scores without inference; they remain available when long export is skipped.

Runtime rows describe separate invocations, including resumed work. `load_seconds` describes
the bundle's initial loading measurement; when `run` shares one model bundle across phases,
that same measurement appears on the phase rows and must only be counted once. Extraction
timing includes candidate replay, readout projection, and any `replayed_for_batch_context`
examples needed to finish interrupted batches. Cached analysis cost is separately
recorded in `analysis_manifest.json`. No early-exit or KV-cache savings are claimed.

Deduplicated sensitivity keeps one row per test text hash and excludes text also appearing
in train or validation. Canonical metrics preserve all 4,590 test rows. Layerwise intervals
are pointwise, and optional calibration is a diagnostic statistic alongside ordinary top-1
accuracy. Missing reference/pilot/test shards stop analysis instead of reducing its denominator.
