# Gemma 2 9B M03 pilot results and canonical-only run plan

Use **prefix batch 1 and projection batch 1** for the full canonical-only run under the existing technical tolerance. All three settings produced complete scores on the same 64 comments, and 1/1 was the only setting that passed the technical checks. At the predeclared layer, the tuned lens led direct top-1 accuracy at **25.00% (16/64)** in all three pilots. Gemma's saved greedy responses on those comments scored **15.63% strict (10/64)** and **21.88% relaxed (14/64)**. The validated 1/1 rate projects to **27.61 hours of scoring across all supported layers** for the full canonical-only run, including the unlabeled calibration pass.

## Evaluation and coverage

The base checkpoint was `google/gemma-2-9b`, revision `33c193028431c2fde6c6e51f29e6f17b60cbfac6`, in bfloat16 on an NVIDIA A40. The matched pilots covered 64 unique single-label GoEmotions training comments selected by text hash with seed 42. All three used the saved EXP2 canonical prompt, the same candidate token paths and label order for scoring, and the same frozen lens artifacts. Each comment had 125 readouts across 28 candidate labels. The logit lens covered zero-based post-block layers 0–41, the tuned and Jacobian lenses covered 0–40, and native scoring used the final block. The standard 8/8 launch and the original 16/16 starting configuration had no complete score cache. The completed 8/8 and 8/1 caches came from diagnostic runs. [Frozen 1/1 protocol](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-full-bs1/protocol.json) · [Frozen layer plan](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-full-bs1/freeze.json)

The saved pilot caches confirm full coverage of the supported source layers. Both fitted lenses use released, frozen parameters for every nonfinal block (0–40). The tuned checkpoint also has an embedding translator, outside the post-block comparison. Final block 41 is the target of the Jacobian fit. The logit lens is a fit-free readout of all 42 blocks. Layer 30 was predeclared for the headline accuracy and fixed contrast comparisons. [Frozen layer plan](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-full-bs1/freeze.json) · [M03 methods](../experiments/m03/METHODS.md)

### Native candidates and greedy accuracy

**Native candidates** is the model's final-block baseline under the same scoring rule as the three lenses. For each comment, M03 scores every one of the 28 allowed emotion labels as a continuation of the prompt. It sums the log probabilities of all the label tokens and the end-of-sequence (EOS) token after the label, then chooses the label with the highest total. Each token probability comes from Gemma's full vocabulary, and the final choice is always one of the 28 labels. [M03 methods](../experiments/m03/METHODS.md)

**Greedy strict** and **greedy relaxed** score the same saved free-form responses. Greedy decoding chooses the most probable next token from the full vocabulary at each step. Strict parsing accepts an allowed label as the entire normalized response. Relaxed parsing also accepts a valid first word, so `joy because…` counts as `joy` under relaxed parsing and as invalid under strict parsing. On the matched pilot, 22 of 64 responses were valid under strict parsing and 36 under relaxed parsing. Invalid responses count as incorrect. All three baselines use all 64 comments as the denominator. The two greedy accuracies are shared across the batch settings because they come from the same saved responses. [Greedy parsers](../src/affect_geometry/classification/protocol.py) · [Prepared pilot](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-full-bs1/prepared/pilot.parquet)

## Matched top-1 accuracy and time

Lens columns use predeclared layer 30. The counts in parentheses are correct predictions out of 64.

| Prefix / projection | Technical gate | Logit lens | Tuned lens | Jacobian lens | Native candidates | Greedy strict | Greedy relaxed | Scorer time, all 64 | Extraction wall time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1 / 1** | **Passed** | 17.19% (11) | **25.00% (16)** | 20.31% (13) | 23.44% (15) | 15.63% (10) | 21.88% (14) | 949.26 s | 594.77 s, resumed 40 |
| 8 / 8 | Failed, diagnostic | 18.75% (12) | **25.00% (16)** | 21.88% (14) | 25.00% (16) | 15.63% (10) | 21.88% (14) | 169.17 s | 170.73 s, all 64 |
| 8 / 1 | Failed, diagnostic | 18.75% (12) | **25.00% (16)** | 21.88% (14) | 25.00% (16) | 15.63% (10) | 21.88% (14) | 900.10 s | 901.50 s, all 64 |

At 1/1, the tuned lens got six more correct than strict greedy, two more than relaxed greedy, and one more than native candidates. Against relaxed greedy, it repaired five misses and lost three correct predictions. The direct logit and Jacobian lenses both exceeded strict greedy and trailed relaxed greedy at 1/1. At prefix batch 8, the logit, Jacobian and native candidates each gained one net correct answer relative to 1/1. Their predictions at the reported readouts changed on two, five and six comments, respectively. The tuned predictions at layer 30 matched across all settings. [Plot-ready accuracy CSV](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-pilot-accuracy/analysis-report/plot_data/direct-top1-by-layer.csv)

### Why logit-lens accuracy differs from greedy decoding

At layer 30, the logit lens applies Gemma's final normalization, language-model head and logit softcap to an intermediate state. Blocks 31–41 can still change that state and the resulting token probabilities. At block 41, the logit lens recovers Gemma's native logits for each prefix, so its candidate predictions match the native candidate baseline. The validated 1/1 run scored **17.19% (11/64)** at layer 30 and **23.44% (15/64)** at layer 41. The final-block score was one correct response above relaxed greedy and five above strict greedy. The final-block gap reflects the change from ranking complete label paths to generating and parsing free-form text. [M03 methods](../experiments/m03/METHODS.md) · [1/1 equivalence checks](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-full-bs1/pilot_checks.json) · [Plot-ready accuracy CSV](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-pilot-accuracy/analysis-report/plot_data/direct-top1-by-layer.csv)

The matched scorer times sum timings from all retained shards and exclude model loading, technical checks and checkpoint I/O. The 1/1 sum includes 24 scores from an earlier attempt and 40 from the resumed run. At 8/8, scoring was **5.61× as fast** as at 1/1. At 8/1, scoring took **5.18% less time** than at 1/1 and **5.32× as long** as at 8/8. Both prefix-8 configurations ran 232 backbone forwards, so most of their observed timing difference came from changing projection batch size. The recorded 8/1 command took **18 min 31.33 s** end to end, including preparation, model loading, technical checks, scoring and comparison. [8/8 runtime](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-paired/runtime.csv) · [8/1 runtime](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-mixed/runtime.csv) · [8/1 command timing](../logs/google--gemma-2-9b__m03__m03-gemma2-mixed/console.log)

### Best direct top-1 layers

![Gemma 2 9B direct top-1 accuracy by layer, lens and prefix/projection setting](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-pilot-accuracy/analysis-report/plots/direct-top1-by-layer.png)

Each panel plots direct top-1 accuracy against zero-based post-block layer for one batch setting. The vertical line marks block 30, the horizontal lines show strict and relaxed greedy accuracy, and the gray band spans native-candidate accuracy across batch settings. The table gives every tied maximum. [Plot-ready accuracy CSV](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-pilot-accuracy/analysis-report/plot_data/direct-top1-by-layer.csv)

| Lens | Validated 1/1 peak | Best layers, 1/1 | Diagnostic peak, both 8/8 and 8/1 | Best layers, both diagnostics |
| --- | ---: | --- | ---: | --- |
| Logit | 25.00% (16/64) | 39 | 25.00% (16/64) | 39, 41 |
| Tuned | 32.81% (21/64) | 9, 29 | 34.38% (22/64) | 9 |
| Jacobian | 35.94% (23/64) | 2 | 34.38% (22/64) | 4, 5, 6, 8, 10 |

The logit lens peaked late, the tuned lens tied at an early and a late layer in the validated run, and the Jacobian lens peaked early under all three batch settings. The early peaks were sensitive to batch setting. Always predicting neutral would have scored **34.38% (22/64)**, one correct comment below the Jacobian peak at 1/1. One comment changes accuracy by 1.5625 percentage points. These maxima describe the 64 training comments used to select them. Layer 30 remains the predeclared comparison.

### Direct readout metrics for the three lenses

These metrics use the validated 1/1 cache and the label-plus-EOS score. Macro-F1 averages all 28 emotion labels, with zero for an undefined class F1. The pilot contains 18 gold labels, several with only one example. Mean reciprocal rank (MRR) averages the reciprocal of the gold label's rank. Recall@3 and recall@5 count how often the gold label appears among the first three or five candidates. The median gold margin is the gold-label score minus the highest-scoring rival, in summed log-probability units. [Metric definitions](../m-03-docs.md#layerwise-metrics-and-what-we-do-with-them) · [1/1 pilot cache](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-full-bs1/scores/pilot/canonical/index.json)

| Lens | Layer | Macro-F1 | MRR | Recall@3 | Recall@5 | Median gold margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logit | 30 | 13.90% | 0.314 | 19/64 | 26/64 | −4.71 |
| Tuned | 30 | 21.53% | 0.398 | 30/64 | 31/64 | −2.42 |
| Jacobian | 30 | 19.04% | 0.297 | 18/64 | 25/64 | −1.65 |
| Logit | 39 (peak) | 17.14% | 0.406 | 28/64 | 34/64 | −1.63 |
| Tuned | 9 (tied peak) | 1.88% | 0.427 | 28/64 | 32/64 | −1.86 |
| Tuned | 29 (tied peak) | 28.06% | 0.449 | 29/64 | 37/64 | −1.85 |
| Jacobian | 2 (peak) | 2.68% | 0.442 | 28/64 | 34/64 | −2.80 |

At layer 30, the tuned lens led on macro-F1 and placed the gold label in the top three on 30 comments, versus 19 for logit and 18 for Jacobian. Each lens corrected five native candidate errors. Logit lost nine native correct predictions, tuned lost four, and Jacobian lost seven. Paired 95% text-group bootstrap intervals for the accuracy differences versus native covered zero for all three lenses (logit −17.19 to +4.69 pp, tuned −7.81 to +10.94 pp, Jacobian −14.06 to +7.81 pp, 2,000 resamples, seed 42). The Jacobian lens had the least negative median gold margin but also the tightest median gap between first and second place (0.21, versus 1.02 for logit and 1.14 for tuned). Its smaller score gaps accompanied lower rank-based performance than the tuned lens.

The tuned lens's top-1 tie hides a large difference in class coverage. At layer 9 it predicted neutral on 58 comments and all 21 correct predictions were neutral. At layer 29 it predicted neutral eight times, got four neutral comments right, and classified another 17 comments correctly across 11 emotions. The Jacobian lens also predicted neutral 58 times at its layer-2 peak. Its 23 correct predictions came from all 22 neutral comments and one admiration comment. The logit lens improved in macro-F1 and gold-label ranking from layer 30 to its layer-39 peak, where its 16 correct predictions included six neutral comments and ten from other emotions.

At layer 30, logit and tuned agreed on 34/64 predictions and shared four of their five native-error repairs. Tuned and Jacobian agreed on 31/64 and shared three repairs. Logit and Jacobian agreed on 20/64 and shared two repairs. Across adjacent supported layers, the mean fraction of comments that changed predicted label was **28.62%** for logit, **34.65%** for tuned and **34.57%** for Jacobian. At least one layer got the answer right on **29/64**, **44/64** and **45/64** comments, respectively. These counts use the gold labels to look across layers after scoring and measure how often the answer is available somewhere in depth.

Removing the EOS term at layer 30 changed **3**, **2** and **25** predictions for logit, tuned and Jacobian, respectively. Their correct counts moved from 11 to 12, 16 to 16, and 13 to 11. The early peaks were especially dependent on EOS scores. The tuned lens at layer 9 went from 21 to 3 correct, and the Jacobian lens at layer 2 went from 23 to 2. Dividing the complete score by the label-plus-EOS token count left predictions unchanged at layer 30 and at all the listed 1/1 peaks. Full-vocabulary KL and calibration metrics are unavailable in the saved pilot outputs. [1/1 pilot cache](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-full-bs1/scores/pilot/canonical/index.json) · [Score modes and vocabulary diagnostics](../experiments/m03/METHODS.md)

### Fixed contrast at layer 30

The predeclared native-versus-lens contrast uses layer 30 and beta 0.5. It reranks the same candidates with `native + 0.5 × (native − lens)` using cached scores. Contrast accuracies were identical across the three settings.

The additional contrast metrics below use the validated 1/1 cache.

| Contrast using | Correct / 64 | Accuracy | Macro-F1 | MRR | Recall@3 | Recall@5 | Native errors corrected / native correct lost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logit lens | 0 | 0.00% | 0.00% | 0.146 | 7/64 | 17/64 | 0 / 15 |
| Tuned lens | 8 | 12.50% | 7.72% | 0.307 | 22/64 | 33/64 | 3 / 10 |
| Jacobian lens | 17 | 26.56% | 15.63% | 0.446 | 30/64 | 41/64 | 5 / 3 |

The Jacobian contrast led on every metric in the contrast table. Compared with direct Jacobian at layer 30, contrast raised recall@5 from 25/64 to 41/64 and reduced macro-F1 from 19.04% to 15.63%. It gained three net correct comments over relaxed greedy, repairing nine and losing six, and gained seven over strict greedy, repairing ten and losing three. Its margin over matched native candidates was two comments at 1/1 (paired 95% bootstrap interval −4.69 to +12.50 pp) and one in each diagnostic. [Predeclared contrast methods](../experiments/m03/METHODS.md)

### Batch stability

| Readout | 8/8 flips vs 1/1 | 8/1 flips vs 1/1 | Matched predictions |
| --- | ---: | ---: | ---: |
| Logit lens, all layers | 52 | 52 | 2,688 |
| Tuned lens, all layers | 91 | 93 | 2,624 |
| Jacobian lens, all layers | 145 | 145 | 2,624 |
| Native candidates | 6 | 6 | 64 |
| **All readouts** | **294 (3.68%)** | **296 (3.70%)** | **8,000** |

The all-readout denominator counts comment–readout pairs, including the final logit-lens readout and its matching native readout. Switching from 8/8 to 8/1 changed eight tuned-lens predictions across all layers. Only layer 12 changed in accuracy, from 12 to 13 correct with 8/1. The first failing candidate-score comparison exceeded the **0.25** absolute tolerance in six of 3,500 entries in each diagnostic, with maxima **0.29523** for 8/8 and **0.29518** for 8/1. Either prefix-8 configuration changed six of 64 native predictions. The numerical and prediction checks support 1/1 even though the diagnostics were faster and had similar headline accuracies. These timings and checks cover one matched cohort on one GPU. [1/1 checks](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-full-bs1/pilot_checks.json) · [8/8 checks](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-paired-bs8-diagnostic/diagnostic_checks.json) · [8/1 checks](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-mixed-prefix8-proj1-diagnostic/diagnostic_checks.json)

## Gemma greedy accuracy on the full dataset

| EXP2 evaluation population | Comments | Strict correct | Relaxed correct | Relaxed gain |
| --- | ---: | ---: | ---: | ---: |
| Matched M03 training pilot | 64 | 10 (15.625%) | 14 (21.875%) | +4 (+6.250 pp) |
| Full single-label test split | 4,590 | 384 (8.366%) | 787 (17.146%) | +403 (+8.780 pp) |
| All single-label splits pooled | 45,446 | 3,994 (8.788%) | 7,945 (17.482%) | +3,951 (+8.694 pp) |

Relaxed parsing raised the valid-response count from 22 to 36 on the pilot and from 1,714 to 3,077 on test. Use the test-only row as the full-run greedy baseline. The pilot row gives the sample-matched comparison with the lenses. The full reference and test lens caches are still pending, so the pilot establishes no full-test lens advantage. [EXP2 classification report](emotion-classification.md)

## Full label-free setup with one prompt

Use the original EXP2 canonical **template and exact rendered prompt for each comment**, with its original per-comment candidate menu order. The canonical prompt uses the fixed EXP2 instructions and a deterministic label shuffle per sample. All three lenses and the native output share the same backbone passes. For a single prompt per comment, disable both optional test arms. `alternate_prompt` changes the wording on 64 test comments, and `alternate_menu` reverses the labels in the original template on all 4,590 test comments.

Start from a copy of the all-label-free configuration with these settings and a fresh run ID so the configuration is frozen at preparation.

```json
{
  "prefix_batch_size": 1,
  "projection_batch_size": 1,
  "calibrated": true,
  "length_normalized": true,
  "alternate_menu": false,
  "alternate_prompt": false
}
```

Validate the technical pilot, extract the 2,048 canonical training reference comments, freeze the unlabeled calibration thresholds, then extract and evaluate the 4,590 canonical test comments with all three lenses. The full layer and beta scans, fixed contrast rules, label-token-only scoring, length normalization and canonical diagnostics all use those scores. Calibration uses unlabeled training scores and provides a threshold diagnostic. Gold labels enter the retrospective accuracy evaluation. The existing M03 analysis uses the strict greedy parser by default. To report relaxed accuracy and paired gains in the final analysis, rescore the same saved canonical responses with `parse_compat` and join by `sample_id`. This requires no additional generation. The canonical-only design gives up direct measurement of sensitivity to wording and label order. [Current all-label-free config](../configs/m03/gemma2-all-label-free.json) · [Execution and freeze workflow](../experiments/m03/README.md)

### Scoring budget on one A40

The measured pilot rate already includes all 42 Gemma blocks and all 125 readouts per comment (42 logit, 41 tuned, 41 Jacobian and one native). Each candidate-prefix backbone pass supplies the states for all supported readouts. **Use 27.61 hours directly for the canonical-only 1/1 scoring budget.** Multiplying by 42 would count work already included in the measured rate. [M03 methods](../experiments/m03/METHODS.md) · [1/1 pilot timing and readout count](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-full-bs1/pilot_checks.json)

These are linear projections from the measured pilot scorer rates, in hours. The canonical-only total counts 64 pilot scores, 2,048 reference scores and 4,590 test scores as separate stages (6,702 scoring jobs). The pilot IDs also lie within the training reference set, but the current scorer stores these phases separately. The all-variants total adds 4,590 reversed-menu and 64 reworded test jobs. Model loading, technical validation, checkpoint I/O, CPU analysis, exports, and any changes in runtime with prompt length or device load are outside these estimates.

| Prefix / projection | Canonical test only | Canonical reference | Canonical-only total | All-variants total | Estimated hours saved |
| --- | ---: | ---: | ---: | ---: | ---: |
| **1 / 1, validated** | 18.91 | 8.44 | **27.61** | 46.79 | **19.18** |
| 8 / 8, diagnostic | 3.37 | 1.50 | 4.92 | 8.34 | 3.42 |
| 8 / 1, diagnostic | 17.93 | 8.00 | 26.18 | 44.36 | 18.18 |

Canonical-only removes **41.0%** of scoring jobs relative to all variants. Disabling only the reworded alternate prompt would save 64 jobs, about **16 minutes** at 1/1, because the reversed-menu pass accounts for most of the added work. The 1/1 estimate is the budget supported by the technical gate. The faster rows show the timing tradeoff observed in the diagnostics.

## Sources and verification

- Recomputed direct accuracies and flips from the 192 canonical score shards. Verified completed indexes, shard metadata, ordered sample IDs, prompt hashes, token-path identities, finite score arrays, and the plotted layer counts. [1/1 cache index](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-full-bs1/scores/pilot/canonical/index.json) · [8/8 cache index](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-paired-bs8-diagnostic/scores/pilot/canonical/index.json) · [8/1 cache index](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-mixed-prefix8-proj1-diagnostic/scores/pilot/canonical/index.json)
- Verified the accuracy denominators, matched rendered prompts and greedy responses, and rescored the pinned local EXP2 snapshot. All three split Parquet hashes matched the protocol. [Prepared pilot](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-full-bs1/prepared/pilot.parquet) · [Pinned EXP2 provenance](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-full-bs1/protocol.json)
- Cross-checked retained scorer times from the same shards and the layer-wise batch agreement against the saved diagnostic tables. [8/8 layer agreement](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-paired/plot_data/layer_agreement.csv) · [8/1 layer agreement](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-mixed/plot_data/layer_agreement.csv)
- Computed the additional lens metrics from the saved 1/1 pilot shards and joined them to the prepared pilot by sample ID. Used the M03 fixed-taxonomy metric, stable rank, and paired text-group bootstrap implementations. [M03 metric implementation](../src/affect_geometry/m03/metrics.py) · [1/1 pilot cache index](../outputs/m03_lens_comparison/google--gemma-2-9b/m03-gemma2-full-bs1/scores/pilot/canonical/index.json)
