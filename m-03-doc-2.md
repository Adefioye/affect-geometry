# M03 experiments with direct readout and layer contrast

This document specifies two ways to classify a GoEmotions comment from frozen vocabulary
readouts. Direct readout selects the highest-scoring complete emotion label at a particular
layer. Layer contrast adjusts native final-layer scores using an intermediate readout, then
selects the highest-scoring complete label. Both methods are evaluated independently with
the logit lens, tuned lens, and Jacobian lens.

The current experiments are label-free with respect to GoEmotions annotations: prediction
rules and settings are fixed without using gold labels to select configurations. Variants
that select settings using validation labels are reserved for future work.

Both approaches explore all available layers across early, middle, and late depths for
each of the three lenses. Representative depth samples do not replace the full layerwise
evaluation. A predeclared single-layer result and the smaller dynamic-selection panel are
specific prediction rules within this broader all-layer study.

The main outcome is top-1 label accuracy at each layer, measured on the same comments as
the existing zero-shot greedy baseline. Native final-layer candidate scoring supplies a
second baseline that shares the candidate paths and selection procedure with the lenses.
The experiments also measure macro-F1, repairs, damage, label ranks, prediction changes
across depth, and inference cost.

## Scientific questions and method provenance

| Question | Experiment and evidence |
| --- | --- |
| Can an intermediate readout select the annotated emotion more accurately? | Compare direct-readout top-1 accuracy with both baselines at every available layer. |
| Does the change between intermediate and final scores improve the decision? | Compare layer contrast with native final scoring and direct readout at the same layer. |
| Which readout makes the useful information most accessible? | Compare the three lenses at common layers using identical prompts and candidate paths. |
| Are improvements broad or concentrated on particular emotions? | Inspect macro-F1, per-class recall, confusion matrices, and neutral-related transitions. |
| Could choosing a different depth help particular comments? | Measure correct-label availability across layers and the overlap of repairs. |

Belrose et al. motivate evaluating intermediate predictions across depth. Their classification
application reproduces Halawi et al.'s few-shot experiments, including misleading
demonstrations. Applying the evaluation to our zero-shot GoEmotions prompt is a proposed
task transfer. [Belrose et al., Sections 3 and 5.2](https://arxiv.org/html/2303.08112v6)

Sharma et al. motivate the contrast score, selection of an intermediate comparison layer,
and the layer-distance rule for contrast strength. Their experiments use fine-tuned
classifiers and validation-selected hyperparameters. The label-free configuration rules,
complete-label vocabulary scoring, and three-lens comparison below are proposed
adaptations. [Sharma et al., Sections 3–4](https://aclanthology.org/2024.emnlp-main.1161.pdf)

The Jacobian lens provides a fitted map from intermediate residual states to vocabulary
readouts. The notebook in this workspace returns logits over the entire vocabulary.
Applying those logits to complete-label classification and layer contrast is a proposed
evaluation of that readout. [Jacobian Lens method](https://transformer-circuits.pub/2026/workspace/)
and [reference readout implementation](https://github.com/anthropics/jacobian-lens/blob/main/jlens/lens.py)

## Models, artifacts, and population

Begin with Gemma, for which public tuned and Jacobian lens weights are available. Replicate
the selected comparisons on OLMo 2 or Apertus once checkpoint-specific lenses are available.

| Model identifier | EXP2 model revision | Blocks |
| --- | --- | ---: |
| `google/gemma-2-9b` | `33c193028431c2fde6c6e51f29e6f17b60cbfac6` | 42 |
| `allenai/OLMo-2-1124-7B-Instruct` | `470b1fba1ae01581f270116362ee4aa1b97f4c84` | 32 |
| `swiss-ai/Apertus-8B-Instruct-2509` | `b946d40447b2b597999b9c86d44bee0b452c919f` | 32 |

Use the EXP2 tokenizer, rendered prompt, label-menu order, precision, and model interface.
Gemma uses the base completion interface. OLMo 2 and Apertus use their recorded native chat
templates. Pin each lens artifact revision and record its fit corpus, source layers, target
layer, normalization convention, and compatibility checks. A lens artifact's target layer
and the comparison layer used in our contrast experiment are separate configuration fields.
[Public Gemma tuned lenses](https://huggingface.co/chutommy/tuned-lenses) and
[public Gemma Jacobian lens](https://huggingface.co/neuronpedia/jacobian-lens/tree/main/gemma-2-9b/jlens/Salesforce-wikitext)

Use the EXP2 single-label population at dataset revision
`add492243ff905527e67aeb8b80c082af02207c3`, retaining all 28 classes, including neutral.

| Original split | Eligible comments | Use in this experiment |
| --- | ---: | --- |
| Train | 36,308 | Technical pilot and any optional unlabeled reference statistics. |
| Validation | 4,548 | Reserved for future validation-tuned variants; labels are not used to choose current settings. |
| Test | 4,590 | Evaluation after the scoring procedure and comparison families are frozen. |

Verify these counts against the pinned data. Preserve source IDs and split membership.
Audit duplicate text across splits and report a deduplicated sensitivity analysis alongside
the canonical population. Use duplicate-text groups for uncertainty estimates.
[EXP2 population and scoring protocol](docs/exp2.md)

The current comparison uses frozen models and fitted lenses, with configuration rules
fixed before viewing GoEmotions accuracy. Its initial defaults appear below. Gold labels
are used to calculate evaluation metrics and retrospective diagnostics, not to select
layers, beta, masks, prompts, or other settings. Any reference statistics use unlabeled
training text. Validation-based selection is outside the current run and reporting scope;
its protocol is preserved under [Future work](#future-work--validation-tuned-variants).

If earlier EXP2 test outcomes influenced M03 method choices, describe the official test
comparison as exploratory. An untouched reserve or independently annotated confirmation
set can provide the confirmatory evaluation. Simply having run a fixed baseline on the test
set does not by itself establish that later method selection used its outcomes. Record the
actual history of inspection and selection.

## All-layer coverage and shared computation

The all-layer scan is part of the current experiment scope for both direct readout and
layer contrast. It is not deferred to future work or restricted to a few representative
early, middle, and late layers.

| Component | Required layer coverage |
| --- | --- |
| Logit-lens direct readout | Every post-block state, including the final-block readout used to check agreement with native scoring. |
| Tuned- and Jacobian-lens direct readout | Every fitted post-block source layer supplied by the compatible artifact. Report each lens's full coverage and the intersection shared by all three lenses. |
| Contrast with the native final layer | Every available nonfinal post-block source layer for each lens, crossed with the predefined beta grid `0`, `0.25`, `0.5`, and `1`. The native final layer remains the expert reference. |
| Main fixed-setting comparison | The common nonfinal layer nearest 75% depth, with beta `0.5` for contrast. This supplies a predeclared headline configuration without reducing the all-layer scan. |
| Secondary dynamic contrast | The four predefined depths `0.65`, `0.75`, `0.85`, and `0.95` restrict this rule's per-comment selection only. They do not restrict score extraction, direct-readout curves, or the full contrast scan. |

For Gemma-2-9B, the backbone has 42 blocks. The logit-lens scan therefore has 42 post-block
positions, and there are up to 41 nonfinal positions for contrast with the native final
layer. Confirm the tuned and Jacobian artifacts' actual source coverage when loading them;
do not infer post-block coverage from the number of saved matrices alone. Map artifact
indices to backbone block indices explicitly. If an artifact also includes an embedding
state, record it separately rather than counting it as a transformer block. Record missing
or unsupported source layers and their reasons instead of silently dropping depths or
fabricating fitted maps. The final native baseline does not require a fitted translator.

All-layer contrast means comparing each intermediate layer with the native final layer;
it does not require enumerating every pair of intermediate layers. The secondary
within-lens contrast similarly uses its one declared upper reference layer.

Reuse each prompt or candidate-prefix backbone computation across all requested layers
and lenses. Capture the required prediction-position states and apply the frozen readouts
in batches or layer chunks; do not rerun the full backbone separately for each layer or
lens. Shared-prefix caching can reuse the prompt and common candidate prefixes. Extract
complete-label scores once and reuse them for direct readout, contrast, every predefined
beta, and the dynamic rule. These contrast calculations do not require further backbone
inference.

Pre-fitted artifacts remove lens-fitting cost, but all-layer vocabulary projection and
complete-label candidate scoring still consume inference time and memory. Save the compact
candidate scores needed for analysis; full-vocabulary distributions need not be retained
except for the declared diagnostic sample. Use the technical pilot to measure throughput
and peak memory, choose batching and chunk sizes without gold-label feedback, and estimate
full-test runtime on the actual GPU. Report extraction cost separately from cached score
analysis. Runtime estimates do not justify silently reducing the declared layer coverage.

## Shared complete-label scores

For comment `x`, construct a canonical token continuation `y_c` for each candidate emotion
`c`. Append the model-specific declared stop event. Preserve the exact prompt token IDs
when adding a continuation, and verify the decoded label and boundary spaces. Use the same
paths and stop convention for every readout within a model.

At each candidate prefix, obtain full-vocabulary logits from the native model or lens and
apply numerically stable log-softmax over the full vocabulary. The prompt plus the candidate
tokens preceding the prediction position is the input for that step. Score every candidate
independently of the annotation. Shared-prefix caching can reuse the corresponding states.

Let `r` identify a lens, `l` a zero-based post-block index, and `v` a vocabulary token.

```text
q[r,l](v | prefix) = softmax_vocabulary(z[r,l](prefix))[v]
S[r,l](c;x)        = sum_t log q[r,l](y_c[t] | x, y_c[<t])
P[r,l](c;x)        = softmax_over_28_labels(S[r,l](c;x))
E[r,l](c;x)        = log P[r,l](c;x)
F(c;x)            = log P[native,final](c;x)
```

`S` and `E` give identical label rankings. `P` is the normalized distribution over the
declared canonical paths. The Jacobian version is a distribution defined by that lens's
readout, whose usefulness for predicting labels is tested here. Save the label-token sum,
stop-token contribution, and token count so their effects can be inspected.

Read the state immediately before each candidate token is predicted. The first prediction
uses the final prompt token. Your Jacobian notebook's `positions=[-2]` is specific to its
illustrative example. Preserve prediction-position alignment when reusing its readout code.
[Workspace Jacobian notebook](../Interpretability_Survey_Safety/notebooks/04_intermediate_computation/03_jacobian_lens.ipynb)

Check the following on a small deterministic sample of training comments before full runs.

- Every canonical path decodes to the intended complete label and fits the declared output
  budget, including its terminal event.
- The raw final-block logit-lens readout reproduces native model logits and candidate scores
  within a recorded numerical tolerance. Include the actual final normalization and any
  model-specific logit transformation, including Gemma soft-capping, exactly once.
- The tuned lens and Jacobian lens use the normalization and output-head conventions of
  their artifacts. Record any readout-wrapper adaptation needed for native-logit agreement.
- Layer indices refer to the same post-block states across methods. The final normalized
  model output and raw post-block state have distinct capture locations.
- Scores agree when candidate paths are evaluated separately or through shared prefixes,
  and when pilot examples are run individually or in a batch.
- Complete rows have finite scores for all 28 labels. Resolve failed inference rows before
  producing a full-split metric.

## Baselines and the common accuracy endpoint

| Baseline | Prediction rule | Purpose |
| --- | --- | --- |
| Greedy generation | Parse the saved EXP2 response with `strict_label_v1`. | Compare against the existing zero-shot classification behavior. |
| Native candidate scoring | Choose the complete label with maximum native `F`. | Measure the benefit of each lens under the same candidate scoring setup. |

For a split with `N` comments, every method uses the same label-match endpoint.

```text
accuracy = (1/N) * sum_i 1[prediction_i == gold_i]
delta_accuracy_pp = 100 * (accuracy_method - accuracy_baseline)
```

Compute macro-F1 over the fixed list of 28 labels. For each label `c`, use
`F1_c = 2*TP_c / (2*TP_c + FP_c + FN_c)` and average these values. An invalid greedy
prediction is a false negative for its gold emotion. Record support and use a declared
zero-division convention of zero for labels with an undefined F1 denominator.

An invalid greedy response contributes zero to the numerator. It remains in the denominator.
Candidate methods emit one legal canonical label by construction. Their comparison with
greedy therefore includes the effects of format compliance and candidate search. The native
candidate baseline measures their additional effect on emotion choice.

The paired gains satisfy the following decomposition on the same examples.

```text
accuracy_lens - accuracy_greedy =
    (accuracy_native_candidates - accuracy_greedy)
    + (accuracy_lens - accuracy_native_candidates)
```

Match baselines to the exact split and sample IDs. A lens result on 4,590 test comments is
compared with greedy accuracy on those same 4,590 comments. The published aggregate over
45,446 train, validation, and test rows answers a different population-level question.

## Experiment 1 — Direct readout at every layer

For every lens and available layer, rank the 28 complete labels by `S[r,l]`, emit the first
label, and compare it with the single annotated emotion. Break exact score ties by the
fixed canonical label order shared by every arm.

```text
prediction_direct[r,l](x) = argmax_c S[r,l](c;x)
accuracy_direct[r,l]      = mean_i 1[prediction_direct[r,l](x_i) == gold_i]
```

Evaluate all post-block layers for the logit lens and all fitted post-block source layers
for the tuned and Jacobian lenses, following the all-layer coverage policy above. Report
a common-layer panel for direct three-way comparison and the additional available layers
for each lens. Use actual artifact coverage. Plot the native final-layer score baseline
as a horizontal reference at every depth.

The primary predeclared direct-readout setting is the common fitted nonfinal layer nearest
to 75% depth. Define normalized depth as `(l + 1) / number_of_blocks` and choose the lower
index on an exact distance tie. This is an initial experimental choice, whose effect is
visible in the full curve. It is recorded before viewing accuracy.

For each layer, retain top-1 accuracy, macro-F1, per-class precision and recall, gold-label
rank, and the IDs of repairs and damages against each baseline. These tables support the
layerwise analysis below without additional model inference.

## Experiment 2 — Contrast with the native final layer

Use the same complete-label scores as Experiment 1. At a declared intermediate layer,
compute the following score separately for each lens.

```text
C[r,l,beta](c;x) = F(c;x) + beta * (F(c;x) - E[r,l](c;x))
prediction_contrast[r,l,beta](x) = argmax_c C[r,l,beta](c;x)
```

Positive beta favors candidates that gained relative support between the intermediate
readout and final scores. It can repair a fine-grained label that gained ground but still
lost the final argmax. It can also damage a correct answer when the late change favored a
wrong label. At beta zero, all contrast arms reproduce native candidate scoring exactly.

For an illustrative two-label comparison, the following values are log probabilities over
complete candidate paths. The remaining 26 candidates are still scored in the actual run.

| Label | Intermediate `E` | Final `F` | Contrast with beta 0.5 |
| --- | ---: | ---: | ---: |
| neutral | -0.7 | -0.9 | -1.0 |
| disappointment | -2.3 | -1.1 | -0.5 |

Direct intermediate and final scoring favor neutral in this pair. Positive contrast favors
disappointment because its late increase is larger. The annotation determines afterward
whether this flip was a repair or damage.

The primary contrast setting uses the same predeclared layer as Experiment 1, beta `0.5`,
and all 28 candidates. Run a fixed beta grid `0`, `0.25`, `0.5`, and `1` across every
available nonfinal post-block source layer for each lens, including early, middle, and
late depths, for descriptive curves and sensitivity heatmaps. Reuse the direct-readout
score cache. Report the declared grid without promoting its best test setting into the
headline configuration.

Compare contrast with native candidate scoring, greedy generation, direct readout at the
same lens and layer, and that lens's predeclared direct-readout configuration. The
same-layer comparison isolates the scoring rule. For dynamic contrast, the comparison
with the predeclared direct readout also measures the effect of the fixed layer-selection
rule; neither comparison uses accuracy to choose a layer.

Three additional contrast analyses help interpret the main result.

| Analysis | Definition and use |
| --- | --- |
| Sharma dynamic layer and beta | Use fixed eligible normalized depths `0.65`, `0.75`, `0.85`, and `0.95`, mapped to available nonfinal common layers. Per comment and lens, select the layer with greatest cosine distance between `P[r,l]` and native `P[final]`. Set `beta = 2 / (l_final - l_selected)`, corresponding to a target two block indices beyond the final block. Record this rule before labels are opened and report its selected-layer and beta distributions. |
| Plausibility filtering | Compare the main all-candidate arm with Sharma's relative probability mask at fixed `alpha=0.1`, using `P_final(c) >= alpha * max_j P_final(j)`. Report the fraction of gold labels excluded, including per-class exclusion. |
| Contrast within each lens | Let the upper comparison layer be the highest fitted nonfinal layer common to all three lenses. Use `E[r,upper] + 0.5 * (E[r,upper] - E[r,l])` for lower layers and compare with direct `E[r,upper]`. This tests contrast using one readout at both depths, especially where the Jacobian-to-native comparison depends on changing readout conventions. |

Treat these as secondary adaptations of Experiment 2. Their results are reported under
their declared rules. A negative-beta interpolation arm can later extend the contrast
family if the positive-contrast pilot motivates testing recovery of earlier predictions.

## Layerwise metrics and what we do with them

| Metric or analysis | Computation | Question it answers |
| --- | --- | --- |
| Top-1 accuracy by depth | Exact label-match rate for every lens and layer, and for each declared contrast setting. | Does intermediate readout or contrast beat either baseline, and where? |
| Macro-F1 by depth | Compute F1 per label and average over the fixed 28-label taxonomy. | Does the gain reach less frequent emotions? |
| Accuracy difference by depth | Paired difference from native candidate scoring and from greedy, in percentage points. | How large is the benefit at each depth? |
| Gold-label rank and recall at k | Rank all 28 labels with the same deterministic tie rule. Report mean reciprocal rank and recall at `k=1,3,5`. | Is the correct label available among leading candidates even when it loses top-1? |
| Gold-versus-rival margin | `S(gold) - max_{c != gold} S(c)` for direct readout and the analogous `C` margin for contrast. | Does the method move the correct label closer to winning? |
| Prediction stability | Fraction of comments whose predicted label changes between adjacent available layers. Record the index gap between those layers. | Where do readouts stabilize or switch? |
| Per-class performance | Recall, precision, F1, and support by layer. Use confusion matrices for the frozen headline configurations. | Which emotions benefit or deteriorate? |
| Agreement and repair overlap | Pairwise prediction agreement and intersections of repair sets across lenses at common depths. | Do the lenses recover the same errors or complementary ones? |

Use the curves to identify broad useful depth ranges and to see whether improvements
persist over neighboring layers. A test peak found after viewing the entire curve is an
exploratory maximum. Only the predeclared settings supply the current headline results.
If a plot includes that maximum, label it as selected using test outcomes.

Report the mean or median accuracy across the common layer panel only as an optional
descriptive summary. A single deployed configuration still predicts with one frozen
readout rule. Cross-model comparisons use normalized depth and record each model's actual
layer indices and artifact coverage.

Use gold-label availability across layers as an offline opportunity diagnostic.

```text
availability = mean_i 1[any evaluated layer predicts gold_i]
baseline_error_availability =
    count(baseline wrong and any layer predicts gold) / count(baseline wrong)
```

Specify the evaluated layers and lenses when reporting availability. Its value depends
on the size of that panel. It gives an upper bound for an ideal per-comment selector over
those predictions. Report the gap between availability and the best frozen configuration
to quantify potential future selection work.

To study loss and recovery across depth, follow each comment's correctness sequence over
the available layers. Record the first layer with the correct top-1 label, whether the
native final-score winner is correct, and the number of correct-to-wrong and wrong-to-correct
transitions. Include correct-final examples as well as final errors. These are observations
about readout trajectories. Claims about a particular computation causing the change would
require a subsequent intervention experiment.

Inspect full-vocabulary divergence from the native next-token distribution on a fixed
small set of candidate-prefix positions. Summarize KL divergence by layer for each lens
and compare it with label accuracy. This measures how faithfully the readout predicts
native output alongside how well it predicts the annotation. Record which prefixes and
positions were sampled so the divergence panel can be reproduced.

The planned figures and their interpretation are as follows.

| Figure | Use in the paper |
| --- | --- |
| Direct top-1 and macro-F1 curves for all three lenses | Show depth ranges that improve on the greedy and native-candidate reference lines. Mark the predeclared layer for each lens. |
| Contrast accuracy curves at beta 0.5, plus a test layer-by-beta sensitivity heatmap | Show where contrast helps and how sensitive the result is to the predefined strength and depth grid. The heatmap is descriptive and does not select a configuration. |
| Per-emotion recall heatmap over depth | Distinguish broad gains from improvement concentrated on neutral or a few frequent emotions. |
| Paired repair and damage counts for frozen configurations | Show which net gains result from useful corrections and which configurations sacrifice correct decisions. |
| Correct-label rank trajectories and repair-overlap table | Show whether the lenses expose the correct label at different depths and on different comments. Sample examples using a declared retrospective analysis rule after predictions are frozen. |

If direct readout improves but positive contrast deteriorates, examine whether useful
earlier predictions lose support late. If contrast improves while the corresponding direct
readout does poorly, inspect whether late score changes contain useful ranking information.
If gains occur only against greedy, quantify the contribution of candidate search and
format compliance through the native-candidate baseline. If top-1 improves while macro-F1
falls, identify the classes responsible for that tradeoff. Neighboring-layer stability and
the confirmation results determine how much weight to place on an isolated peak.

## Repairs, damage, and baseline interpretation

For every frozen configuration and each baseline, retain paired sample outcomes.

```text
R = count(baseline incorrect and method correct)
D = count(baseline correct and method incorrect)
accuracy_method - accuracy_baseline = (R - D) / N
repair_rate = R / count(baseline incorrect)
damage_rate = D / count(baseline correct)
```

Report undefined conditional rates explicitly when their denominator is zero. Against the
greedy baseline, break repairs into valid-but-wrong emotion outputs and invalid outputs.
Against native candidate scoring, repairs concern the selected emotion label. Also count
wrong-to-different-wrong changes, which affect confusion patterns while contributing zero
to the net accuracy difference.

Keep neutral in the task and show neutral-to-emotion and emotion-to-neutral changes with
their repair and damage counts. Report the full confusion matrix and inspect representative
repairs, damages, and unchanged errors using a declared retrospective sampling rule.
These diagnostics do not revise prediction settings. Rare-label estimates carry their
sample counts and uncertainty.

## Belrose-style calibrated readout as a secondary analysis

Belrose's classification figure follows Halawi's calibrated metric. For a `K`-way task,
Halawi evaluates whether the gold label's probability exceeds the `(K-1)/K` quantile of
that label's scores across examples. The binary version uses the median. This produces a
calibrated statistic at each layer. [Belrose et al., Figure 11](https://arxiv.org/html/2303.08112v6#S5.SS2)
and [Halawi et al., Appendix A.2](https://arxiv.org/pdf/2307.09476#page=16)

For an inductive GoEmotions adaptation, estimate thresholds once using a deterministic
unlabeled sample of up to 2,048 unique training comments, selected by text hash with seed
42. Preserve the sample IDs. The reference calculation uses the same complete-label
scoring as the evaluated arm. Freeze each threshold before test prediction.

```text
tau[arm,c] = quantile_27/28({P_arm(c;x) for x in reference_comments})
calibrated_readout_statistic =
    mean_i 1[P_arm(gold_i;x_i) > tau[arm,gold_i]]
```

For contrast, define `P_arm = softmax_over_28_labels(C)` if this diagnostic is requested.
Threshold estimation can reuse the cached scores for the selected arms. Several labels
can exceed their respective thresholds on one example. Report the fraction exceeding
threshold and class frequencies alongside the statistic, since the GoEmotions population
is imbalanced. Ordinary top-1 accuracy remains the common classification endpoint for
comparison with greedy decoding.

## Uncertainty, robustness, and compute

Calculate 95% bootstrap confidence intervals for accuracy and macro-F1 and their paired differences,
resampling duplicate-text groups within the evaluated split. Use the same resampled groups
for every compared arm and 2,000 replicates with a recorded random seed. Layerwise curve
intervals are pointwise. Use simultaneous bands or account for the full comparison family
if making claims about significance across a searched depth range.

The initial primary hypothesis family consists of the six predeclared configurations
against native candidate scoring, one direct and one contrast configuration for each
lens. If reporting significance tests, account for these six comparisons with a declared
procedure such as Holm correction. Report paired effect sizes and intervals against greedy
as well. Identify exploratory configurations from the full layer and beta scans.

Use the following bounded, predeclared checks to understand the results.

- Compare scores including the declared stop event with label-token-only scores. Record
  how many label choices change and whether gains concentrate on a preferred termination.
- Report performance by candidate token length. An optional length-normalized score is a
  separate declared scoring arm with its own matched native baseline.
- Rerun the predeclared methods and native candidate baseline under a fixed alternate label
  menu order. Recompute readouts on that prompt. Compare their paired changes.
- Repeat a small fixed sample with another placeholder-free prompt formulation if testing
  prompt robustness. Apply every compared readout to the same formulation.
- Report wall-clock time, peak device memory, candidate-prefix forwards, and the exact
  GPU. Separate model/lens loading, score extraction, and cached contrast calculations.
  Include every candidate path in the online cost. Measure early-exit savings only if
  the implementation actually stops the backbone at the readout layer.

## Run order and saved outputs

1. Verify the model, tokenizer, data, lens artifacts, and EXP2 prediction provenance. Run
   the technical checks on 64 deterministic training comments, verify complete layer
   coverage and index mapping, and measure extraction throughput and peak memory without
   consulting gold labels to choose settings.
2. Freeze the full source-layer lists, common-layer intersection, primary layer-mapping
   rule, beta grid, dynamic comparison-layer panel and target offset, masks, scoring
   conventions, robustness checks, baseline sample IDs, and reference-sample IDs. Save
   the experiment configuration before evaluating accuracy.
3. Estimate any optional reference statistics using the declared unlabeled training
   comments. Leave validation-based configuration selection for future work.
4. Extract complete-label scores at every declared layer on test comments, sharing
   backbone computations across lenses. Verify that every required sample-layer-label
   combination is present. Compute direct-readout and contrast metrics from that cache,
   including the predeclared configurations, full layer curves, and beta sensitivity
   heatmaps. Do not revise the configuration shortlist using those results.
5. Interpret the class-level results and trajectory diagnostics. Replicate the main
   comparisons on the second model using the same protocol and checkpoint-specific lenses.

Store generated artifacts under `outputs/m03_lens_comparison/<model>/<run-id>/`. Use the
project's structured logger and `logs/<model-and-run-identifier>/` for operational logs.
Every essential plot has its own tidy, self-contained CSV, and plotting code consumes that
CSV. Preserve enough context to reconstruct the comparisons without model inference.

| Artifact | Required contents |
| --- | --- |
| Run manifest | Model, tokenizer, dataset and lens revisions, software, prompt hashes, full source-layer lists and index mappings, common-layer intersection, any missing layers and reasons, normalization conventions, terminal event, seeds, and hardware. |
| Complete-label score table | Sample and split IDs, lens, layer, label, token count, label-token score, stop score, total score, and native reference score. |
| Prediction table | Sample ID, gold label, arm and settings, prediction, correctness, both baseline predictions and their validity, gold rank, and top-two margin. |
| Metrics table | Split, model, arm, layer, beta, sample count, accuracy, macro-F1, differences in percentage points, repair and damage counts, and confidence intervals. |
| Class metrics table | Label, support, predicted count, precision, recall, F1, and the configuration and split identifiers. |
| Plot-ready tables | Values and context for every plotted layer curve, beta heatmap, confusion matrix, repair overlap, or rank trajectory. |
| Runtime table | GPU, memory, batch size, number of examples and prefixes, readout layers, extraction time, and per-comment latency. |

Cache entries are identified by the model, lens, prompt, sample, token-path, and readout
configuration. Resumed extraction verifies these fields and the completeness of the
28-label rows before reusing them.

The main paper table has rows for both baselines and the six predeclared label-free lens
methods. It reports top-1 accuracy, macro-F1, gains against both baselines, repairs, damages,
and runtime. Report the predefined dynamic contrast and other secondary label-free analyses
separately. The layerwise curves and secondary analyses explain where those results arise
and which errors they affect. Validation-tuned results are outside the current study.

## Future work — validation-tuned variants

These extensions are deferred and are not required for the current label-free experiments,
run sequence, figures, or main results table. They would use GoEmotions gold labels for
hyperparameter selection while keeping the backbone and fitted lenses frozen, so they must
be reported as validation-tuned rather than label-free.

| Future variant | Selection protocol |
| --- | --- |
| Direct readout | For each model and lens, choose one layer by validation top-1 accuracy. Break ties by macro-F1 and then lower layer index. Freeze the layer before test evaluation. |
| Layer contrast | For each model and lens, select `(layer, beta)` by validation top-1 accuracy over a predefined search space, initially the available nonfinal layers and beta `0`, `0.25`, `0.5`, and `1`. Break ties by macro-F1, smaller beta, and lower layer index. Freeze the pair before test evaluation. |

A future run would first extract complete-label scores on validation and make a validation
layer-by-beta heatmap. After selecting and freezing the configurations, compare tuned
contrast with tuned direct readout, direct readout at the same layer, the fixed label-free
methods, and both baselines on matched test comments. Record the search budget and report
these results separately from the label-free study.

If the official test results have already been examined during the current study, a later
evaluation on that same test set is a follow-up comparison. Reserve an untouched evaluation
set for confirmatory claims about methods developed after inspecting those results.

## Related local documents

- [Earlier M03 experiment portfolio](m-03-doc.md)
- [Lens definitions and shared evaluation protocol](docs/lens-methods.md)
- [Complete-label layer contrast proposal](docs/notes-7.md)
- [Original EXP2 zero-shot classification protocol](docs/exp2.md)
