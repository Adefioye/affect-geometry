# Emotion classification: four-model results

Analysis date: 2026-10-27. Saved full runs only; pilots are excluded.

Quick links: [Overall results](#overall-and-test-set-performance) · [Gemma-2](#gemma-2) · [LLaMA-2](#llama-2) · [Apertus](#apertus) · [OLMo-2](#olmo-2) · [Limitations](#conclusions-and-limitations)

## Main findings

All four models completed the same **45,446 single-label GoEmotions examples**, including neutral, with no unresolved inference errors or truncated inputs. OLMo-2 has the highest observed strict accuracy (26.242%), followed by Apertus (22.684%), Gemma-2 (8.788%), and LLaMA-2 (0.004%). These are results for these checkpoints, prompts and parsers, not an architecture-level ranking.

**Operational completion did not imply successful classification.** LLaMA-2 produced invalid labels on 99.987% of examples; Gemma-2 did so on 61.829%. The base checkpoints use plain completion prompts, whereas Apertus and OLMo-2 are instruction checkpoints with native chat templates. Prompt-following and output formatting therefore confound comparisons of emotion recognition.

An always-neutral predictor would be correct on **16,021 / 45,446 = 35.253%** overall and **1,606 / 4,590 = 34.989%** on test. Every evaluated model falls below this accuracy baseline, so the numerical ranking alone is not evidence of a useful classifier. The class imbalance also makes macro metrics necessary.

## Data, protocol and validation

| Model | Checkpoint | Prompt format | Model revision |
| --- | --- | --- | --- |
| Gemma-2 | `google/gemma-2-9b` | plain completion | `33c193028431c2fde6c6e51f29e6f17b60cbfac6` |
| LLaMA-2 | `meta-llama/Llama-2-7b-hf` | plain completion | `01c7f73d771dfac7d292323805ebc428287df4f9` |
| Apertus | `swiss-ai/Apertus-8B-Instruct-2509` | native chat | `b946d40447b2b597999b9c86d44bee0b452c919f` |
| OLMo-2 | `allenai/OLMo-2-1124-7B-Instruct` | native chat | `470b1fba1ae01581f270116362ee4aa1b97f4c84` |

Dataset: `google-research-datasets/go_emotions` (`simplified`), revision `add492243ff905527e67aeb8b80c082af02207c3`. Protocol: `rq2_single_label_v1`; run ID: `full`; seed 42; bfloat16; batch size 32; deterministic greedy decoding; maximum 10 new tokens. Label order is shuffled deterministically per sample and matches across models. No training or fine-tuning is performed in this pipeline. The original train split is an evaluation population, not a training procedure.

| Original split | Source examples | Retained single-label examples | Excluded multi-label examples |
| --- | --- | --- | --- |
| train | 43,410 | 36,308 | 7,102 |
| validation | 5,426 | 4,548 | 878 |
| test | 5,427 | 4,590 | 837 |

Validation independently rescored all **181,784** raw responses, reconciled all 464 split/emotion summary rows, confusion counts and macro metrics, checked saved sample-outcome records, and verified identical sample IDs, text hashes, true labels, split membership, label order, dataset revision and code hash across models. All checks passed. Each split includes all 28 labels. Correct + incorrect = completed = total in every group. No missing results were silently removed.

There are 45,213 distinct text hashes; 146 texts occur multiple times (379 rows), and 61 texts appear across original splits. Original splits are preserved. Paired bootstrap comparisons group identical texts together; ordinary Wilson intervals below do not adjust for these duplicates.

### Metric definitions

- **Correct / incorrect %:** count divided by all completed examples in that exact split/emotion row, multiplied by 100. Incorrect includes wrong valid labels and invalid responses. Completion coverage is 100%, so the total and completed denominators coincide.
- **Strict valid label:** the entire normalized response must be one of the 28 allowed labels; trimming and punctuation rules follow `strict_label_v1`. Invalid responses count as incorrect.
- **First-word diagnostic:** the saved `choi_weber_first_word_v1` parser can accept an allowed first word even if extra output follows. It is a sensitivity analysis, not a replacement for strict scoring.
- **Macro precision/recall/F1:** unweighted mean over all 28 labels; undefined scores use zero. Per-emotion correct % is recall for that emotion.
- **95% Wilson CI:** descriptive binomial interval for strict accuracy. **pp** means percentage points. Percentages are rounded to three decimals; counts are exact.

## Overall and test-set performance

| Model | Correct / total | Correct % | Incorrect % | 95% Wilson CI | Macro F1 % |
| --- | --- | --- | --- | --- | --- |
| Gemma-2 | 3,994 / 45,446 | 8.788% | 91.212% | 8.532–9.052% | 12.786% |
| LLaMA-2 | 2 / 45,446 | 0.004% | 99.996% | 0.001–0.016% | 0.014% |
| Apertus | 10,309 / 45,446 | 22.684% | 77.316% | 22.301–23.071% | 21.742% |
| OLMo-2 | 11,926 / 45,446 | 26.242% | 73.758% | 25.840–26.649% | 25.369% |

| Model | Test correct / total | Test correct % | 95% Wilson CI | Test macro precision % | Test macro recall % | Test macro F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma-2 | 384 / 4,590 | 8.366% | 7.599–9.202% | 31.933% | 11.698% | 12.112% |
| LLaMA-2 | 0 / 4,590 | 0.000% | 0.000–0.084% | 0.000% | 0.000% | 0.000% |
| Apertus | 1,036 / 4,590 | 22.571% | 21.385–23.803% | 30.511% | 26.015% | 20.641% |
| OLMo-2 | 1,223 / 4,590 | 26.645% | 25.386–27.943% | 30.972% | 31.377% | 24.643% |

The strict accuracy ordering is the same on test and on the combined population. The test split avoids pooling in original train/validation rows, but it is not proof of absence of pretraining contamination or prompt-selection bias. Macro F1 distinguishes label coverage from accuracy dominated by frequent classes.

### Formatting and stopping diagnostics

| Model | Wrong valid label n (%) | Invalid n (%) | First-word correct % | Strict → first-word gain (pp) | At token cap n (%) | Blank n |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma-2 | 13,353 (29.382%) | 28,099 (61.829%) | 17.482% | 8.694 | 13,918 (30.625%) | 13,473 |
| LLaMA-2 | 4 (0.009%) | 45,440 (99.987%) | 3.855% | 3.851 | 45,279 (99.633%) | 135 |
| Apertus | 34,121 (75.080%) | 1,016 (2.236%) | 22.686% | 0.002 | 28 (0.062%) | 0 |
| OLMo-2 | 32,816 (72.209%) | 704 (1.549%) | 26.242% | 0.000 | 0 (0.000%) | 0 |

First-word scoring increases Gemma-2 and LLaMA-2 accuracy, consistent with output-format failures contributing to their low strict results. It does not recover competitive accuracy for these runs. Reaching the token cap is a stopping diagnostic, not by itself an error or proof that a correct label would appear with a larger budget. No inputs were truncated.

![Response outcomes](../outputs/rq2_classification/comparison/analysis-report/plots/response-outcomes.png)

The panels show the shares of correct responses (green), wrong allowed labels (amber), and invalid output (red), for all splits and test respectively. LLaMA-2 is almost entirely invalid; Gemma-2 is mostly invalid; the instruction models are mostly valid but often choose a wrong label. These bars cannot separate underlying emotion knowledge from prompt-following ability.

## Paired model comparisons

| Model A | Model B | A − B correct (pp), all examples | 95% paired bootstrap CI (pp) |
| --- | --- | --- | --- |
| google/gemma-2-9b | meta-llama/Llama-2-7b-hf | 8.784 | [8.523, 9.055] |
| google/gemma-2-9b | swiss-ai/Apertus-8B-Instruct-2509 | -13.896 | [-14.306, -13.501] |
| google/gemma-2-9b | allenai/OLMo-2-1124-7B-Instruct | -17.454 | [-17.877, -17.028] |
| meta-llama/Llama-2-7b-hf | swiss-ai/Apertus-8B-Instruct-2509 | -22.680 | [-23.058, -22.298] |
| meta-llama/Llama-2-7b-hf | allenai/OLMo-2-1124-7B-Instruct | -26.238 | [-26.651, -25.838] |
| swiss-ai/Apertus-8B-Instruct-2509 | allenai/OLMo-2-1124-7B-Instruct | -3.558 | [-3.927, -3.161] |

| Model A | Model B | A − B correct (pp), test only | 95% paired bootstrap CI (pp) |
| --- | --- | --- | --- |
| Gemma-2 | LLaMA-2 | 8.366 | [7.582, 9.214] |
| Gemma-2 | Apertus | -14.205 | [-15.451, -13.028] |
| Gemma-2 | OLMo-2 | -18.279 | [-19.582, -16.928] |
| LLaMA-2 | Apertus | -22.571 | [-23.823, -21.386] |
| LLaMA-2 | OLMo-2 | -26.645 | [-27.889, -25.414] |
| Apertus | OLMo-2 | -4.074 | [-5.253, -2.922] |

Intervals use 1,000 paired cluster bootstrap resamples, seed 42, with identical texts sampled as a unit. Combined-population intervals come from the repository comparison command; test-only intervals were computed from the same saved predictions. These are exploratory, unadjusted 95% intervals; no correction for six pairwise comparisons or per-emotion exploration was applied. They quantify sampling variation for these responses, not variation across prompt designs, model seeds or training histories. The OLMo-2 versus Apertus difference favors OLMo-2 under this protocol, with output formats and training types still confounded.

## Emotion-level overview

![Strict accuracy by emotion](../outputs/rq2_classification/comparison/analysis-report/plots/emotion-accuracy.png)

Rows are true emotions, columns are models, and color/text encode strict correct percentages over all splits on a common 0–100% scale. Frequent neutral examples have low recall despite their large support, while several more explicit emotions have much higher recall. Rare classes can have unstable percentages; the detailed tables below give every denominator. The plot describes prediction outcomes, not an emotion geometry.

## Gemma-2

Checkpoint: `google/gemma-2-9b`. Plain completion prompt; base checkpoint.

| Split | Total | Correct n | Incorrect n | Correct % | Incorrect % | Invalid n | Invalid % | Macro F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 45,446 | 3,994 | 41,452 | 8.788% | 91.212% | 28,099 | 61.829% | 12.786% |
| train | 36,308 | 3,199 | 33,109 | 8.811% | 91.189% | 22,391 | 61.670% | 12.655% |
| validation | 4,548 | 411 | 4,137 | 9.037% | 90.963% | 2,832 | 62.269% | 13.978% |
| test | 4,590 | 384 | 4,206 | 8.366% | 91.634% | 2,876 | 62.658% | 12.112% |

Highest observed all-data recall: **sadness 38.285%** (384/1,003); lowest: **realization 0.801%** (6/749). Neutral recall is **5.449%** (873/16,021). These are descriptive extremes selected after inspection; small supports require caution.

### Correct and incorrect by emotion — all splits combined

| Emotion | Total | Correct n | Correct % | Incorrect n | Incorrect % |
| --- | --- | --- | --- | --- | --- |
| admiration | 3,384 | 406 | 11.998% | 2,978 | 88.002% |
| amusement | 2,046 | 208 | 10.166% | 1,838 | 89.834% |
| anger | 1,265 | 348 | 27.510% | 917 | 72.490% |
| annoyance | 1,809 | 33 | 1.824% | 1,776 | 98.176% |
| approval | 2,367 | 28 | 1.183% | 2,339 | 98.817% |
| caring | 831 | 7 | 0.842% | 824 | 99.158% |
| confusion | 1,057 | 92 | 8.704% | 965 | 91.296% |
| curiosity | 1,729 | 93 | 5.379% | 1,636 | 94.621% |
| desire | 497 | 4 | 0.805% | 493 | 99.195% |
| disappointment | 888 | 148 | 16.667% | 740 | 83.333% |
| disapproval | 1,809 | 73 | 4.035% | 1,736 | 95.965% |
| disgust | 635 | 164 | 25.827% | 471 | 74.173% |
| embarrassment | 246 | 71 | 28.862% | 175 | 71.138% |
| excitement | 619 | 89 | 14.378% | 530 | 85.622% |
| fear | 553 | 105 | 18.987% | 448 | 81.013% |
| gratitude | 2,378 | 177 | 7.443% | 2,201 | 92.557% |
| grief | 47 | 2 | 4.255% | 45 | 95.745% |
| joy | 1,052 | 350 | 33.270% | 702 | 66.730% |
| love | 1,760 | 85 | 4.830% | 1,675 | 95.170% |
| nervousness | 105 | 7 | 6.667% | 98 | 93.333% |
| optimism | 1,087 | 9 | 0.828% | 1,078 | 99.172% |
| pride | 67 | 19 | 28.358% | 48 | 71.642% |
| realization | 749 | 6 | 0.801% | 743 | 99.199% |
| relief | 103 | 15 | 14.563% | 88 | 85.437% |
| remorse | 437 | 35 | 8.009% | 402 | 91.991% |
| sadness | 1,003 | 384 | 38.285% | 619 | 61.715% |
| surprise | 902 | 163 | 18.071% | 739 | 81.929% |
| neutral | 16,021 | 873 | 5.449% | 15,148 | 94.551% |
| **TOTAL** | 45,446 | 3,994 | 8.788% | 41,452 | 91.212% |

This table uses only the combined train, validation and test populations; each percentage uses its own emotion-row total. The TOTAL row matches the overall comparison above.

### Correct and incorrect by emotion — train

| Emotion | Total | Correct n | Correct % | Incorrect n | Incorrect % |
| --- | --- | --- | --- | --- | --- |
| admiration | 2,710 | 321 | 11.845% | 2,389 | 88.155% |
| amusement | 1,652 | 166 | 10.048% | 1,486 | 89.952% |
| anger | 1,025 | 281 | 27.415% | 744 | 72.585% |
| annoyance | 1,451 | 25 | 1.723% | 1,426 | 98.277% |
| approval | 1,873 | 21 | 1.121% | 1,852 | 98.879% |
| caring | 649 | 5 | 0.770% | 644 | 99.230% |
| confusion | 858 | 75 | 8.741% | 783 | 91.259% |
| curiosity | 1,389 | 77 | 5.544% | 1,312 | 94.456% |
| desire | 389 | 3 | 0.771% | 386 | 99.229% |
| disappointment | 709 | 125 | 17.630% | 584 | 82.370% |
| disapproval | 1,402 | 57 | 4.066% | 1,345 | 95.934% |
| disgust | 498 | 126 | 25.301% | 372 | 74.699% |
| embarrassment | 203 | 55 | 27.094% | 148 | 72.906% |
| excitement | 510 | 75 | 14.706% | 435 | 85.294% |
| fear | 430 | 79 | 18.372% | 351 | 81.628% |
| gratitude | 1,857 | 139 | 7.485% | 1,718 | 92.515% |
| grief | 39 | 1 | 2.564% | 38 | 97.436% |
| joy | 853 | 270 | 31.653% | 583 | 68.347% |
| love | 1,427 | 72 | 5.046% | 1,355 | 94.954% |
| nervousness | 85 | 6 | 7.059% | 79 | 92.941% |
| optimism | 861 | 9 | 1.045% | 852 | 98.955% |
| pride | 51 | 14 | 27.451% | 37 | 72.549% |
| realization | 586 | 4 | 0.683% | 582 | 99.317% |
| relief | 88 | 13 | 14.773% | 75 | 85.227% |
| remorse | 353 | 30 | 8.499% | 323 | 91.501% |
| sadness | 817 | 314 | 38.433% | 503 | 61.567% |
| surprise | 720 | 131 | 18.194% | 589 | 81.806% |
| neutral | 12,823 | 705 | 5.498% | 12,118 | 94.502% |
| **TOTAL** | 36,308 | 3,199 | 8.811% | 33,109 | 91.189% |

This table uses only the original train split; each percentage uses its own emotion-row total. Differences from other splits are descriptive; rare emotion percentages can move substantially with a few predictions.

### Correct and incorrect by emotion — validation

| Emotion | Total | Correct n | Correct % | Incorrect n | Incorrect % |
| --- | --- | --- | --- | --- | --- |
| admiration | 326 | 42 | 12.883% | 284 | 87.117% |
| amusement | 208 | 20 | 9.615% | 188 | 90.385% |
| anger | 109 | 35 | 32.110% | 74 | 67.890% |
| annoyance | 164 | 3 | 1.829% | 161 | 98.171% |
| approval | 258 | 2 | 0.775% | 256 | 99.225% |
| caring | 96 | 2 | 2.083% | 94 | 97.917% |
| confusion | 102 | 8 | 7.843% | 94 | 92.157% |
| curiosity | 164 | 8 | 4.878% | 156 | 95.122% |
| desire | 52 | 1 | 1.923% | 51 | 98.077% |
| disappointment | 91 | 10 | 10.989% | 81 | 89.011% |
| disapproval | 212 | 7 | 3.302% | 205 | 96.698% |
| disgust | 61 | 24 | 39.344% | 37 | 60.656% |
| embarrassment | 20 | 9 | 45.000% | 11 | 55.000% |
| excitement | 52 | 5 | 9.615% | 47 | 90.385% |
| fear | 58 | 9 | 15.517% | 49 | 84.483% |
| gratitude | 261 | 23 | 8.812% | 238 | 91.188% |
| grief | 6 | 1 | 16.667% | 5 | 83.333% |
| joy | 106 | 41 | 38.679% | 65 | 61.321% |
| love | 173 | 7 | 4.046% | 166 | 95.954% |
| nervousness | 8 | 0 | 0.000% | 8 | 100.000% |
| optimism | 119 | 0 | 0.000% | 119 | 100.000% |
| pride | 9 | 4 | 44.444% | 5 | 55.556% |
| realization | 74 | 0 | 0.000% | 74 | 100.000% |
| relief | 8 | 1 | 12.500% | 7 | 87.500% |
| remorse | 40 | 3 | 7.500% | 37 | 92.500% |
| sadness | 84 | 35 | 41.667% | 49 | 58.333% |
| surprise | 95 | 18 | 18.947% | 77 | 81.053% |
| neutral | 1,592 | 93 | 5.842% | 1,499 | 94.158% |
| **TOTAL** | 4,548 | 411 | 9.037% | 4,137 | 90.963% |

This table uses only the original validation split; each percentage uses its own emotion-row total. Differences from other splits are descriptive; rare emotion percentages can move substantially with a few predictions.

### Correct and incorrect by emotion — test

| Emotion | Total | Correct n | Correct % | Incorrect n | Incorrect % |
| --- | --- | --- | --- | --- | --- |
| admiration | 348 | 43 | 12.356% | 305 | 87.644% |
| amusement | 186 | 22 | 11.828% | 164 | 88.172% |
| anger | 131 | 32 | 24.427% | 99 | 75.573% |
| annoyance | 194 | 5 | 2.577% | 189 | 97.423% |
| approval | 236 | 5 | 2.119% | 231 | 97.881% |
| caring | 86 | 0 | 0.000% | 86 | 100.000% |
| confusion | 97 | 9 | 9.278% | 88 | 90.722% |
| curiosity | 176 | 8 | 4.545% | 168 | 95.455% |
| desire | 56 | 0 | 0.000% | 56 | 100.000% |
| disappointment | 88 | 13 | 14.773% | 75 | 85.227% |
| disapproval | 195 | 9 | 4.615% | 186 | 95.385% |
| disgust | 76 | 14 | 18.421% | 62 | 81.579% |
| embarrassment | 23 | 7 | 30.435% | 16 | 69.565% |
| excitement | 57 | 9 | 15.789% | 48 | 84.211% |
| fear | 65 | 17 | 26.154% | 48 | 73.846% |
| gratitude | 260 | 15 | 5.769% | 245 | 94.231% |
| grief | 2 | 0 | 0.000% | 2 | 100.000% |
| joy | 93 | 39 | 41.935% | 54 | 58.065% |
| love | 160 | 6 | 3.750% | 154 | 96.250% |
| nervousness | 12 | 1 | 8.333% | 11 | 91.667% |
| optimism | 107 | 0 | 0.000% | 107 | 100.000% |
| pride | 7 | 1 | 14.286% | 6 | 85.714% |
| realization | 89 | 2 | 2.247% | 87 | 97.753% |
| relief | 7 | 1 | 14.286% | 6 | 85.714% |
| remorse | 44 | 2 | 4.545% | 42 | 95.455% |
| sadness | 102 | 35 | 34.314% | 67 | 65.686% |
| surprise | 87 | 14 | 16.092% | 73 | 83.908% |
| neutral | 1,606 | 75 | 4.670% | 1,531 | 95.330% |
| **TOTAL** | 4,590 | 384 | 8.366% | 4,206 | 91.634% |

This table uses only the original test split; each percentage uses its own emotion-row total. Differences from other splits are descriptive; rare emotion percentages can move substantially with a few predictions.

### Main wrong-label confusions (all splits)

| True emotion | Wrong predicted label | Count | % of that true emotion |
| --- | --- | --- | --- |
| neutral | joy | 1,148 | 7.166% |
| neutral | anger | 745 | 4.650% |
| admiration | joy | 698 | 20.626% |
| neutral | sadness | 572 | 3.570% |
| gratitude | joy | 389 | 16.358% |

These are the largest off-diagonal counts among valid labels; INVALID responses are excluded here and reported separately above. Counts favor frequent true classes and should be read with their within-emotion percentages.

### Examples of invalid raw responses

| Raw response (escaped, up to 180 characters) | Count among invalid outputs |
| --- | --- |
| `\n` | 13,291 |
| ` neutral\n\nText: I'm not sure if` | 572 |
| ` [EMOTION]\n` | 460 |

These are the three most frequent exact invalid responses, with newlines escaped. They diagnose response formatting or use of labels outside the allowed vocabulary; they do not establish the cause.

### Inspect individual correct and incorrect examples

[Sample-level outcome table](../outputs/rq2_classification/gemma2/full/analyses/sample_outcomes.csv) contains every example with `correct`, `outcome`, true/predicted emotion, original text and raw response. Filter `correct=True` for correct data and `correct=False` for incorrect data; within the latter, `incorrect_label` and `invalid_output` distinguish errors. [Saved sample indexes](../outputs/rq2_classification/gemma2/full/analyses/sample_indexes.json) groups sample IDs by split, emotion and Boolean outcome. [Per-label precision/recall/F1](../outputs/rq2_classification/gemma2/full/analyses/label_metrics.csv) and [Complete confusion table](../outputs/rq2_classification/gemma2/full/analyses/confusion.csv) retain the full detail. No redundant copies of these canonical files were created.

## LLaMA-2

Checkpoint: `meta-llama/Llama-2-7b-hf`. Plain completion prompt; base checkpoint.

| Split | Total | Correct n | Incorrect n | Correct % | Incorrect % | Invalid n | Invalid % | Macro F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 45,446 | 2 | 45,444 | 0.004% | 99.996% | 45,440 | 99.987% | 0.014% |
| train | 36,308 | 2 | 36,306 | 0.006% | 99.994% | 36,302 | 99.983% | 0.017% |
| validation | 4,548 | 0 | 4,548 | 0.000% | 100.000% | 4,548 | 100.000% | 0.000% |
| test | 4,590 | 0 | 4,590 | 0.000% | 100.000% | 4,590 | 100.000% | 0.000% |

Highest observed all-data recall: **sadness 0.199%** (2/1,003); lowest: **neutral 0.000%** (0/16,021). Neutral recall is **0.000%** (0/16,021). These are descriptive extremes selected after inspection; small supports require caution.

### Correct and incorrect by emotion — all splits combined

| Emotion | Total | Correct n | Correct % | Incorrect n | Incorrect % |
| --- | --- | --- | --- | --- | --- |
| admiration | 3,384 | 0 | 0.000% | 3,384 | 100.000% |
| amusement | 2,046 | 0 | 0.000% | 2,046 | 100.000% |
| anger | 1,265 | 0 | 0.000% | 1,265 | 100.000% |
| annoyance | 1,809 | 0 | 0.000% | 1,809 | 100.000% |
| approval | 2,367 | 0 | 0.000% | 2,367 | 100.000% |
| caring | 831 | 0 | 0.000% | 831 | 100.000% |
| confusion | 1,057 | 0 | 0.000% | 1,057 | 100.000% |
| curiosity | 1,729 | 0 | 0.000% | 1,729 | 100.000% |
| desire | 497 | 0 | 0.000% | 497 | 100.000% |
| disappointment | 888 | 0 | 0.000% | 888 | 100.000% |
| disapproval | 1,809 | 0 | 0.000% | 1,809 | 100.000% |
| disgust | 635 | 0 | 0.000% | 635 | 100.000% |
| embarrassment | 246 | 0 | 0.000% | 246 | 100.000% |
| excitement | 619 | 0 | 0.000% | 619 | 100.000% |
| fear | 553 | 0 | 0.000% | 553 | 100.000% |
| gratitude | 2,378 | 0 | 0.000% | 2,378 | 100.000% |
| grief | 47 | 0 | 0.000% | 47 | 100.000% |
| joy | 1,052 | 0 | 0.000% | 1,052 | 100.000% |
| love | 1,760 | 0 | 0.000% | 1,760 | 100.000% |
| nervousness | 105 | 0 | 0.000% | 105 | 100.000% |
| optimism | 1,087 | 0 | 0.000% | 1,087 | 100.000% |
| pride | 67 | 0 | 0.000% | 67 | 100.000% |
| realization | 749 | 0 | 0.000% | 749 | 100.000% |
| relief | 103 | 0 | 0.000% | 103 | 100.000% |
| remorse | 437 | 0 | 0.000% | 437 | 100.000% |
| sadness | 1,003 | 2 | 0.199% | 1,001 | 99.801% |
| surprise | 902 | 0 | 0.000% | 902 | 100.000% |
| neutral | 16,021 | 0 | 0.000% | 16,021 | 100.000% |
| **TOTAL** | 45,446 | 2 | 0.004% | 45,444 | 99.996% |

This table uses only the combined train, validation and test populations; each percentage uses its own emotion-row total. The TOTAL row matches the overall comparison above.

### Correct and incorrect by emotion — train

| Emotion | Total | Correct n | Correct % | Incorrect n | Incorrect % |
| --- | --- | --- | --- | --- | --- |
| admiration | 2,710 | 0 | 0.000% | 2,710 | 100.000% |
| amusement | 1,652 | 0 | 0.000% | 1,652 | 100.000% |
| anger | 1,025 | 0 | 0.000% | 1,025 | 100.000% |
| annoyance | 1,451 | 0 | 0.000% | 1,451 | 100.000% |
| approval | 1,873 | 0 | 0.000% | 1,873 | 100.000% |
| caring | 649 | 0 | 0.000% | 649 | 100.000% |
| confusion | 858 | 0 | 0.000% | 858 | 100.000% |
| curiosity | 1,389 | 0 | 0.000% | 1,389 | 100.000% |
| desire | 389 | 0 | 0.000% | 389 | 100.000% |
| disappointment | 709 | 0 | 0.000% | 709 | 100.000% |
| disapproval | 1,402 | 0 | 0.000% | 1,402 | 100.000% |
| disgust | 498 | 0 | 0.000% | 498 | 100.000% |
| embarrassment | 203 | 0 | 0.000% | 203 | 100.000% |
| excitement | 510 | 0 | 0.000% | 510 | 100.000% |
| fear | 430 | 0 | 0.000% | 430 | 100.000% |
| gratitude | 1,857 | 0 | 0.000% | 1,857 | 100.000% |
| grief | 39 | 0 | 0.000% | 39 | 100.000% |
| joy | 853 | 0 | 0.000% | 853 | 100.000% |
| love | 1,427 | 0 | 0.000% | 1,427 | 100.000% |
| nervousness | 85 | 0 | 0.000% | 85 | 100.000% |
| optimism | 861 | 0 | 0.000% | 861 | 100.000% |
| pride | 51 | 0 | 0.000% | 51 | 100.000% |
| realization | 586 | 0 | 0.000% | 586 | 100.000% |
| relief | 88 | 0 | 0.000% | 88 | 100.000% |
| remorse | 353 | 0 | 0.000% | 353 | 100.000% |
| sadness | 817 | 2 | 0.245% | 815 | 99.755% |
| surprise | 720 | 0 | 0.000% | 720 | 100.000% |
| neutral | 12,823 | 0 | 0.000% | 12,823 | 100.000% |
| **TOTAL** | 36,308 | 2 | 0.006% | 36,306 | 99.994% |

This table uses only the original train split; each percentage uses its own emotion-row total. Differences from other splits are descriptive; rare emotion percentages can move substantially with a few predictions.

### Correct and incorrect by emotion — validation

| Emotion | Total | Correct n | Correct % | Incorrect n | Incorrect % |
| --- | --- | --- | --- | --- | --- |
| admiration | 326 | 0 | 0.000% | 326 | 100.000% |
| amusement | 208 | 0 | 0.000% | 208 | 100.000% |
| anger | 109 | 0 | 0.000% | 109 | 100.000% |
| annoyance | 164 | 0 | 0.000% | 164 | 100.000% |
| approval | 258 | 0 | 0.000% | 258 | 100.000% |
| caring | 96 | 0 | 0.000% | 96 | 100.000% |
| confusion | 102 | 0 | 0.000% | 102 | 100.000% |
| curiosity | 164 | 0 | 0.000% | 164 | 100.000% |
| desire | 52 | 0 | 0.000% | 52 | 100.000% |
| disappointment | 91 | 0 | 0.000% | 91 | 100.000% |
| disapproval | 212 | 0 | 0.000% | 212 | 100.000% |
| disgust | 61 | 0 | 0.000% | 61 | 100.000% |
| embarrassment | 20 | 0 | 0.000% | 20 | 100.000% |
| excitement | 52 | 0 | 0.000% | 52 | 100.000% |
| fear | 58 | 0 | 0.000% | 58 | 100.000% |
| gratitude | 261 | 0 | 0.000% | 261 | 100.000% |
| grief | 6 | 0 | 0.000% | 6 | 100.000% |
| joy | 106 | 0 | 0.000% | 106 | 100.000% |
| love | 173 | 0 | 0.000% | 173 | 100.000% |
| nervousness | 8 | 0 | 0.000% | 8 | 100.000% |
| optimism | 119 | 0 | 0.000% | 119 | 100.000% |
| pride | 9 | 0 | 0.000% | 9 | 100.000% |
| realization | 74 | 0 | 0.000% | 74 | 100.000% |
| relief | 8 | 0 | 0.000% | 8 | 100.000% |
| remorse | 40 | 0 | 0.000% | 40 | 100.000% |
| sadness | 84 | 0 | 0.000% | 84 | 100.000% |
| surprise | 95 | 0 | 0.000% | 95 | 100.000% |
| neutral | 1,592 | 0 | 0.000% | 1,592 | 100.000% |
| **TOTAL** | 4,548 | 0 | 0.000% | 4,548 | 100.000% |

This table uses only the original validation split; each percentage uses its own emotion-row total. Differences from other splits are descriptive; rare emotion percentages can move substantially with a few predictions.

### Correct and incorrect by emotion — test

| Emotion | Total | Correct n | Correct % | Incorrect n | Incorrect % |
| --- | --- | --- | --- | --- | --- |
| admiration | 348 | 0 | 0.000% | 348 | 100.000% |
| amusement | 186 | 0 | 0.000% | 186 | 100.000% |
| anger | 131 | 0 | 0.000% | 131 | 100.000% |
| annoyance | 194 | 0 | 0.000% | 194 | 100.000% |
| approval | 236 | 0 | 0.000% | 236 | 100.000% |
| caring | 86 | 0 | 0.000% | 86 | 100.000% |
| confusion | 97 | 0 | 0.000% | 97 | 100.000% |
| curiosity | 176 | 0 | 0.000% | 176 | 100.000% |
| desire | 56 | 0 | 0.000% | 56 | 100.000% |
| disappointment | 88 | 0 | 0.000% | 88 | 100.000% |
| disapproval | 195 | 0 | 0.000% | 195 | 100.000% |
| disgust | 76 | 0 | 0.000% | 76 | 100.000% |
| embarrassment | 23 | 0 | 0.000% | 23 | 100.000% |
| excitement | 57 | 0 | 0.000% | 57 | 100.000% |
| fear | 65 | 0 | 0.000% | 65 | 100.000% |
| gratitude | 260 | 0 | 0.000% | 260 | 100.000% |
| grief | 2 | 0 | 0.000% | 2 | 100.000% |
| joy | 93 | 0 | 0.000% | 93 | 100.000% |
| love | 160 | 0 | 0.000% | 160 | 100.000% |
| nervousness | 12 | 0 | 0.000% | 12 | 100.000% |
| optimism | 107 | 0 | 0.000% | 107 | 100.000% |
| pride | 7 | 0 | 0.000% | 7 | 100.000% |
| realization | 89 | 0 | 0.000% | 89 | 100.000% |
| relief | 7 | 0 | 0.000% | 7 | 100.000% |
| remorse | 44 | 0 | 0.000% | 44 | 100.000% |
| sadness | 102 | 0 | 0.000% | 102 | 100.000% |
| surprise | 87 | 0 | 0.000% | 87 | 100.000% |
| neutral | 1,606 | 0 | 0.000% | 1,606 | 100.000% |
| **TOTAL** | 4,590 | 0 | 0.000% | 4,590 | 100.000% |

This table uses only the original test split; each percentage uses its own emotion-row total. Differences from other splits are descriptive; rare emotion percentages can move substantially with a few predictions.

### Main wrong-label confusions (all splits)

| True emotion | Wrong predicted label | Count | % of that true emotion |
| --- | --- | --- | --- |
| neutral | disapproval | 2 | 0.012% |
| curiosity | embarrassment | 1 | 0.058% |
| neutral | anger | 1 | 0.006% |

These are the largest off-diagonal counts among valid labels; INVALID responses are excluded here and reported separately above. Counts favor frequent true classes and should be read with their within-emotion percentages.

### Examples of invalid raw responses

| Raw response (escaped, up to 180 characters) | Count among invalid outputs |
| --- | --- |
| `\n\nText: I'm not sure if` | 3,926 |
| `\n\nText: I'm not sure what` | 1,045 |
| `\n\nText: I’m not sure if` | 884 |

These are the three most frequent exact invalid responses, with newlines escaped. They diagnose response formatting or use of labels outside the allowed vocabulary; they do not establish the cause.

### Inspect individual correct and incorrect examples

[Sample-level outcome table](../outputs/rq2_classification/llama2/full/analyses/sample_outcomes.csv) contains every example with `correct`, `outcome`, true/predicted emotion, original text and raw response. Filter `correct=True` for correct data and `correct=False` for incorrect data; within the latter, `incorrect_label` and `invalid_output` distinguish errors. [Saved sample indexes](../outputs/rq2_classification/llama2/full/analyses/sample_indexes.json) groups sample IDs by split, emotion and Boolean outcome. [Per-label precision/recall/F1](../outputs/rq2_classification/llama2/full/analyses/label_metrics.csv) and [Complete confusion table](../outputs/rq2_classification/llama2/full/analyses/confusion.csv) retain the full detail. No redundant copies of these canonical files were created.

## Apertus

Checkpoint: `swiss-ai/Apertus-8B-Instruct-2509`. Native chat template; instruction checkpoint.

| Split | Total | Correct n | Incorrect n | Correct % | Incorrect % | Invalid n | Invalid % | Macro F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 45,446 | 10,309 | 35,137 | 22.684% | 77.316% | 1,016 | 2.236% | 21.742% |
| train | 36,308 | 8,226 | 28,082 | 22.656% | 77.344% | 823 | 2.267% | 21.844% |
| validation | 4,548 | 1,047 | 3,501 | 23.021% | 76.979% | 88 | 1.935% | 21.634% |
| test | 4,590 | 1,036 | 3,554 | 22.571% | 77.429% | 105 | 2.288% | 20.641% |

Highest observed all-data recall: **anger 57.866%** (732/1,265); lowest: **remorse 2.975%** (13/437). Neutral recall is **12.228%** (1,959/16,021). These are descriptive extremes selected after inspection; small supports require caution.

### Correct and incorrect by emotion — all splits combined

| Emotion | Total | Correct n | Correct % | Incorrect n | Incorrect % |
| --- | --- | --- | --- | --- | --- |
| admiration | 3,384 | 206 | 6.087% | 3,178 | 93.913% |
| amusement | 2,046 | 1,168 | 57.087% | 878 | 42.913% |
| anger | 1,265 | 732 | 57.866% | 533 | 42.134% |
| annoyance | 1,809 | 177 | 9.784% | 1,632 | 90.216% |
| approval | 2,367 | 124 | 5.239% | 2,243 | 94.761% |
| caring | 831 | 48 | 5.776% | 783 | 94.224% |
| confusion | 1,057 | 212 | 20.057% | 845 | 79.943% |
| curiosity | 1,729 | 427 | 24.696% | 1,302 | 75.304% |
| desire | 497 | 34 | 6.841% | 463 | 93.159% |
| disappointment | 888 | 413 | 46.509% | 475 | 53.491% |
| disapproval | 1,809 | 996 | 55.058% | 813 | 44.942% |
| disgust | 635 | 204 | 32.126% | 431 | 67.874% |
| embarrassment | 246 | 99 | 40.244% | 147 | 59.756% |
| excitement | 619 | 139 | 22.456% | 480 | 77.544% |
| fear | 553 | 262 | 47.378% | 291 | 52.622% |
| gratitude | 2,378 | 941 | 39.571% | 1,437 | 60.429% |
| grief | 47 | 12 | 25.532% | 35 | 74.468% |
| joy | 1,052 | 549 | 52.186% | 503 | 47.814% |
| love | 1,760 | 406 | 23.068% | 1,354 | 76.932% |
| nervousness | 105 | 19 | 18.095% | 86 | 81.905% |
| optimism | 1,087 | 160 | 14.719% | 927 | 85.281% |
| pride | 67 | 36 | 53.731% | 31 | 46.269% |
| realization | 749 | 55 | 7.343% | 694 | 92.657% |
| relief | 103 | 35 | 33.981% | 68 | 66.019% |
| remorse | 437 | 13 | 2.975% | 424 | 97.025% |
| sadness | 1,003 | 489 | 48.754% | 514 | 51.246% |
| surprise | 902 | 394 | 43.681% | 508 | 56.319% |
| neutral | 16,021 | 1,959 | 12.228% | 14,062 | 87.772% |
| **TOTAL** | 45,446 | 10,309 | 22.684% | 35,137 | 77.316% |

This table uses only the combined train, validation and test populations; each percentage uses its own emotion-row total. The TOTAL row matches the overall comparison above.

### Correct and incorrect by emotion — train

| Emotion | Total | Correct n | Correct % | Incorrect n | Incorrect % |
| --- | --- | --- | --- | --- | --- |
| admiration | 2,710 | 164 | 6.052% | 2,546 | 93.948% |
| amusement | 1,652 | 936 | 56.659% | 716 | 43.341% |
| anger | 1,025 | 601 | 58.634% | 424 | 41.366% |
| annoyance | 1,451 | 143 | 9.855% | 1,308 | 90.145% |
| approval | 1,873 | 96 | 5.125% | 1,777 | 94.875% |
| caring | 649 | 35 | 5.393% | 614 | 94.607% |
| confusion | 858 | 169 | 19.697% | 689 | 80.303% |
| curiosity | 1,389 | 337 | 24.262% | 1,052 | 75.738% |
| desire | 389 | 31 | 7.969% | 358 | 92.031% |
| disappointment | 709 | 334 | 47.109% | 375 | 52.891% |
| disapproval | 1,402 | 768 | 54.779% | 634 | 45.221% |
| disgust | 498 | 155 | 31.124% | 343 | 68.876% |
| embarrassment | 203 | 84 | 41.379% | 119 | 58.621% |
| excitement | 510 | 115 | 22.549% | 395 | 77.451% |
| fear | 430 | 197 | 45.814% | 233 | 54.186% |
| gratitude | 1,857 | 738 | 39.742% | 1,119 | 60.258% |
| grief | 39 | 9 | 23.077% | 30 | 76.923% |
| joy | 853 | 444 | 52.052% | 409 | 47.948% |
| love | 1,427 | 338 | 23.686% | 1,089 | 76.314% |
| nervousness | 85 | 16 | 18.824% | 69 | 81.176% |
| optimism | 861 | 127 | 14.750% | 734 | 85.250% |
| pride | 51 | 27 | 52.941% | 24 | 47.059% |
| realization | 586 | 44 | 7.509% | 542 | 92.491% |
| relief | 88 | 33 | 37.500% | 55 | 62.500% |
| remorse | 353 | 12 | 3.399% | 341 | 96.601% |
| sadness | 817 | 387 | 47.368% | 430 | 52.632% |
| surprise | 720 | 326 | 45.278% | 394 | 54.722% |
| neutral | 12,823 | 1,560 | 12.166% | 11,263 | 87.834% |
| **TOTAL** | 36,308 | 8,226 | 22.656% | 28,082 | 77.344% |

This table uses only the original train split; each percentage uses its own emotion-row total. Differences from other splits are descriptive; rare emotion percentages can move substantially with a few predictions.

### Correct and incorrect by emotion — validation

| Emotion | Total | Correct n | Correct % | Incorrect n | Incorrect % |
| --- | --- | --- | --- | --- | --- |
| admiration | 326 | 19 | 5.828% | 307 | 94.172% |
| amusement | 208 | 123 | 59.135% | 85 | 40.865% |
| anger | 109 | 65 | 59.633% | 44 | 40.367% |
| annoyance | 164 | 15 | 9.146% | 149 | 90.854% |
| approval | 258 | 16 | 6.202% | 242 | 93.798% |
| caring | 96 | 9 | 9.375% | 87 | 90.625% |
| confusion | 102 | 21 | 20.588% | 81 | 79.412% |
| curiosity | 164 | 41 | 25.000% | 123 | 75.000% |
| desire | 52 | 2 | 3.846% | 50 | 96.154% |
| disappointment | 91 | 45 | 49.451% | 46 | 50.549% |
| disapproval | 212 | 118 | 55.660% | 94 | 44.340% |
| disgust | 61 | 25 | 40.984% | 36 | 59.016% |
| embarrassment | 20 | 8 | 40.000% | 12 | 60.000% |
| excitement | 52 | 9 | 17.308% | 43 | 82.692% |
| fear | 58 | 30 | 51.724% | 28 | 48.276% |
| gratitude | 261 | 101 | 38.697% | 160 | 61.303% |
| grief | 6 | 3 | 50.000% | 3 | 50.000% |
| joy | 106 | 52 | 49.057% | 54 | 50.943% |
| love | 173 | 35 | 20.231% | 138 | 79.769% |
| nervousness | 8 | 0 | 0.000% | 8 | 100.000% |
| optimism | 119 | 22 | 18.487% | 97 | 81.513% |
| pride | 9 | 7 | 77.778% | 2 | 22.222% |
| realization | 74 | 6 | 8.108% | 68 | 91.892% |
| relief | 8 | 1 | 12.500% | 7 | 87.500% |
| remorse | 40 | 0 | 0.000% | 40 | 100.000% |
| sadness | 84 | 50 | 59.524% | 34 | 40.476% |
| surprise | 95 | 36 | 37.895% | 59 | 62.105% |
| neutral | 1,592 | 188 | 11.809% | 1,404 | 88.191% |
| **TOTAL** | 4,548 | 1,047 | 23.021% | 3,501 | 76.979% |

This table uses only the original validation split; each percentage uses its own emotion-row total. Differences from other splits are descriptive; rare emotion percentages can move substantially with a few predictions.

### Correct and incorrect by emotion — test

| Emotion | Total | Correct n | Correct % | Incorrect n | Incorrect % |
| --- | --- | --- | --- | --- | --- |
| admiration | 348 | 23 | 6.609% | 325 | 93.391% |
| amusement | 186 | 109 | 58.602% | 77 | 41.398% |
| anger | 131 | 66 | 50.382% | 65 | 49.618% |
| annoyance | 194 | 19 | 9.794% | 175 | 90.206% |
| approval | 236 | 12 | 5.085% | 224 | 94.915% |
| caring | 86 | 4 | 4.651% | 82 | 95.349% |
| confusion | 97 | 22 | 22.680% | 75 | 77.320% |
| curiosity | 176 | 49 | 27.841% | 127 | 72.159% |
| desire | 56 | 1 | 1.786% | 55 | 98.214% |
| disappointment | 88 | 34 | 38.636% | 54 | 61.364% |
| disapproval | 195 | 110 | 56.410% | 85 | 43.590% |
| disgust | 76 | 24 | 31.579% | 52 | 68.421% |
| embarrassment | 23 | 7 | 30.435% | 16 | 69.565% |
| excitement | 57 | 15 | 26.316% | 42 | 73.684% |
| fear | 65 | 35 | 53.846% | 30 | 46.154% |
| gratitude | 260 | 102 | 39.231% | 158 | 60.769% |
| grief | 2 | 0 | 0.000% | 2 | 100.000% |
| joy | 93 | 53 | 56.989% | 40 | 43.011% |
| love | 160 | 33 | 20.625% | 127 | 79.375% |
| nervousness | 12 | 3 | 25.000% | 9 | 75.000% |
| optimism | 107 | 11 | 10.280% | 96 | 89.720% |
| pride | 7 | 2 | 28.571% | 5 | 71.429% |
| realization | 89 | 5 | 5.618% | 84 | 94.382% |
| relief | 7 | 1 | 14.286% | 6 | 85.714% |
| remorse | 44 | 1 | 2.273% | 43 | 97.727% |
| sadness | 102 | 52 | 50.980% | 50 | 49.020% |
| surprise | 87 | 32 | 36.782% | 55 | 63.218% |
| neutral | 1,606 | 211 | 13.138% | 1,395 | 86.862% |
| **TOTAL** | 4,590 | 1,036 | 22.571% | 3,554 | 77.429% |

This table uses only the original test split; each percentage uses its own emotion-row total. Differences from other splits are descriptive; rare emotion percentages can move substantially with a few predictions.

### Main wrong-label confusions (all splits)

| True emotion | Wrong predicted label | Count | % of that true emotion |
| --- | --- | --- | --- |
| neutral | disapproval | 4,169 | 26.022% |
| neutral | disappointment | 2,103 | 13.127% |
| neutral | anger | 1,286 | 8.027% |
| neutral | amusement | 1,236 | 7.715% |
| admiration | joy | 968 | 28.605% |

These are the largest off-diagonal counts among valid labels; INVALID responses are excluded here and reported separately above. Counts favor frequent true classes and should be read with their within-emotion percentages.

### Examples of invalid raw responses

| Raw response (escaped, up to 180 characters) | Count among invalid outputs |
| --- | --- |
| `disbelief` | 83 |
| `apology` | 56 |
| `nostalgia` | 55 |

These are the three most frequent exact invalid responses, with newlines escaped. They diagnose response formatting or use of labels outside the allowed vocabulary; they do not establish the cause.

### Inspect individual correct and incorrect examples

[Sample-level outcome table](../outputs/rq2_classification/apertus/full/analyses/sample_outcomes.csv) contains every example with `correct`, `outcome`, true/predicted emotion, original text and raw response. Filter `correct=True` for correct data and `correct=False` for incorrect data; within the latter, `incorrect_label` and `invalid_output` distinguish errors. [Saved sample indexes](../outputs/rq2_classification/apertus/full/analyses/sample_indexes.json) groups sample IDs by split, emotion and Boolean outcome. [Per-label precision/recall/F1](../outputs/rq2_classification/apertus/full/analyses/label_metrics.csv) and [Complete confusion table](../outputs/rq2_classification/apertus/full/analyses/confusion.csv) retain the full detail. No redundant copies of these canonical files were created.

## OLMo-2

Checkpoint: `allenai/OLMo-2-1124-7B-Instruct`. Native chat template; instruction checkpoint.

| Split | Total | Correct n | Incorrect n | Correct % | Incorrect % | Invalid n | Invalid % | Macro F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 45,446 | 11,926 | 33,520 | 26.242% | 73.758% | 704 | 1.549% | 25.369% |
| train | 36,308 | 9,512 | 26,796 | 26.198% | 73.802% | 578 | 1.592% | 25.354% |
| validation | 4,548 | 1,191 | 3,357 | 26.187% | 73.813% | 48 | 1.055% | 25.863% |
| test | 4,590 | 1,223 | 3,367 | 26.645% | 73.355% | 78 | 1.699% | 24.643% |

Highest observed all-data recall: **amusement 70.919%** (1,451/2,046); lowest: **grief 0.000%** (0/47). Neutral recall is **10.280%** (1,647/16,021). These are descriptive extremes selected after inspection; small supports require caution.

### Correct and incorrect by emotion — all splits combined

| Emotion | Total | Correct n | Correct % | Incorrect n | Incorrect % |
| --- | --- | --- | --- | --- | --- |
| admiration | 3,384 | 263 | 7.772% | 3,121 | 92.228% |
| amusement | 2,046 | 1,451 | 70.919% | 595 | 29.081% |
| anger | 1,265 | 742 | 58.656% | 523 | 41.344% |
| annoyance | 1,809 | 109 | 6.025% | 1,700 | 93.975% |
| approval | 2,367 | 241 | 10.182% | 2,126 | 89.818% |
| caring | 831 | 9 | 1.083% | 822 | 98.917% |
| confusion | 1,057 | 297 | 28.098% | 760 | 71.902% |
| curiosity | 1,729 | 615 | 35.570% | 1,114 | 64.430% |
| desire | 497 | 92 | 18.511% | 405 | 81.489% |
| disappointment | 888 | 214 | 24.099% | 674 | 75.901% |
| disapproval | 1,809 | 1,168 | 64.566% | 641 | 35.434% |
| disgust | 635 | 144 | 22.677% | 491 | 77.323% |
| embarrassment | 246 | 57 | 23.171% | 189 | 76.829% |
| excitement | 619 | 268 | 43.296% | 351 | 56.704% |
| fear | 553 | 253 | 45.750% | 300 | 54.250% |
| gratitude | 2,378 | 1,546 | 65.013% | 832 | 34.987% |
| grief | 47 | 0 | 0.000% | 47 | 100.000% |
| joy | 1,052 | 519 | 49.335% | 533 | 50.665% |
| love | 1,760 | 804 | 45.682% | 956 | 54.318% |
| nervousness | 105 | 19 | 18.095% | 86 | 81.905% |
| optimism | 1,087 | 217 | 19.963% | 870 | 80.037% |
| pride | 67 | 38 | 56.716% | 29 | 43.284% |
| realization | 749 | 38 | 5.073% | 711 | 94.927% |
| relief | 103 | 36 | 34.951% | 67 | 65.049% |
| remorse | 437 | 94 | 21.510% | 343 | 78.490% |
| sadness | 1,003 | 585 | 58.325% | 418 | 41.675% |
| surprise | 902 | 460 | 50.998% | 442 | 49.002% |
| neutral | 16,021 | 1,647 | 10.280% | 14,374 | 89.720% |
| **TOTAL** | 45,446 | 11,926 | 26.242% | 33,520 | 73.758% |

This table uses only the combined train, validation and test populations; each percentage uses its own emotion-row total. The TOTAL row matches the overall comparison above.

### Correct and incorrect by emotion — train

| Emotion | Total | Correct n | Correct % | Incorrect n | Incorrect % |
| --- | --- | --- | --- | --- | --- |
| admiration | 2,710 | 207 | 7.638% | 2,503 | 92.362% |
| amusement | 1,652 | 1,156 | 69.976% | 496 | 30.024% |
| anger | 1,025 | 599 | 58.439% | 426 | 41.561% |
| annoyance | 1,451 | 87 | 5.996% | 1,364 | 94.004% |
| approval | 1,873 | 204 | 10.892% | 1,669 | 89.108% |
| caring | 649 | 7 | 1.079% | 642 | 98.921% |
| confusion | 858 | 243 | 28.322% | 615 | 71.678% |
| curiosity | 1,389 | 489 | 35.205% | 900 | 64.795% |
| desire | 389 | 79 | 20.308% | 310 | 79.692% |
| disappointment | 709 | 172 | 24.260% | 537 | 75.740% |
| disapproval | 1,402 | 906 | 64.622% | 496 | 35.378% |
| disgust | 498 | 103 | 20.683% | 395 | 79.317% |
| embarrassment | 203 | 48 | 23.645% | 155 | 76.355% |
| excitement | 510 | 219 | 42.941% | 291 | 57.059% |
| fear | 430 | 197 | 45.814% | 233 | 54.186% |
| gratitude | 1,857 | 1,220 | 65.697% | 637 | 34.303% |
| grief | 39 | 0 | 0.000% | 39 | 100.000% |
| joy | 853 | 414 | 48.535% | 439 | 51.465% |
| love | 1,427 | 658 | 46.111% | 769 | 53.889% |
| nervousness | 85 | 16 | 18.824% | 69 | 81.176% |
| optimism | 861 | 168 | 19.512% | 693 | 80.488% |
| pride | 51 | 28 | 54.902% | 23 | 45.098% |
| realization | 586 | 29 | 4.949% | 557 | 95.051% |
| relief | 88 | 33 | 37.500% | 55 | 62.500% |
| remorse | 353 | 74 | 20.963% | 279 | 79.037% |
| sadness | 817 | 467 | 57.160% | 350 | 42.840% |
| surprise | 720 | 373 | 51.806% | 347 | 48.194% |
| neutral | 12,823 | 1,316 | 10.263% | 11,507 | 89.737% |
| **TOTAL** | 36,308 | 9,512 | 26.198% | 26,796 | 73.802% |

This table uses only the original train split; each percentage uses its own emotion-row total. Differences from other splits are descriptive; rare emotion percentages can move substantially with a few predictions.

### Correct and incorrect by emotion — validation

| Emotion | Total | Correct n | Correct % | Incorrect n | Incorrect % |
| --- | --- | --- | --- | --- | --- |
| admiration | 326 | 24 | 7.362% | 302 | 92.638% |
| amusement | 208 | 151 | 72.596% | 57 | 27.404% |
| anger | 109 | 72 | 66.055% | 37 | 33.945% |
| annoyance | 164 | 9 | 5.488% | 155 | 94.512% |
| approval | 258 | 21 | 8.140% | 237 | 91.860% |
| caring | 96 | 2 | 2.083% | 94 | 97.917% |
| confusion | 102 | 28 | 27.451% | 74 | 72.549% |
| curiosity | 164 | 61 | 37.195% | 103 | 62.805% |
| desire | 52 | 4 | 7.692% | 48 | 92.308% |
| disappointment | 91 | 23 | 25.275% | 68 | 74.725% |
| disapproval | 212 | 127 | 59.906% | 85 | 40.094% |
| disgust | 61 | 22 | 36.066% | 39 | 63.934% |
| embarrassment | 20 | 8 | 40.000% | 12 | 60.000% |
| excitement | 52 | 21 | 40.385% | 31 | 59.615% |
| fear | 58 | 25 | 43.103% | 33 | 56.897% |
| gratitude | 261 | 158 | 60.536% | 103 | 39.464% |
| grief | 6 | 0 | 0.000% | 6 | 100.000% |
| joy | 106 | 51 | 48.113% | 55 | 51.887% |
| love | 173 | 71 | 41.040% | 102 | 58.960% |
| nervousness | 8 | 1 | 12.500% | 7 | 87.500% |
| optimism | 119 | 24 | 20.168% | 95 | 79.832% |
| pride | 9 | 7 | 77.778% | 2 | 22.222% |
| realization | 74 | 4 | 5.405% | 70 | 94.595% |
| relief | 8 | 1 | 12.500% | 7 | 87.500% |
| remorse | 40 | 11 | 27.500% | 29 | 72.500% |
| sadness | 84 | 55 | 65.476% | 29 | 34.524% |
| surprise | 95 | 45 | 47.368% | 50 | 52.632% |
| neutral | 1,592 | 165 | 10.364% | 1,427 | 89.636% |
| **TOTAL** | 4,548 | 1,191 | 26.187% | 3,357 | 73.813% |

This table uses only the original validation split; each percentage uses its own emotion-row total. Differences from other splits are descriptive; rare emotion percentages can move substantially with a few predictions.

### Correct and incorrect by emotion — test

| Emotion | Total | Correct n | Correct % | Incorrect n | Incorrect % |
| --- | --- | --- | --- | --- | --- |
| admiration | 348 | 32 | 9.195% | 316 | 90.805% |
| amusement | 186 | 144 | 77.419% | 42 | 22.581% |
| anger | 131 | 71 | 54.198% | 60 | 45.802% |
| annoyance | 194 | 13 | 6.701% | 181 | 93.299% |
| approval | 236 | 16 | 6.780% | 220 | 93.220% |
| caring | 86 | 0 | 0.000% | 86 | 100.000% |
| confusion | 97 | 26 | 26.804% | 71 | 73.196% |
| curiosity | 176 | 65 | 36.932% | 111 | 63.068% |
| desire | 56 | 9 | 16.071% | 47 | 83.929% |
| disappointment | 88 | 19 | 21.591% | 69 | 78.409% |
| disapproval | 195 | 135 | 69.231% | 60 | 30.769% |
| disgust | 76 | 19 | 25.000% | 57 | 75.000% |
| embarrassment | 23 | 1 | 4.348% | 22 | 95.652% |
| excitement | 57 | 28 | 49.123% | 29 | 50.877% |
| fear | 65 | 31 | 47.692% | 34 | 52.308% |
| gratitude | 260 | 168 | 64.615% | 92 | 35.385% |
| grief | 2 | 0 | 0.000% | 2 | 100.000% |
| joy | 93 | 54 | 58.065% | 39 | 41.935% |
| love | 160 | 75 | 46.875% | 85 | 53.125% |
| nervousness | 12 | 2 | 16.667% | 10 | 83.333% |
| optimism | 107 | 25 | 23.364% | 82 | 76.636% |
| pride | 7 | 3 | 42.857% | 4 | 57.143% |
| realization | 89 | 5 | 5.618% | 84 | 94.382% |
| relief | 7 | 2 | 28.571% | 5 | 71.429% |
| remorse | 44 | 9 | 20.455% | 35 | 79.545% |
| sadness | 102 | 63 | 61.765% | 39 | 38.235% |
| surprise | 87 | 42 | 48.276% | 45 | 51.724% |
| neutral | 1,606 | 166 | 10.336% | 1,440 | 89.664% |
| **TOTAL** | 4,590 | 1,223 | 26.645% | 3,367 | 73.355% |

This table uses only the original test split; each percentage uses its own emotion-row total. Differences from other splits are descriptive; rare emotion percentages can move substantially with a few predictions.

### Main wrong-label confusions (all splits)

| True emotion | Wrong predicted label | Count | % of that true emotion |
| --- | --- | --- | --- |
| neutral | disapproval | 5,413 | 33.787% |
| neutral | amusement | 1,355 | 8.458% |
| neutral | anger | 1,019 | 6.360% |
| neutral | surprise | 879 | 5.487% |
| annoyance | disapproval | 871 | 48.148% |

These are the largest off-diagonal counts among valid labels; INVALID responses are excluded here and reported separately above. Counts favor frequent true classes and should be read with their within-emotion percentages.

### Examples of invalid raw responses

| Raw response (escaped, up to 180 characters) | Count among invalid outputs |
| --- | --- |
| `nostalgia` | 106 |
| `concern` | 55 |
| `regret` | 49 |

These are the three most frequent exact invalid responses, with newlines escaped. They diagnose response formatting or use of labels outside the allowed vocabulary; they do not establish the cause.

### Inspect individual correct and incorrect examples

[Sample-level outcome table](../outputs/rq2_classification/olmo2/full/analyses/sample_outcomes.csv) contains every example with `correct`, `outcome`, true/predicted emotion, original text and raw response. Filter `correct=True` for correct data and `correct=False` for incorrect data; within the latter, `incorrect_label` and `invalid_output` distinguish errors. [Saved sample indexes](../outputs/rq2_classification/olmo2/full/analyses/sample_indexes.json) groups sample IDs by split, emotion and Boolean outcome. [Per-label precision/recall/F1](../outputs/rq2_classification/olmo2/full/analyses/label_metrics.csv) and [Complete confusion table](../outputs/rq2_classification/olmo2/full/analyses/confusion.csv) retain the full detail. No redundant copies of these canonical files were created.

## Conclusions and limitations

OLMo-2 is the strongest observed classifier under the current strict protocol, with Apertus second, but all four fall below the always-neutral accuracy baseline. The base-model runs are severely limited by format compliance; changing prompt design or output constraints would constitute a new experiment and must use a fresh run directory. The existing first-word diagnostic shows that formatting explains part, but not all, of their low strict performance.

The counts and split-level patterns are complete and internally consistent. Generalization remains limited by single-label filtering (multi-label cases were excluded), the imbalanced and partly duplicated population, possible pretraining overlap, a single prompt protocol/seed, a ten-token generation budget, and base-versus-instruction checkpoint differences. There was no paired batch-size-1 validation, so these results establish successful batch-32 execution, not equivalence to single-example generation. No causal claim about architecture or instruction tuning follows.

## Provenance and reproducibility

The report reads saved local results only; it performs no model inference, downloading, publication or modification of canonical run outputs. The user-requested report location is `analyses/`; generated plot data and figures remain under `outputs/`.

- **Gemma-2:** [manifest.json](../outputs/rq2_classification/gemma2/full/manifest.json), [source_audit.json](../outputs/rq2_classification/gemma2/full/source_audit.json), [duplicate_audit.json](../outputs/rq2_classification/gemma2/full/duplicate_audit.json), [analyses/summary.csv](../outputs/rq2_classification/gemma2/full/analyses/summary.csv), [analyses/macro_metrics.csv](../outputs/rq2_classification/gemma2/full/analyses/macro_metrics.csv), [analyses/sample_outcomes.csv](../outputs/rq2_classification/gemma2/full/analyses/sample_outcomes.csv); dataset inputs: `outputs/rq2_classification/gemma2/full/dataset/{train,validation,test}.parquet`.
- **LLaMA-2:** [manifest.json](../outputs/rq2_classification/llama2/full/manifest.json), [source_audit.json](../outputs/rq2_classification/llama2/full/source_audit.json), [duplicate_audit.json](../outputs/rq2_classification/llama2/full/duplicate_audit.json), [analyses/summary.csv](../outputs/rq2_classification/llama2/full/analyses/summary.csv), [analyses/macro_metrics.csv](../outputs/rq2_classification/llama2/full/analyses/macro_metrics.csv), [analyses/sample_outcomes.csv](../outputs/rq2_classification/llama2/full/analyses/sample_outcomes.csv); dataset inputs: `outputs/rq2_classification/llama2/full/dataset/{train,validation,test}.parquet`.
- **Apertus:** [manifest.json](../outputs/rq2_classification/apertus/full/manifest.json), [source_audit.json](../outputs/rq2_classification/apertus/full/source_audit.json), [duplicate_audit.json](../outputs/rq2_classification/apertus/full/duplicate_audit.json), [analyses/summary.csv](../outputs/rq2_classification/apertus/full/analyses/summary.csv), [analyses/macro_metrics.csv](../outputs/rq2_classification/apertus/full/analyses/macro_metrics.csv), [analyses/sample_outcomes.csv](../outputs/rq2_classification/apertus/full/analyses/sample_outcomes.csv); dataset inputs: `outputs/rq2_classification/apertus/full/dataset/{train,validation,test}.parquet`.
- **OLMo-2:** [manifest.json](../outputs/rq2_classification/olmo2/full/manifest.json), [source_audit.json](../outputs/rq2_classification/olmo2/full/source_audit.json), [duplicate_audit.json](../outputs/rq2_classification/olmo2/full/duplicate_audit.json), [analyses/summary.csv](../outputs/rq2_classification/olmo2/full/analyses/summary.csv), [analyses/macro_metrics.csv](../outputs/rq2_classification/olmo2/full/analyses/macro_metrics.csv), [analyses/sample_outcomes.csv](../outputs/rq2_classification/olmo2/full/analyses/sample_outcomes.csv); dataset inputs: `outputs/rq2_classification/olmo2/full/dataset/{train,validation,test}.parquet`.
- **Comparisons:** [Paired cluster-bootstrap results](../outputs/rq2_classification/comparison/paired_model_comparisons.csv).
- **New figures:** [Self-contained plot-ready CSV](../outputs/rq2_classification/comparison/analysis-report/plot_data/model-emotion-results.csv); both figures consume this saved CSV, with checkpoint/dataset revisions, run ID, protocol, seed, batch size, split, emotion, denominators and outcomes.

The combined-population paired comparison can be regenerated with:

```bash
python -m affect_geometry.classification compare \
  --run-dir outputs/rq2_classification/comparison \
  --compare-runs outputs/rq2_classification/{gemma2,llama2,apertus,olmo2}/full
```
