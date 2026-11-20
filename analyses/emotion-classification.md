# Emotion classification: all-splits emotion breakdown

All four models were evaluated on the same **45,446 single-label GoEmotions examples** across train (36,308), validation (4,548), and test (4,590), covering 28 labels including neutral. These splits are pooled for evaluation; no training or fine-tuning was performed. Results below were recomputed from saved responses in the private Hugging Face `full` runs.

**Strict scoring** requires the entire normalized response to be one allowed emotion label. **Relaxed scoring** uses `choi_weber_first_word_v1`: it accepts an exact label or an allowed label as the first whitespace-separated word after the Choi–Weber case, whitespace, asterisk, and punctuation cleanup. Additional words may follow; a label appearing later in the response does not qualify. Thus `joy because…` is accepted as joy, whereas `The emotion is joy` is invalid. The implementation matches [the original parser](../../COLM_Affective_LLM_Representations/gemma_utils.py) on every saved response.

In each emotion table, **Correct % is per-emotion recall**, using that row's total as the denominator. Incorrect includes wrong labels and invalid responses; TOTAL gives overall accuracy. Percentages are rounded to three decimals.

## Strict versus relaxed summary

| Model | Strict correct n | Strict correct % | Relaxed correct n | Relaxed correct % | Additional correct n | Increase (pp) | Relative increase % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-2 | 3,994 | 8.788% | 7,945 | 17.482% | 3,951 | 8.694 | 98.923% |
| LLaMA-2 | 2 | 0.004% | 1,752 | 3.855% | 1,750 | 3.851 | 87,500.000% |
| Apertus | 10,309 | 22.684% | 10,310 | 22.686% | 1 | 0.002 | 0.010% |
| OLMo-2 | 11,926 | 26.242% | 11,926 | 26.242% | 0 | 0.000 | 0.000% |

**Increase (pp)** = relaxed accuracy − strict accuracy, in percentage points. **Relative increase %** = 100 × (relaxed correct − strict correct) / strict correct, calculated from unrounded counts. LLaMA-2’s large relative increase reflects its baseline of only two correct predictions. No strictly correct sample becomes incorrect under relaxed scoring in these runs.

## Test-only strict versus relaxed accuracy

Each model was evaluated on the same **4,590 single-label examples from the official GoEmotions test split**, including neutral. These results come from the original `full` runs with **batch size 32**. All test examples completed successfully. Accuracy uses all 4,590 examples as its denominator, with invalid responses counted as incorrect.

| Model | Test samples | Strict correct n | Strict accuracy | Relaxed correct n | Relaxed accuracy | Additional correct n | Increase (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-2 | 4,590 | 384 | 8.366% | 787 | 17.146% | 403 | 8.780 |
| OLMo-2 | 4,590 | 1,223 | 26.645% | 1,223 | 26.645% | 0 | 0.000 |
| Apertus | 4,590 | 1,036 | 22.571% | 1,036 | 22.571% | 0 | 0.000 |

Relaxed scoring adds **403 correct responses for Gemma-2**, increasing test accuracy by **8.780 percentage points**. OLMo-2 and Apertus have identical strict and relaxed correct counts on this split.

The totals were checked against the saved split summaries, and rescoring the saved raw responses reproduced both correctness fields. Test sample IDs, text hashes, and gold labels match across all three models. This calculation used existing local snapshots without accessing the dotenv token.

- **Gemma-2** — [Saved test predictions](../../outputs/exp2_classification/gemma2/full/hf_snapshot/cbe0b8e3dd8df548fac272246f70a291a2c74e1a/data/test-00000-of-00001.parquet), [saved split summary](../../outputs/exp2_classification/gemma2/full/hf_snapshot/cbe0b8e3dd8df548fac272246f70a291a2c74e1a/analyses/summary.csv).
- **OLMo-2** — [Saved test predictions](../../outputs/exp2_classification/olmo2/full/hf_snapshot/04c26868cb2b16ab127d9182e8d06264e13dd698/data/test-00000-of-00001.parquet), [saved split summary](../../outputs/exp2_classification/olmo2/full/hf_snapshot/04c26868cb2b16ab127d9182e8d06264e13dd698/analyses/summary.csv).
- **Apertus** — [Saved test predictions](../../outputs/exp2_classification/apertus/full/hf_snapshot/d61972e561c168e9be9fb55b29208999acf2c00f/data/test-00000-of-00001.parquet), [saved split summary](../../outputs/exp2_classification/apertus/full/hf_snapshot/d61972e561c168e9be9fb55b29208999acf2c00f/analyses/summary.csv).

## Gemma-2

Checkpoint: `google/gemma-2-9b`. Plain completion prompt; base checkpoint.

### Strict — all splits combined

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

### Relaxed — all splits combined

| Emotion | Total | Correct n | Correct % | Incorrect n | Incorrect % |
| --- | --- | --- | --- | --- | --- |
| admiration | 3,384 | 766 | 22.636% | 2,618 | 77.364% |
| amusement | 2,046 | 505 | 24.682% | 1,541 | 75.318% |
| anger | 1,265 | 668 | 52.806% | 597 | 47.194% |
| annoyance | 1,809 | 43 | 2.377% | 1,766 | 97.623% |
| approval | 2,367 | 55 | 2.324% | 2,312 | 97.676% |
| caring | 831 | 16 | 1.925% | 815 | 98.075% |
| confusion | 1,057 | 139 | 13.150% | 918 | 86.850% |
| curiosity | 1,729 | 163 | 9.427% | 1,566 | 90.573% |
| desire | 497 | 16 | 3.219% | 481 | 96.781% |
| disappointment | 888 | 223 | 25.113% | 665 | 74.887% |
| disapproval | 1,809 | 101 | 5.583% | 1,708 | 94.417% |
| disgust | 635 | 264 | 41.575% | 371 | 58.425% |
| embarrassment | 246 | 82 | 33.333% | 164 | 66.667% |
| excitement | 619 | 172 | 27.787% | 447 | 72.213% |
| fear | 553 | 261 | 47.197% | 292 | 52.803% |
| gratitude | 2,378 | 524 | 22.035% | 1,854 | 77.965% |
| grief | 47 | 11 | 23.404% | 36 | 76.596% |
| joy | 1,052 | 694 | 65.970% | 358 | 34.030% |
| love | 1,760 | 414 | 23.523% | 1,346 | 76.477% |
| nervousness | 105 | 12 | 11.429% | 93 | 88.571% |
| optimism | 1,087 | 12 | 1.104% | 1,075 | 98.896% |
| pride | 67 | 31 | 46.269% | 36 | 53.731% |
| realization | 749 | 13 | 1.736% | 736 | 98.264% |
| relief | 103 | 30 | 29.126% | 73 | 70.874% |
| remorse | 437 | 72 | 16.476% | 365 | 83.524% |
| sadness | 1,003 | 625 | 62.313% | 378 | 37.687% |
| surprise | 902 | 341 | 37.805% | 561 | 62.195% |
| neutral | 16,021 | 1,692 | 10.561% | 14,329 | 89.439% |
| **TOTAL** | 45,446 | 7,945 | 17.482% | 37,501 | 82.518% |

## LLaMA-2

Checkpoint: `meta-llama/Llama-2-7b-hf`. Plain completion prompt; base checkpoint.

### Strict — all splits combined

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

### Relaxed — all splits combined

| Emotion | Total | Correct n | Correct % | Incorrect n | Incorrect % |
| --- | --- | --- | --- | --- | --- |
| admiration | 3,384 | 36 | 1.064% | 3,348 | 98.936% |
| amusement | 2,046 | 20 | 0.978% | 2,026 | 99.022% |
| anger | 1,265 | 195 | 15.415% | 1,070 | 84.585% |
| annoyance | 1,809 | 19 | 1.050% | 1,790 | 98.950% |
| approval | 2,367 | 5 | 0.211% | 2,362 | 99.789% |
| caring | 831 | 9 | 1.083% | 822 | 98.917% |
| confusion | 1,057 | 61 | 5.771% | 996 | 94.229% |
| curiosity | 1,729 | 27 | 1.562% | 1,702 | 98.438% |
| desire | 497 | 6 | 1.207% | 491 | 98.793% |
| disappointment | 888 | 47 | 5.293% | 841 | 94.707% |
| disapproval | 1,809 | 28 | 1.548% | 1,781 | 98.452% |
| disgust | 635 | 106 | 16.693% | 529 | 83.307% |
| embarrassment | 246 | 59 | 23.984% | 187 | 76.016% |
| excitement | 619 | 41 | 6.624% | 578 | 93.376% |
| fear | 553 | 30 | 5.425% | 523 | 94.575% |
| gratitude | 2,378 | 3 | 0.126% | 2,375 | 99.874% |
| grief | 47 | 0 | 0.000% | 47 | 100.000% |
| joy | 1,052 | 131 | 12.452% | 921 | 87.548% |
| love | 1,760 | 584 | 33.182% | 1,176 | 66.818% |
| nervousness | 105 | 1 | 0.952% | 104 | 99.048% |
| optimism | 1,087 | 11 | 1.012% | 1,076 | 98.988% |
| pride | 67 | 10 | 14.925% | 57 | 85.075% |
| realization | 749 | 6 | 0.801% | 743 | 99.199% |
| relief | 103 | 4 | 3.883% | 99 | 96.117% |
| remorse | 437 | 0 | 0.000% | 437 | 100.000% |
| sadness | 1,003 | 207 | 20.638% | 796 | 79.362% |
| surprise | 902 | 37 | 4.102% | 865 | 95.898% |
| neutral | 16,021 | 69 | 0.431% | 15,952 | 99.569% |
| **TOTAL** | 45,446 | 1,752 | 3.855% | 43,694 | 96.145% |

## Apertus

Checkpoint: `swiss-ai/Apertus-8B-Instruct-2509`. Native chat template; instruction checkpoint.

### Strict — all splits combined

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

### Relaxed — all splits combined

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
| disgust | 635 | 205 | 32.283% | 430 | 67.717% |
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
| **TOTAL** | 45,446 | 10,310 | 22.686% | 35,136 | 77.314% |

## OLMo-2

Checkpoint: `allenai/OLMo-2-1124-7B-Instruct`. Native chat template; instruction checkpoint.

### Strict — all splits combined

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

### Relaxed — all splits combined

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

## Brief analysis

- **Relaxed scoring mainly benefits the base models.** Gemma-2 gains 3,951 correct predictions, rising from 8.788% to 17.482% (+8.694 percentage points). LLaMA-2 gains 1,750, rising from 0.004% to 3.855% (+3.851 points). This shows that leading correct labels followed by extra output account for a substantial part of their strict-scoring failures, although overall accuracy remains low.
- **The model ordering is unchanged.** OLMo-2 remains highest at 26.242%; Apertus reaches 22.686%, gaining only one correct prediction. Their results are nearly insensitive to this relaxation. All four still fall below the always-neutral accuracy baseline of 35.253% (16,021 / 45,446).
- **Emotion-level patterns remain uneven.** With relaxed scoring, Gemma-2’s strongest recall is joy (65.970%), followed by sadness (62.313%) and anger (52.806%); LLaMA-2’s strongest is love (33.182%). Apertus’s only additional correct prediction is disgust. OLMo-2's relaxed results are identical to its strict results, with strongest recall for amusement (70.919%), gratitude (65.013%), and disapproval (64.566%). The tables describe these checkpoints and prompts, with base-versus-instruction differences still confounded. Rare emotions such as grief (47 examples) and pride (67) have small denominators; pooled results are descriptive rather than a held-out generalization estimate.

## Finding correct and incorrect sample IDs

Each CSV below contains **all 45,446 sample IDs**, their original split and true emotion, strict and relaxed predictions/outcomes, and the pinned Hugging Face source revision. Filter **`compat_correct=True`** for relaxed-correct samples or **`compat_correct=False`** for relaxed-incorrect samples. Filter **`newly_correct=True`** for samples rescued by the relaxed metric. `compat_outcome` distinguishes `correct`, `incorrect_label`, and `invalid_output`; an empty parsed label means invalid output.

These IDs match the private Hugging Face `analyses/sample_outcomes.csv` records exactly and can be joined on `sample_id` to retrieve original text and raw responses. The same IDs refer to the same examples across all four models; use model plus sample ID when combining model results. The remote sample-outcome CSVs contain strict results; the linked local files add relaxed results.

| Model | Sample-level results | Relaxed correct n | Relaxed incorrect n |
| --- | --- | --- | --- |
| Gemma-2 | [CSV](../../outputs/exp2_classification/gemma2/full/relaxed_analysis/sample_outcomes.csv) | 7,945 | 37,501 |
| LLaMA-2 | [CSV](../../outputs/exp2_classification/llama2/full/relaxed_analysis/sample_outcomes.csv) | 1,752 | 43,694 |
| Apertus | [CSV](../../outputs/exp2_classification/apertus/full/relaxed_analysis/sample_outcomes.csv) | 10,310 | 35,136 |
| OLMo-2 | [CSV](../../outputs/exp2_classification/olmo2/full/relaxed_analysis/sample_outcomes.csv) | 11,926 | 33,520 |

## Sources and validation

All **181,784 responses** were rescored with both parsers. Results match all **464 saved split/emotion summary rows**, and the four strict tables match the previous report exactly. Sample IDs are unique within each model; IDs, original splits, true emotions, and text match across models. Dataset revision, protocol, seed, and code hash also agree. No inference or remote dataset update was performed.

Sources are the pinned private Hugging Face manifests, `analyses/summary.csv`, and `analyses/sample_outcomes.csv`; local snapshots and file checksums are retained under `outputs/exp2_classification/<model>/full/hf_snapshot/<revision>/`. Parsing uses [the project's scoring code](../../src/affect_geometry/classification/protocol.py). Downloaded private data and derived sample files remain under the ignored `outputs/` directory.

- **Gemma-2:** [Hugging Face snapshot](https://huggingface.co/datasets/kokolamba/rq2-goemotions-single-label-gemma2/tree/cbe0b8e3dd8df548fac272246f70a291a2c74e1a) · [Local validation and checksums](../../outputs/exp2_classification/gemma2/full/relaxed_analysis/provenance.json).
- **LLaMA-2:** [Hugging Face snapshot](https://huggingface.co/datasets/kokolamba/rq2-goemotions-single-label-llama2/tree/d2f2fe89594e750788e1a6ddbbfb8aee196e181a) · [Local validation and checksums](../../outputs/exp2_classification/llama2/full/relaxed_analysis/provenance.json).
- **Apertus:** [Hugging Face snapshot](https://huggingface.co/datasets/kokolamba/rq2-goemotions-single-label-apertus/tree/d61972e561c168e9be9fb55b29208999acf2c00f) · [Local validation and checksums](../../outputs/exp2_classification/apertus/full/relaxed_analysis/provenance.json).
- **OLMo-2:** [Hugging Face snapshot](https://huggingface.co/datasets/kokolamba/rq2-goemotions-single-label-olmo2/tree/04c26868cb2b16ab127d9182e8d06264e13dd698) · [Local validation and checksums](../../outputs/exp2_classification/olmo2/full/relaxed_analysis/provenance.json).

## Emotions with more than 100 correct responses under relaxed scoring

All splits combined, for Gemma-2, Apertus, and OLMo-2. An emotion qualifies when **relaxed correct n > 100**. Neutral is included, consistent with the full tables. Percentages retain each emotion’s original total as the denominator.

### Gemma-2

| Emotion | Total | Correct n | Correct % | Incorrect n | Incorrect % |
| --- | --- | --- | --- | --- | --- |
| admiration | 3,384 | 766 | 22.636% | 2,618 | 77.364% |
| amusement | 2,046 | 505 | 24.682% | 1,541 | 75.318% |
| anger | 1,265 | 668 | 52.806% | 597 | 47.194% |
| confusion | 1,057 | 139 | 13.150% | 918 | 86.850% |
| curiosity | 1,729 | 163 | 9.427% | 1,566 | 90.573% |
| disappointment | 888 | 223 | 25.113% | 665 | 74.887% |
| disapproval | 1,809 | 101 | 5.583% | 1,708 | 94.417% |
| disgust | 635 | 264 | 41.575% | 371 | 58.425% |
| excitement | 619 | 172 | 27.787% | 447 | 72.213% |
| fear | 553 | 261 | 47.197% | 292 | 52.803% |
| gratitude | 2,378 | 524 | 22.035% | 1,854 | 77.965% |
| joy | 1,052 | 694 | 65.970% | 358 | 34.030% |
| love | 1,760 | 414 | 23.523% | 1,346 | 76.477% |
| sadness | 1,003 | 625 | 62.313% | 378 | 37.687% |
| surprise | 902 | 341 | 37.805% | 561 | 62.195% |
| neutral | 16,021 | 1,692 | 10.561% | 14,329 | 89.439% |

### Apertus

| Emotion | Total | Correct n | Correct % | Incorrect n | Incorrect % |
| --- | --- | --- | --- | --- | --- |
| admiration | 3,384 | 206 | 6.087% | 3,178 | 93.913% |
| amusement | 2,046 | 1,168 | 57.087% | 878 | 42.913% |
| anger | 1,265 | 732 | 57.866% | 533 | 42.134% |
| annoyance | 1,809 | 177 | 9.784% | 1,632 | 90.216% |
| approval | 2,367 | 124 | 5.239% | 2,243 | 94.761% |
| confusion | 1,057 | 212 | 20.057% | 845 | 79.943% |
| curiosity | 1,729 | 427 | 24.696% | 1,302 | 75.304% |
| disappointment | 888 | 413 | 46.509% | 475 | 53.491% |
| disapproval | 1,809 | 996 | 55.058% | 813 | 44.942% |
| disgust | 635 | 205 | 32.283% | 430 | 67.717% |
| excitement | 619 | 139 | 22.456% | 480 | 77.544% |
| fear | 553 | 262 | 47.378% | 291 | 52.622% |
| gratitude | 2,378 | 941 | 39.571% | 1,437 | 60.429% |
| joy | 1,052 | 549 | 52.186% | 503 | 47.814% |
| love | 1,760 | 406 | 23.068% | 1,354 | 76.932% |
| optimism | 1,087 | 160 | 14.719% | 927 | 85.281% |
| sadness | 1,003 | 489 | 48.754% | 514 | 51.246% |
| surprise | 902 | 394 | 43.681% | 508 | 56.319% |
| neutral | 16,021 | 1,959 | 12.228% | 14,062 | 87.772% |

### OLMo-2

| Emotion | Total | Correct n | Correct % | Incorrect n | Incorrect % |
| --- | --- | --- | --- | --- | --- |
| admiration | 3,384 | 263 | 7.772% | 3,121 | 92.228% |
| amusement | 2,046 | 1,451 | 70.919% | 595 | 29.081% |
| anger | 1,265 | 742 | 58.656% | 523 | 41.344% |
| annoyance | 1,809 | 109 | 6.025% | 1,700 | 93.975% |
| approval | 2,367 | 241 | 10.182% | 2,126 | 89.818% |
| confusion | 1,057 | 297 | 28.098% | 760 | 71.902% |
| curiosity | 1,729 | 615 | 35.570% | 1,114 | 64.430% |
| disappointment | 888 | 214 | 24.099% | 674 | 75.901% |
| disapproval | 1,809 | 1,168 | 64.566% | 641 | 35.434% |
| disgust | 635 | 144 | 22.677% | 491 | 77.323% |
| excitement | 619 | 268 | 43.296% | 351 | 56.704% |
| fear | 553 | 253 | 45.750% | 300 | 54.250% |
| gratitude | 2,378 | 1,546 | 65.013% | 832 | 34.987% |
| joy | 1,052 | 519 | 49.335% | 533 | 50.665% |
| love | 1,760 | 804 | 45.682% | 956 | 54.318% |
| optimism | 1,087 | 217 | 19.963% | 870 | 80.037% |
| sadness | 1,003 | 585 | 58.325% | 418 | 41.675% |
| surprise | 902 | 460 | 50.998% | 442 | 49.002% |
| neutral | 16,021 | 1,647 | 10.280% | 14,374 | 89.720% |

**Qualifying label counts (out of 28, including neutral):** **Gemma-2: 16**; **Apertus: 19**; **OLMo-2: 19**. This threshold selects labels by the number of correct responses, so it reflects both class size and recall.
