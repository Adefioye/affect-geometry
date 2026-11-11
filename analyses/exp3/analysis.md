# EXP3: correctness-conditioned affective geometry across three models

Analysis of saved results, aligned to the supplied **EXP3 implementation design**
(working draft).
The complete descriptive runs cover all/correct/incorrect cohorts, MDS, Isomap and
Diffusion Maps
(DM), 1D valence and 2D valence–arousal (VA), at every transformer block. Tables below
summarize
every saved result family; detailed layer, setting, emotion and pair observations remain
linked
to their canonical CSVs. Rounding is for presentation only.

**Main finding.** Correct cohorts have higher mean layerwise VA recovery for every
model–method
pair. Incorrect cohorts retain substantial valence and VA recovery. DM improves mean VA
recovery
over MDS in all nine model–cohort cells, but its advantage often accompanies weak
arousal
prediction and does not survive every layer-selection comparison. The geometry therefore
supports
correctness-associated affective organization, with descriptive nonlinear-method
advantages;
it does **not** establish a nonlinear, intrinsically two-dimensional VA manifold.

**Inference status.** These are full-layer, non-smoke **descriptive** runs. Each
requested 2,000
permutations but explicitly completed zero in both target dimensions
(`user_requested_skip`).
No corrected p-values, first significant layers, significant intervals, prompt-sampling
confidence
intervals, or significance claims are available. The unfinished Gemma/Apertus
inferential runs
are not substituted for the completed descriptive runs.

## 1. Scope, provenance and design alignment

The question is whether emotion-centroid representations recover human valence and
arousal,
whether nonlinear embeddings improve on classical MDS, and how recovery differs with
classification
correctness. Native layer indices are zero-based; normalized depth is `(layer + 1) / L`.

| Model | Model identifier | Run ID | Block layers | Emotions K | Valid score cells |
| --- | --- | --- | --- | --- | --- |
| Gemma-2 | google/gemma-2-9b | rq3-analysis-full-v1-descriptive | 0–41 (42) | 16 | 756/756 |
| Apertus | swiss-ai/Apertus-8B-Instruct-2509 | rq3-analysis-full-v1-descriptive | 0–31 (32) | 19 | 576/576 |
| OLMo-2 | allenai/OLMo-2-1124-7B-Instruct | rq3-analysis-full-bs8-v1-descriptive | 0–31 (32) | 19 | 576/576 |

All sources contain 45,446 completed, single-label GoEmotions examples (36,308 train,
4,548 validation, 4,590 test), with zero unresolved outcomes. Cohort membership uses
saved
`compat_correct` from `choi_weber_first_word_v1`, not strict correctness. Seed 42
determines
one balanced draw per model. Representations are final non-padding classification-prompt
states
before answer generation, with separate model/layer/cohort neutral PCA fits at variance
threshold
0.5 and equal-weight non-neutral-centroid centering. Neutral remains in all analyses.
Cosine dissimilarity `1 − cosine_similarity` is the only distance.
Persisted projected sample states use float32 (unnormalized); downstream centroids are
normalized.
Gemma has hidden width 3,584; Apertus and OLMo have width 4,096. The selected OLMo run
uses the
completed `bs8` activation bundle, not an earlier incomplete extraction.

NRC-VAD v2.1 supplies unrescaled `[-1, 1]` valence and arousal, with exact mappings for
all
28 source labels, including neutral. All three runs share reference SHA-256
`5021c35285b626e19ba06974d14d0782c07ea4b7a381ed9415f529455dd035dd`.
The reference provenance records a university-hosted copy after the official download
returned
HTTP 406. Model revisions and source dataset revision are retained in the manifest links
below.

Outer evaluation hides one emotion's complete NRC row; hyperparameter selection uses
inner emotion
folds. Activation points and pooled neutral preprocessing remain transductive. R²
predicts
held-out **ratings**, not unseen prompts or unseen activation points. Procrustes permits
translation,
rotation/reflection and scale, so recovered axes need not be native model axes.

| EXP3 requirement | Policy | Saved-result assessment |
| --- | --- | --- |
| Cohort eligibility/balancing | >100 correct; min(correct, incorrect) per subset; exact union | Verified from saved IDs and counts |
| Coverage | 3 models × 3 cohorts × 3 methods × 2 targets; every block | 1,908/1,908 valid score cells; 54/54 aggregates |
| Selection | Nested LOEO settings; separate nested layer-and-setting scores | Saved for every model/cohort/method/target |
| Graphs | Union kNN; k ∈ {3,5,7,10}; deterministic emotion-ID ties | Disconnected candidates explicitly rejected |
| Diffusion | k ∈ {3,5,7,10}; scale rank ∈ {3,5}; t ∈ {1,2,4}; α=1 | 24 primary settings; no α sensitivity runs |
| Diagnostics | Cosine/native fidelity; spectral dimensions; H0/H1; full CKA | Available; stability scope discussed below |
| Permutation inference | 2,000 complete NRC-row permutations; separate max-stat families | Skipped: 0/2,000 for each target in each model |
| Cross-model matched support | Common eligible emotion intersection | 16 common labels identified; no saved matched-support comparison run |

The supplied draft is an implementation guide, not a frozen preregistration. The
manifests mark
settings frozen and `pilot_verified=false`. This report audits saved outputs; it does
not certify
extraction numerics or re-execute the pipeline. Selected source/plot tables were checked
against
the run checkpoint hashes, and primary means were independently checked from saved layer
scores.

## 2. Cohort support, exclusions and composition

| Model | Source compat-correct | Source accuracy | K | Each subset n | Union n | Neutral/subset | Compat-invalid in incorrect |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-2 | 7,945 | 17.48% | 16 | 6,898 | 13,796 | 1692 | 2,482 (35.98%) |
| Apertus | 10,310 | 22.69% | 19 | 9,241 | 18,482 | 1959 | 286 (3.09%) |
| OLMo-2 | 11,926 | 26.24% | 19 | 9,042 | 18,084 | 1647 | 153 (1.69%) |

Source accuracy is calculated before eligibility/downsampling across all 28 labels. The
geometry
cohorts are deliberately 50:50 correct/incorrect **within every retained emotion**;
their proportions
cannot estimate population accuracy. `all` means the post-downsampling union and
overlaps both subsets.
The union excludes discarded examples. All retained emotions have positive paired
support and NRC
coverage; neutral is well supported in every model. Gemma's large invalid-output
contribution
(35.98% of its incorrect subset) is a material difference in cohort composition.

| Gold emotion | Gemma correct/incorrect → sampled/subset | Apertus correct/incorrect → sampled/subset | OLMo correct/incorrect → sampled/subset |
| --- | --- | --- | --- |
| admiration | 766/2618 → 766 | 206/3178 → 206 | 263/3121 → 263 |
| amusement | 505/1541 → 505 | 1168/878 → 878 | 1451/595 → 595 |
| anger | 668/597 → 597 | 732/533 → 533 | 742/523 → 523 |
| annoyance | 43/1766 → 0 | 177/1632 → 177 | 109/1700 → 109 |
| approval | 55/2312 → 0 | 124/2243 → 124 | 241/2126 → 241 |
| caring | 16/815 → 0 | 48/783 → 0 | 9/822 → 0 |
| confusion | 139/918 → 139 | 212/845 → 212 | 297/760 → 297 |
| curiosity | 163/1566 → 163 | 427/1302 → 427 | 615/1114 → 615 |
| desire | 16/481 → 0 | 34/463 → 0 | 92/405 → 0 |
| disappointment | 223/665 → 223 | 413/475 → 413 | 214/674 → 214 |
| disapproval | 101/1708 → 101 | 996/813 → 813 | 1168/641 → 641 |
| disgust | 264/371 → 264 | 205/430 → 205 | 144/491 → 144 |
| embarrassment | 82/164 → 0 | 99/147 → 0 | 57/189 → 0 |
| excitement | 172/447 → 172 | 139/480 → 139 | 268/351 → 268 |
| fear | 261/292 → 261 | 262/291 → 262 | 253/300 → 253 |
| gratitude | 524/1854 → 524 | 941/1437 → 941 | 1546/832 → 832 |
| grief | 11/36 → 0 | 12/35 → 0 | 0/47 → 0 |
| joy | 694/358 → 358 | 549/503 → 503 | 519/533 → 519 |
| love | 414/1346 → 414 | 406/1354 → 406 | 804/956 → 804 |
| nervousness | 12/93 → 0 | 19/86 → 0 | 19/86 → 0 |
| neutral | 1692/14329 → 1692 | 1959/14062 → 1959 | 1647/14374 → 1647 |
| optimism | 12/1075 → 0 | 160/927 → 160 | 217/870 → 217 |
| pride | 31/36 → 0 | 36/31 → 0 | 38/29 → 0 |
| realization | 13/736 → 0 | 55/694 → 0 | 38/711 → 0 |
| relief | 30/73 → 0 | 35/68 → 0 | 36/67 → 0 |
| remorse | 72/365 → 0 | 13/424 → 0 | 94/343 → 0 |
| sadness | 625/378 → 378 | 489/514 → 489 | 585/418 → 418 |
| surprise | 341/561 → 341 | 394/508 → 394 | 460/442 → 442 |

A zero after the arrow denotes exclusion under the >100-correct rule, not a zero
centroid.
Gemma retains 16 emotions; Apertus and OLMo retain those same 16 plus annoyance,
approval and optimism
(K=19). All three exclude caring, desire, embarrassment, grief, nervousness, pride,
realization,
relief and remorse. Gemma disapproval only narrowly qualifies (101 correct); OLMo
annoyance has
109 correct, so those centroids have comparatively limited sample support. Centroid
alignment weights
emotions equally despite unequal per-emotion sample counts.

The common cross-model emotion intersection is the 16 Gemma labels. No persisted
common-support
re-centering/projection/re-embedding comparison is available. Consequently, the
cross-model numbers
below describe each model's primary support; they do not rank models on a matched
16-label experiment.
Model-specific correctness and prompt templates also differ. Optional shared-input
comparisons
would require their own saved sample identities and balance audits.

| Model | Cohort | Mean text length (characters) | Train | Validation | Test | Compat-correct but strict-incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma-2 | correct | 59.57 | 80.46% | 9.70% | 9.84% | 3488 |
| Gemma-2 | incorrect | 67.69 | 80.15% | 9.68% | 10.16% | 0 |
| Gemma-2 | all | 63.63 | 80.31% | 9.69% | 10.00% | 3488 |
| Apertus | correct | 61.88 | 79.70% | 10.12% | 10.18% | 1 |
| Apertus | incorrect | 68.77 | 80.40% | 9.84% | 9.76% | 0 |
| Apertus | all | 65.33 | 80.05% | 9.98% | 9.97% | 1 |
| OLMo-2 | correct | 61.35 | 80.08% | 9.89% | 10.03% | 0 |
| OLMo-2 | incorrect | 68.67 | 80.28% | 10.32% | 9.40% | 0 |
| OLMo-2 | all | 65.01 | 80.18% | 10.10% | 9.72% | 0 |

Incorrect examples are longer on average: by 8.12 characters in Gemma, 6.89 in Apertus
and 7.32
in OLMo. Split proportions are close but not matched, as the design specifies. Gemma's
relaxed parser
rescues 3,488 correct-cohort examples that strict correctness would misassign; Apertus
rescues one,
OLMo none. Replacing the compatibility parser would materially change Gemma's
experiment.
Correctness contrasts also include cohort-specific fitted neutral transforms and
residual text
composition; they do not identify a causal effect of classification correctness.

## 3. Primary recovery: mean held-out R² across all layers

`A_M` is the equal-weight mean of layerwise LOEO R², preserving negative scores. It is
not a peak,
not an R² of layer-averaged predictions, and not a direct amount of nonlinearity. All 54
aggregates
have complete fixed-layer coverage. `Δ` subtracts MDS within the same
model/cohort/target.

### 3.1 Primary 2D valence–arousal

| Model | Cohort | MDS A | Isomap A | DM A | Δ Isomap−MDS | Δ DM−MDS |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma-2 | correct | 0.281 | 0.281 | 0.352 | -0.001 | 0.071 |
| Gemma-2 | incorrect | 0.096 | 0.112 | 0.272 | 0.016 | 0.175 |
| Gemma-2 | all | 0.099 | 0.124 | 0.235 | 0.025 | 0.135 |
| Apertus | correct | 0.447 | 0.476 | 0.468 | 0.029 | 0.021 |
| Apertus | incorrect | 0.238 | 0.254 | 0.357 | 0.016 | 0.119 |
| Apertus | all | 0.157 | 0.207 | 0.232 | 0.050 | 0.075 |
| OLMo-2 | correct | 0.456 | 0.390 | 0.558 | -0.066 | 0.103 |
| OLMo-2 | incorrect | 0.291 | 0.370 | 0.469 | 0.079 | 0.178 |
| OLMo-2 | all | 0.256 | 0.341 | 0.401 | 0.085 | 0.144 |

DM's mean VA advantage is positive in all nine cells, from +0.021 (Apertus correct) to
+0.178
(OLMo incorrect). Isomap's advantage is mixed: −0.066 for OLMo correct and effectively
zero
(−0.001 rounded) for Gemma correct, despite improvements elsewhere. Thus the evidence
does not
support a uniform benefit from nonlinear embeddings as a class.

Within each model, every method recovers more VA in correct than incorrect examples. The
largest
correctness differences are Gemma MDS/Isomap (~+0.17 to +0.19) and Apertus Isomap
(+0.222).
Incorrect-cohort DM scores remain 0.272, 0.357 and 0.469 for Gemma, Apertus and OLMo
respectively;
classification failure is compatible with retained affective organization. These are
descriptive
contrasts, with no test of method superiority or correctness effects.

![VA recovery by normalized depth, with models as rows and correct/incorrect/all as columns. Colors identify methods; solid lines are held-out-rating R² and dashed lines full-data R². The curves show localized peaks and weaker early/late layers; peak scores overstate depth-wide recovery. Full-data and held-out curves can diverge substantially.](plots/va-layerwise-recovery.png)

VA recovery by normalized depth, with models as rows and correct/incorrect/all as columns. Colors identify methods; solid lines are held-out-rating R² and dashed lines full-data R². The curves show localized peaks and weaker early/late layers; peak scores overstate depth-wide recovery. Full-data and held-out curves can diverge substantially. Source: [plot-ready CSV](../../outputs/exp3/analyses/three-models/plot_data/va-layerwise-recovery.csv).

### 3.2 Supporting 1D valence

Valence and VA predict different targets; their R² values are not interchangeable.

| Model | Cohort | MDS A | Isomap A | DM A | Δ Isomap−MDS | Δ DM−MDS |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma-2 | correct | 0.518 | 0.576 | 0.583 | 0.059 | 0.065 |
| Gemma-2 | incorrect | 0.472 | 0.510 | 0.634 | 0.037 | 0.161 |
| Gemma-2 | all | 0.473 | 0.415 | 0.475 | -0.059 | 0.001 |
| Apertus | correct | 0.573 | 0.651 | 0.683 | 0.079 | 0.111 |
| Apertus | incorrect | 0.466 | 0.450 | 0.490 | -0.015 | 0.025 |
| Apertus | all | 0.305 | 0.334 | 0.335 | 0.029 | 0.030 |
| OLMo-2 | correct | 0.631 | 0.704 | 0.732 | 0.073 | 0.101 |
| OLMo-2 | incorrect | 0.622 | 0.638 | 0.687 | 0.015 | 0.065 |
| OLMo-2 | all | 0.480 | 0.516 | 0.491 | 0.036 | 0.010 |

Valence recovery is numerically stronger than joint VA recovery in these runs. DM
improves mean
valence over MDS in every cell, although the Gemma union gain is only +0.001. Isomap
loses against
MDS for Gemma all (−0.059) and Apertus incorrect (−0.015). Gemma provides a clear
exception to a
uniform correctness advantage: incorrect DM valence A=0.634 exceeds correct A=0.583.
OLMo's MDS
valence means are nearly equal across correct/incorrect (0.631/0.622), while its VA
difference is
much larger (0.456/0.291). Correctness is more consistently associated with joint
recovery than
with a single valence coordinate.

![Valence recovery uses the same panels and line conventions as VA. Gemma's correct peaks concentrate around blocks 23–24, while its incorrect DM curve remains useful later. The high peaks coexist with much lower all-layer averages in the tables; the plots do not establish significant emergence.](plots/valence-layerwise-recovery.png)

Valence recovery uses the same panels and line conventions as VA. Gemma's correct peaks concentrate around blocks 23–24, while its incorrect DM curve remains useful later. The high peaks coexist with much lower all-layer averages in the tables; the plots do not establish significant emergence. Source: [plot-ready CSV](../../outputs/exp3/analyses/three-models/plot_data/valence-layerwise-recovery.csv).

### 3.3 Correctness contrasts and the pooled cohort

| Model | Method | 1D C−I | 1D C−all | 1D I−all | VA C−I | VA C−all | VA I−all |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-2 | MDS | 0.045 | 0.045 | -0.001 | 0.185 | 0.182 | -0.003 |
| Gemma-2 | Isomap | 0.067 | 0.162 | 0.095 | 0.169 | 0.157 | -0.012 |
| Gemma-2 | DM | -0.050 | 0.109 | 0.159 | 0.081 | 0.118 | 0.037 |
| Apertus | MDS | 0.107 | 0.268 | 0.161 | 0.209 | 0.290 | 0.081 |
| Apertus | Isomap | 0.201 | 0.317 | 0.116 | 0.222 | 0.269 | 0.047 |
| Apertus | DM | 0.193 | 0.349 | 0.155 | 0.111 | 0.236 | 0.125 |
| OLMo-2 | MDS | 0.009 | 0.151 | 0.142 | 0.165 | 0.199 | 0.034 |
| OLMo-2 | Isomap | 0.066 | 0.188 | 0.121 | 0.020 | 0.049 | 0.028 |
| OLMo-2 | DM | 0.044 | 0.241 | 0.197 | 0.090 | 0.158 | 0.068 |

The pooled cohort is often worse than both subsets, most clearly for Apertus (all six
method/target combinations) and OLMo VA (all three methods). For example, Apertus DM VA
is 0.468
correct, 0.357 incorrect, but only 0.232 all. This is possible because averaging
opposite subset
patterns, cohort-specific centering/neutral projection, and centroid normalization all
change the
union geometry. The union score need not equal the average subset score. Its larger
sample count
does not guarantee stronger emotion-centroid alignment, and it is not an independent
replication.

## 4. Layer selection, axis recovery and emotion-level errors

### 4.1 Peaks versus nested selection

Cells below are **descriptive peak layer : peak R² / nested-selected R²**. The first
score selects
the maximum from layerwise CV scores and is optimistic as a selected-performance
estimate. The
second comes from separate outer folds that select layer and setting using inner ratings
only.
It estimates that adaptive selection procedure; it is not a score at a single globally
selected
layer and does not replace the primary all-layer mean.

**1D valence.**

| Model | Cohort | MDS | Isomap | DM |
| --- | --- | --- | --- | --- |
| Gemma-2 | correct | 23 : 0.927 / 0.927 | 23 : 0.931 / 0.910 | 24 : 0.962 / 0.944 |
| Gemma-2 | incorrect | 24 : 0.791 / 0.771 | 24 : 0.817 / 0.486 | 40 : 0.888 / 0.685 |
| Gemma-2 | all | 24 : 0.881 / 0.849 | 24 : 0.854 / 0.795 | 24 : 0.937 / 0.890 |
| Apertus | correct | 13 : 0.910 / 0.910 | 14 : 0.888 / 0.853 | 13 : 0.920 / 0.905 |
| Apertus | incorrect | 13 : 0.760 / 0.514 | 3 : 0.809 / 0.411 | 7 : 0.830 / 0.266 |
| Apertus | all | 13 : 0.849 / 0.849 | 13 : 0.854 / 0.757 | 13 : 0.858 / 0.858 |
| OLMo-2 | correct | 13 : 0.920 / 0.920 | 8 : 0.904 / 0.798 | 13 : 0.944 / 0.944 |
| OLMo-2 | incorrect | 10 : 0.818 / 0.753 | 10 : 0.841 / 0.768 | 9 : 0.833 / 0.649 |
| OLMo-2 | all | 7 : 0.875 / 0.843 | 11 : 0.890 / 0.834 | 7 : 0.914 / 0.798 |

**2D VA.**

| Model | Cohort | MDS | Isomap | DM |
| --- | --- | --- | --- | --- |
| Gemma-2 | correct | 23 : 0.589 / 0.437 | 21 : 0.736 / 0.607 | 25 : 0.732 / 0.617 |
| Gemma-2 | incorrect | 24 : 0.635 / 0.635 | 24 : 0.567 / 0.566 | 20 : 0.610 / 0.494 |
| Gemma-2 | all | 24 : 0.643 / 0.643 | 24 : 0.642 / 0.547 | 23 : 0.718 / 0.616 |
| Apertus | correct | 13 : 0.666 / 0.615 | 20 : 0.703 / 0.646 | 14 : 0.716 / 0.551 |
| Apertus | incorrect | 7 : 0.566 / 0.566 | 7 : 0.641 / 0.533 | 7 : 0.631 / 0.572 |
| Apertus | all | 7 : 0.628 / 0.628 | 13 : 0.729 / 0.680 | 13 : 0.665 / 0.565 |
| OLMo-2 | correct | 13 : 0.626 / 0.626 | 24 : 0.675 / 0.333 | 9 : 0.676 / 0.439 |
| OLMo-2 | incorrect | 4 : 0.578 / 0.531 | 10 : 0.697 / 0.637 | 10 : 0.646 / 0.559 |
| OLMo-2 | all | 8 : 0.550 / 0.250 | 6 : 0.728 / 0.627 | 10 : 0.688 / 0.598 |

The distinction changes the interpretation. OLMo correct VA DM falls from a descriptive
peak
0.676 to nested-selected 0.439, below nested MDS 0.626. Gemma incorrect VA similarly
favors MDS
after nested selection (0.635 versus DM 0.494). Apertus all VA favors Isomap (nested
0.680), not
DM (0.565), despite DM having the higher all-layer mean. In incorrect valence, selection
is also
unstable: Apertus DM peak 0.830 becomes 0.266. These examples qualify claims based on
attractive
peak maps or a single summary. All nested scores remain descriptive with respect to
permutation
inference; no selection-adjusted p-values were saved.

The observed depth profiles differ: Gemma's strongest correct-cohort recovery clusters
near blocks
21–25 (normalized depth about 0.52–0.62); Apertus correct VA peaks span blocks 13–20
(0.44–0.66); OLMo correct VA peaks differ markedly by method, at blocks 9, 13 and 24.
There is no shared universal peak depth. The union curves show abrupt drops around Gemma
block 25,
Apertus block 14 and OLMo block 14; neutral-transform changes and candidate-setting
changes are
possible contributors, so these transitions cannot be assigned to a model mechanism from
scores alone.

| Model | Cohort | MDS transitions/declines: 1D; VA | Isomap: 1D; VA | DM: 1D; VA |
| --- | --- | --- | --- | --- |
| Gemma-2 | correct | 14/6; 17/9 | 24/11; 21/11 | 26/11; 24/9 |
| Gemma-2 | incorrect | 11/5; 18/8 | 24/10; 27/13 | 18/8; 20/10 |
| Gemma-2 | all | 10/5; 14/8 | 19/8; 20/11 | 17/8; 19/10 |
| Apertus | correct | 7/2; 8/3 | 6/1; 17/7 | 8/3; 14/6 |
| Apertus | incorrect | 10/3; 11/5 | 21/9; 20/9 | 17/7; 18/8 |
| Apertus | all | 13/6; 12/6 | 18/8; 14/7 | 17/8; 18/10 |
| OLMo-2 | correct | 6/3; 8/5 | 7/4; 16/7 | 10/6; 5/1 |
| OLMo-2 | incorrect | 4/3; 14/8 | 13/7; 19/12 | 10/5; 10/5 |
| OLMo-2 | all | 5/3; 17/9 | 12/8; 21/12 | 15/7; 11/7 |

These are saved counts of adjacent-layer changes with |ΔR²|≥0.1, with declines defined
by
ΔR²≤−0.1; every decline is included among transitions. They quantify profile
irregularity under
the configured descriptive rule, not significant emergence. The exact transition layers
remain
in `metrics/emergence.json`. Stable significant intervals require at least three
corrected
significant layers and are unavailable because permutations were skipped.

### 4.2 Joint VA scores can conceal poor arousal recovery

The table reports pooled out-of-fold Pearson correlations **at each method's descriptive
VA-peak
layer**; it does not average singleton-fold correlations or treat these selected
correlations as
unbiased peak estimates.

| Model | Cohort | MDS rV / rA | Isomap rV / rA | DM rV / rA |
| --- | --- | --- | --- | --- |
| Gemma-2 | correct | L23: 0.955 / 0.097 | L21: 0.935 / 0.506 | L25: 0.955 / 0.169 |
| Gemma-2 | incorrect | L24: 0.888 / 0.290 | L24: 0.883 / 0.131 | L20: 0.887 / -0.122 |
| Gemma-2 | all | L24: 0.935 / 0.243 | L24: 0.918 / 0.284 | L23: 0.947 / 0.029 |
| Apertus | correct | L13: 0.947 / 0.276 | L20: 0.903 / 0.546 | L14: 0.949 / -0.286 |
| Apertus | incorrect | L7: 0.818 / 0.482 | L7: 0.831 / 0.600 | L7: 0.838 / 0.404 |
| Apertus | all | L7: 0.836 / 0.597 | L13: 0.923 / 0.529 | L13: 0.920 / -0.098 |
| OLMo-2 | correct | L13: 0.951 / 0.240 | L24: 0.903 / 0.565 | L9: 0.915 / 0.005 |
| OLMo-2 | incorrect | L4: 0.817 / 0.384 | L10: 0.897 / 0.489 | L10: 0.920 / -0.341 |
| OLMo-2 | all | L8: 0.919 / 0.122 | L6: 0.928 / 0.506 | L10: 0.954 / -0.703 |

At DM's VA peak, Gemma all has rV=0.947 but rA=0.029; Apertus correct has rV=0.949 but
rA=−0.286; OLMo all has rV=0.954 but rA=−0.703. OLMo correct DM arousal is essentially
unrelated
(rA=0.005). Isomap generally gives positive arousal correlations at its VA peaks,
reaching 0.506
for Gemma correct, 0.600 for Apertus incorrect and 0.565 for OLMo correct. This axis
distinction
is essential: a DM advantage in joint R² does not establish better arousal recovery.

The target scales help explain the discrepancy. Across retained labels, valence variance
is 0.612
versus arousal 0.165 for Gemma, and 0.600 versus 0.148 for Apertus/OLMo. Thus valence
contributes
about 79–80% of the joint target sum of squares under the same LOEO denominator
weighting. Joint
R² can improve by fitting valence while arousal remains weak. The analysis retains the
specified
unrescaled VA target and reports the two axes separately.

![Held-out valence (solid) and arousal (dashed) correlations by depth, colored by method. Positive valence recovery is broad; arousal is weaker and sometimes negative. The peak-layer correlation table quantifies the discrepancy; neither axis correlation is a separate significance test.](plots/va-axis-correlations.png)

Held-out valence (solid) and arousal (dashed) correlations by depth, colored by method. Positive valence recovery is broad; arousal is weaker and sometimes negative. The peak-layer correlation table quantifies the discrepancy; neither axis correlation is a separate significance test. Source: [plot-ready CSV](../../outputs/exp3/analyses/three-models/plot_data/va-axis-correlations.csv).

### 4.3 Which emotions benefit from the nonlinear methods?

| Model | Cohort | Method vs MDS | Largest error reduction | Largest error increase | Emotions improved | K |
| --- | --- | --- | --- | --- | --- | --- |
| Apertus | all | DM | sadness: -0.508 | optimism: +0.257 | 15 | 19 |
| Apertus | all | Isomap | sadness: -0.497 | optimism: +0.378 | 15 | 19 |
| Apertus | correct | DM | sadness: -0.588 | neutral: +0.787 | 13 | 19 |
| Apertus | correct | Isomap | sadness: -0.380 | neutral: +0.179 | 8 | 19 |
| Apertus | incorrect | DM | fear: -0.275 | surprise: +0.273 | 16 | 19 |
| Apertus | incorrect | Isomap | sadness: -0.282 | surprise: +0.218 | 10 | 19 |
| Gemma-2 | all | DM | sadness: -0.535 | excitement: +0.074 | 13 | 16 |
| Gemma-2 | all | Isomap | sadness: -0.280 | curiosity: +0.100 | 9 | 16 |
| Gemma-2 | correct | DM | love: -0.258 | neutral: +0.128 | 11 | 16 |
| Gemma-2 | correct | Isomap | neutral: -0.210 | curiosity: +0.145 | 7 | 16 |
| Gemma-2 | incorrect | DM | joy: -0.436 | surprise: +0.073 | 14 | 16 |
| Gemma-2 | incorrect | Isomap | confusion: -0.260 | surprise: +0.368 | 7 | 16 |
| OLMo-2 | all | DM | sadness: -0.590 | optimism: +0.160 | 14 | 19 |
| OLMo-2 | all | Isomap | neutral: -0.503 | optimism: +0.186 | 12 | 19 |
| OLMo-2 | correct | DM | sadness: -0.830 | neutral: +0.430 | 14 | 19 |
| OLMo-2 | correct | Isomap | sadness: -0.102 | anger: +0.219 | 5 | 19 |
| OLMo-2 | incorrect | DM | sadness: -0.381 | approval: +0.173 | 16 | 19 |
| OLMo-2 | incorrect | Isomap | sadness: -0.492 | approval: +0.248 | 13 | 19 |

These are mean layerwise differences in squared held-out VA error per emotion; negative
values
favor the nonlinear method. The table exposes heterogeneous benefits that a positive ΔA
can hide.
Extrema are exploratory and selected for description; no emotion-level significance
follows.
Paired errors use the common NRC-covered emotion order within each model and the same
folds.
Sadness often has the largest error reduction, especially for DM (OLMo correct −0.830),
while
neutral can become worse (Apertus correct DM +0.787). DM improves mean VA error for
11–16 emotions
per condition, whereas OLMo correct Isomap improves only 5/19, consistent with its
negative ΔA.

## 5. Distance fidelity and neighborhood preservation

Tables report equal-layer means at each layer's saved **descriptive alignment-selected
setting**.
This describes the embeddings used in the displayed maps, not an independent
fidelity-based
selection experiment. `S` is scale-optimized Stress-1 (lower is better), `NM` is
nonmetric stress,
`ρ` is Shepard Spearman distance correlation, and `T3/C3` are trustworthiness/continuity
at k=3
(higher is better). Psychological-target alignment uses Pearson; Spearman here compares
distances.
Cosine is the common reference for comparing methods. Native references are cosine for
MDS,
graph geodesics for Isomap, and full-spectrum diffusion distances for DM; native scores
must not
be ranked across those different reference objects.

**1D fidelity against common cosine dissimilarities.**

| Model | Cohort | Method | S | NM | ρ | T3 | C3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-2 | correct | MDS | 0.467 | 0.383 | 0.634 | 0.773 | 0.806 |
| Gemma-2 | correct | Isomap | 0.475 | 0.398 | 0.642 | 0.769 | 0.835 |
| Gemma-2 | correct | DM | 0.484 | 0.401 | 0.655 | 0.801 | 0.852 |
| Gemma-2 | incorrect | MDS | 0.437 | 0.335 | 0.722 | 0.779 | 0.831 |
| Gemma-2 | incorrect | Isomap | 0.438 | 0.347 | 0.726 | 0.806 | 0.862 |
| Gemma-2 | incorrect | DM | 0.453 | 0.365 | 0.704 | 0.824 | 0.868 |
| Gemma-2 | all | MDS | 0.458 | 0.364 | 0.667 | 0.778 | 0.817 |
| Gemma-2 | all | Isomap | 0.458 | 0.369 | 0.680 | 0.791 | 0.840 |
| Gemma-2 | all | DM | 0.464 | 0.378 | 0.665 | 0.809 | 0.848 |
| Apertus | correct | MDS | 0.457 | 0.359 | 0.689 | 0.778 | 0.820 |
| Apertus | correct | Isomap | 0.466 | 0.376 | 0.702 | 0.797 | 0.850 |
| Apertus | correct | DM | 0.504 | 0.394 | 0.702 | 0.813 | 0.862 |
| Apertus | incorrect | MDS | 0.463 | 0.384 | 0.633 | 0.751 | 0.797 |
| Apertus | incorrect | Isomap | 0.475 | 0.409 | 0.624 | 0.760 | 0.842 |
| Apertus | incorrect | DM | 0.485 | 0.414 | 0.625 | 0.779 | 0.847 |
| Apertus | all | MDS | 0.477 | 0.403 | 0.564 | 0.724 | 0.747 |
| Apertus | all | Isomap | 0.486 | 0.425 | 0.574 | 0.732 | 0.804 |
| Apertus | all | DM | 0.494 | 0.421 | 0.594 | 0.745 | 0.808 |
| OLMo-2 | correct | MDS | 0.466 | 0.347 | 0.695 | 0.775 | 0.787 |
| OLMo-2 | correct | Isomap | 0.463 | 0.373 | 0.706 | 0.794 | 0.837 |
| OLMo-2 | correct | DM | 0.484 | 0.369 | 0.728 | 0.814 | 0.841 |
| OLMo-2 | incorrect | MDS | 0.451 | 0.314 | 0.768 | 0.785 | 0.822 |
| OLMo-2 | incorrect | Isomap | 0.449 | 0.333 | 0.758 | 0.808 | 0.858 |
| OLMo-2 | incorrect | DM | 0.459 | 0.327 | 0.765 | 0.824 | 0.856 |
| OLMo-2 | all | MDS | 0.475 | 0.369 | 0.599 | 0.731 | 0.747 |
| OLMo-2 | all | Isomap | 0.475 | 0.388 | 0.651 | 0.752 | 0.808 |
| OLMo-2 | all | DM | 0.479 | 0.386 | 0.639 | 0.758 | 0.805 |

**2D fidelity against common cosine dissimilarities.**

| Model | Cohort | Method | S | NM | ρ | T3 | C3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-2 | correct | MDS | 0.285 | 0.233 | 0.775 | 0.875 | 0.894 |
| Gemma-2 | correct | Isomap | 0.308 | 0.255 | 0.755 | 0.880 | 0.919 |
| Gemma-2 | correct | DM | 0.368 | 0.294 | 0.733 | 0.876 | 0.897 |
| Gemma-2 | incorrect | MDS | 0.243 | 0.183 | 0.861 | 0.893 | 0.921 |
| Gemma-2 | incorrect | Isomap | 0.265 | 0.207 | 0.839 | 0.911 | 0.936 |
| Gemma-2 | incorrect | DM | 0.360 | 0.275 | 0.777 | 0.889 | 0.899 |
| Gemma-2 | all | MDS | 0.271 | 0.211 | 0.802 | 0.886 | 0.902 |
| Gemma-2 | all | Isomap | 0.295 | 0.236 | 0.791 | 0.891 | 0.923 |
| Gemma-2 | all | DM | 0.366 | 0.285 | 0.746 | 0.861 | 0.889 |
| Apertus | correct | MDS | 0.287 | 0.226 | 0.808 | 0.875 | 0.886 |
| Apertus | correct | Isomap | 0.317 | 0.261 | 0.781 | 0.894 | 0.921 |
| Apertus | correct | DM | 0.390 | 0.297 | 0.780 | 0.878 | 0.894 |
| Apertus | incorrect | MDS | 0.283 | 0.229 | 0.781 | 0.853 | 0.883 |
| Apertus | incorrect | Isomap | 0.302 | 0.251 | 0.762 | 0.868 | 0.902 |
| Apertus | incorrect | DM | 0.365 | 0.291 | 0.743 | 0.853 | 0.885 |
| Apertus | all | MDS | 0.295 | 0.244 | 0.734 | 0.828 | 0.846 |
| Apertus | all | Isomap | 0.315 | 0.271 | 0.706 | 0.847 | 0.894 |
| Apertus | all | DM | 0.357 | 0.291 | 0.702 | 0.844 | 0.874 |
| OLMo-2 | correct | MDS | 0.299 | 0.230 | 0.780 | 0.860 | 0.865 |
| OLMo-2 | correct | Isomap | 0.307 | 0.246 | 0.793 | 0.869 | 0.898 |
| OLMo-2 | correct | DM | 0.400 | 0.291 | 0.789 | 0.858 | 0.866 |
| OLMo-2 | incorrect | MDS | 0.276 | 0.200 | 0.858 | 0.892 | 0.907 |
| OLMo-2 | incorrect | Isomap | 0.298 | 0.224 | 0.836 | 0.908 | 0.926 |
| OLMo-2 | incorrect | DM | 0.398 | 0.272 | 0.816 | 0.868 | 0.878 |
| OLMo-2 | all | MDS | 0.312 | 0.251 | 0.705 | 0.812 | 0.830 |
| OLMo-2 | all | Isomap | 0.323 | 0.268 | 0.743 | 0.846 | 0.891 |
| OLMo-2 | all | DM | 0.387 | 0.300 | 0.704 | 0.836 | 0.855 |

Two dimensions improve distance reconstruction and neighborhood preservation over one
dimension,
but substantial stress remains. Better psychological prediction does not imply better
preservation
of the original cosine geometry: classical MDS generally preserves that common reference
more
faithfully while DM can improve VA alignment. This separates the human-rating question
from
the geometric reconstruction question. Native DM stress measures truncation relative to
the full
diffusion spectrum, not agreement with original cosine distances.

| Model | Cohort | MDS native S1 | MDS S2 | Isomap native S1 | Isomap S2 | DM native S1 | DM S2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-2 | correct | 0.467 | 0.285 | 0.398 | 0.214 | 0.428 | 0.067 |
| Gemma-2 | incorrect | 0.437 | 0.243 | 0.375 | 0.190 | 0.426 | 0.036 |
| Gemma-2 | all | 0.458 | 0.271 | 0.406 | 0.206 | 0.401 | 0.081 |
| Apertus | correct | 0.457 | 0.287 | 0.352 | 0.199 | 0.478 | 0.080 |
| Apertus | incorrect | 0.463 | 0.283 | 0.396 | 0.227 | 0.464 | 0.087 |
| Apertus | all | 0.477 | 0.295 | 0.417 | 0.239 | 0.459 | 0.138 |
| OLMo-2 | correct | 0.466 | 0.299 | 0.383 | 0.233 | 0.430 | 0.054 |
| OLMo-2 | incorrect | 0.451 | 0.276 | 0.363 | 0.197 | 0.397 | 0.035 |
| OLMo-2 | all | 0.475 | 0.312 | 0.401 | 0.238 | 0.436 | 0.092 |

Selected-setting DM native 2D stress is small (0.035–0.138), although its common-cosine
stress is 0.357–0.400. Diffusion time can strongly downweight higher modes, making the
truncated
output reproduce its own full-spectrum reference well while distorting the original
geometry.
The matched t=1 diagnostics below provide a more comparable construction across cohorts.

| Method | Dim | T1 range | T3 range | T5 range | C1 range | C3 range | C5 range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MDS | 1 | 0.683–0.720 | 0.724–0.785 | 0.758–0.830 | 0.688–0.777 | 0.747–0.831 | 0.774–0.855 |
| MDS | 2 | 0.784–0.843 | 0.812–0.893 | 0.828–0.911 | 0.804–0.889 | 0.830–0.921 | 0.842–0.926 |
| Isomap | 1 | 0.694–0.771 | 0.732–0.808 | 0.761–0.850 | 0.789–0.835 | 0.804–0.862 | 0.814–0.875 |
| Isomap | 2 | 0.812–0.882 | 0.846–0.911 | 0.857–0.919 | 0.876–0.915 | 0.891–0.936 | 0.887–0.931 |
| DM | 1 | 0.702–0.780 | 0.745–0.824 | 0.775–0.853 | 0.772–0.821 | 0.805–0.868 | 0.817–0.880 |
| DM | 2 | 0.809–0.851 | 0.836–0.889 | 0.846–0.895 | 0.824–0.868 | 0.855–0.899 | 0.862–0.902 |

Ranges span the nine separate model/cohort means, not confidence intervals and not a
pooled
model ranking. The complete k=1/3/5 diagnostics and scale factors are retained in
`metrics.csv`.
Fixed-setting trustworthiness and graph-ratio aggregates are also saved: unsupported
sparse
settings retain incomplete-coverage records instead of being averaged over successful
layers only.

| Model | Complete fixed-setting aggregates | Unsupported | Total |
| --- | --- | --- | --- |
| Gemma-2 | 886 | 170 | 1056 |
| Apertus | 886 | 170 | 1056 |
| OLMo-2 | 971 | 85 | 1056 |

These 1,056 rows per model cover fixed-setting trustworthiness and graph-ratio
summaries,
including all neighborhood/reference variants. Missing sparse-graph layers invalidate
170
Gemma/Apertus aggregates each and 85 OLMo aggregates; they are not silently dropped from
a mean.

## 6. Spectral dimensions, neutral removal and diffusion diagnostics

The common geometry table summarizes classical-MDS positive spectral mass of squared
cosine
dissimilarities. PR is the participation ratio; d90/d95 are dimensions reaching 90%/95%
positive
mass. These characterize the **emotion-centroid dissimilarity geometry**, not intrinsic
dimension
of the sample-level activation manifold. Means of integer dimensions may be fractional.

| Model | Cohort | PR | d90 | d95 | 1D mass | 2D mass | Negative / absolute mass | Neutral PCs removed: mean (range) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-2 | correct | 5.704 | 6.667 | 7.905 | 31.6% | 50.9% | 7.0% | 13.5 (2–41) |
| Gemma-2 | incorrect | 4.547 | 5.929 | 7.238 | 38.5% | 60.0% | 6.2% | 10.7 (3–28) |
| Gemma-2 | all | 5.482 | 6.690 | 7.929 | 33.7% | 53.7% | 6.8% | 12.2 (2–35) |
| Apertus | correct | 5.413 | 7.156 | 8.750 | 34.7% | 53.0% | 7.6% | 11.9 (2–17) |
| Apertus | incorrect | 5.532 | 6.844 | 8.219 | 32.1% | 51.4% | 7.5% | 9.8 (2–17) |
| Apertus | all | 6.319 | 7.375 | 8.656 | 30.3% | 48.1% | 7.1% | 10.3 (2–17) |
| OLMo-2 | correct | 5.535 | 7.438 | 8.938 | 34.2% | 51.8% | 6.8% | 18.0 (3–28) |
| OLMo-2 | incorrect | 5.279 | 7.125 | 8.875 | 36.3% | 53.2% | 5.3% | 15.4 (3–25) |
| OLMo-2 | all | 6.504 | 7.938 | 9.375 | 31.7% | 47.0% | 5.9% | 16.6 (3–27) |

Across the nine cells, mean PR is 4.55–6.50, d90 is 5.93–7.94 and d95 is 7.24–9.38.
Only 47.0–60.0% of positive mass is retained in two dimensions. Therefore, 2D VA is a
useful target
projection, not an adequate description of all centroid geometry. Correctness does not
consistently
reduce dimensions: Gemma incorrect PR (4.55) is lower than correct (5.70), whereas
Apertus correct
PR (5.41) is slightly below incorrect (5.53). Nonzero negative eigenvalue mass shows
that the
cosine-derived dissimilarities are not exactly realizable by a Euclidean MDS embedding.

The number of removed neutral PCs changes substantially with layer and cohort,
especially in
Gemma (up to 41 correct-cohort components). Since transforms are independently fitted,
cohort
differences jointly reflect sample composition and the estimated neutral subspace.
Spectral mass
is not sample-activation variance, and no auxiliary PCA or inferred dimensionality elbow
is added.

![Participation ratio, d90 and d95 by depth; the dashed reference is two dimensions. All nine conditions retain structure beyond two dimensions, corroborating the spectral-mass table. The curves are centroid diagnostics and do not estimate a population manifold dimension.](plots/centroid-spectral-dimensions.png)

Participation ratio, d90 and d95 by depth; the dashed reference is two dimensions. All nine conditions retain structure beyond two dimensions, corroborating the spectral-mass table. The curves are centroid diagnostics and do not estimate a population manifold dimension. Source: [plot-ready CSV](../../outputs/exp3/analyses/three-models/plot_data/centroid-spectral-dimensions.csv).

### 6.1 Diffusion geometry at matched settings

To compare the same diffusion construction across cohorts, this table fixes k=10, local
scale
rank=5, t=1, α=1 (`k10-s5-t1-a1`) at every layer. This transparent diagnostic slice was
chosen for
comparison, without ranking settings by NRC recovery; it is not an additional primary
analysis.
PR/d90/d95 and retained energy use the nonconstant diffusion contributions `|λ|^(2t)`.
DM Stress-1 uses **all** nonconstant modes for the reference distance.

| Model | Cohort | Diffusion–cosine ρ | PR | d90 | d95 | Energy 1D | Energy 2D | Native S1 | Native S2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-2 | correct | 0.902 | 3.692 | 5.143 | 6.381 | 43.9% | 66.6% | 0.415 | 0.223 |
| Gemma-2 | incorrect | 0.928 | 3.247 | 4.476 | 6.286 | 48.7% | 72.6% | 0.401 | 0.191 |
| Gemma-2 | all | 0.916 | 3.599 | 4.833 | 6.190 | 45.2% | 68.6% | 0.416 | 0.211 |
| Apertus | correct | 0.916 | 3.682 | 5.625 | 7.406 | 45.4% | 65.7% | 0.422 | 0.231 |
| Apertus | incorrect | 0.911 | 4.088 | 5.688 | 7.500 | 40.6% | 62.7% | 0.438 | 0.240 |
| Apertus | all | 0.896 | 4.540 | 6.031 | 7.562 | 38.1% | 59.3% | 0.449 | 0.255 |
| OLMo-2 | correct | 0.914 | 3.612 | 5.844 | 7.531 | 46.6% | 65.6% | 0.417 | 0.234 |
| OLMo-2 | incorrect | 0.918 | 3.398 | 5.594 | 7.438 | 49.0% | 68.0% | 0.405 | 0.223 |
| OLMo-2 | all | 0.898 | 4.516 | 6.188 | 7.750 | 40.1% | 58.8% | 0.437 | 0.261 |

Diffusion and direct cosine distances remain strongly associated at these settings
(ρ=0.896–0.928).
Two coordinates retain only 58.8–72.6% of diffusion energy; d95 is 6.19–7.75, again
exceeding two.
The smallest diffusion PR is Gemma incorrect (3.25), while pooled Apertus/OLMo are
around 4.5.
Diffusion modifies connectivity while retaining much of the original distance ordering;
improved
VA recovery need not imply an entirely different manifold. Energy dimensions depend on
time,
graph and density normalization; they are not commensurate with MDS positive-mass
dimensions.

![At fixed diffusion settings, columns show diffusion–cosine distance correlation, 2D retained energy, and native 2D stress; lines identify cohorts. High distance correspondence coexists with incomplete 2D energy retention and nonzero stress. These curves compare matched constructions, not rating-selected winners.](plots/matched-diffusion-diagnostics.png)

At fixed diffusion settings, columns show diffusion–cosine distance correlation, 2D retained energy, and native 2D stress; lines identify cohorts. High distance correspondence coexists with incomplete 2D energy retention and nonzero stress. These curves compare matched constructions, not rating-selected winners. Source: [plot-ready CSV](../../outputs/exp3/analyses/three-models/plot_data/matched-diffusion-diagnostics.csv).

## 7. Nonlinearity evidence: graphs, topology, scatter and spectra

### 7.1 Graph dependence and nonmetric cosine dissimilarities

Geodesic/direct ratios are summarized once per graph, not once per embedding method. The
fixed
k=10 graph is connected at every layer in every cohort. A ratio below one is allowed
because
cosine dissimilarity violates the triangle inequality; ratios are not metric path
stretch.

| Model | Cohort | Triangle violations (mean) | Maximum excess | k10 density (mean) | Min algebraic connectivity | Mean geodesic/direct | Pairs ratio<1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-2 | correct | 0.065% | 0.158 | 0.736 | 6.796 | 1.083 | 0.9% |
| Gemma-2 | incorrect | 0.078% | 0.139 | 0.754 | 6.615 | 1.067 | 1.0% |
| Gemma-2 | all | 0.113% | 0.135 | 0.753 | 6.565 | 1.072 | 1.5% |
| Apertus | correct | 0.105% | 0.287 | 0.628 | 3.950 | 1.113 | 1.7% |
| Apertus | incorrect | 0.043% | 0.139 | 0.628 | 4.785 | 1.120 | 0.7% |
| Apertus | all | 0.067% | 0.153 | 0.630 | 4.128 | 1.122 | 1.1% |
| OLMo-2 | correct | 0.071% | 0.213 | 0.633 | 4.036 | 1.116 | 1.1% |
| OLMo-2 | incorrect | 0.026% | 0.166 | 0.644 | 4.170 | 1.125 | 0.4% |
| OLMo-2 | all | 0.077% | 0.148 | 0.641 | 3.989 | 1.125 | 1.2% |

Triangle violations are infrequent in aggregate (roughly 0.03–0.11%) but have nonzero
maxima,
so metric interpretations remain invalid. The relatively dense k=10 graphs shorten paths
and
can suppress differences from the MDS baseline, especially at K=16. Sparse alternatives
are
not always supported: disconnected candidates are explicitly rejected rather than
repaired.

| Model | Cohort | Unsupported Isomap dim-cells | Unsupported DM dim-cells | Reason |
| --- | --- | --- | --- | --- |
| Gemma-2 | correct | 0 | 0 | none |
| Gemma-2 | incorrect | 12 | 72 | disconnected_graph:2 |
| Gemma-2 | all | 2 | 12 | disconnected_graph:2 |
| Apertus | correct | 8 | 48 | disconnected_graph:2 |
| Apertus | incorrect | 0 | 0 | none |
| Apertus | all | 2 | 12 | disconnected_graph:2 |
| OLMo-2 | correct | 0 | 0 | none |
| OLMo-2 | incorrect | 2 | 12 | disconnected_graph:2 |
| OLMo-2 | all | 0 | 0 | none |

These counts include separate 1D/2D records and DM time/scale settings; they are not
counts of
independent failed layers. All failures are two-component disconnected graphs. They do
not
invalidate the 1,908 selected score cells, which have valid alternatives at every layer.
However,
they limit whole-grid comparisons and explain unsupported fixed-setting aggregates. No
failed
candidate is interpreted as a zero alignment score.

### 7.2 Centroid H0/H1 persistence

Persistence uses the same nonmetric cosine clique filtration, bound 2.0, with a
101-point Betti
grid. Zero-lifetime intervals are retained in canonical outputs but excluded when
counting
positive-persistence H1 features below. H0 merger thresholds summarize finite deaths;
its one
essential connected component is not counted as a finite death. H1 lifetimes and counts
are
descriptive and have no sampling-null calibration.

| Model | Cohort | H0 final merge (mean) | Positive H1/layer (mean) | Longest H1/layer (mean) | Longest H1 overall | Max grid Betti-1 |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma-2 | correct | 0.882 | 6.048 | 0.161 | 0.363 | 6 |
| Gemma-2 | incorrect | 0.904 | 3.976 | 0.150 | 0.274 | 5 |
| Gemma-2 | all | 0.919 | 5.667 | 0.148 | 0.333 | 5 |
| Apertus | correct | 0.854 | 7.688 | 0.180 | 0.281 | 6 |
| Apertus | incorrect | 0.879 | 7.250 | 0.187 | 0.304 | 8 |
| Apertus | all | 0.893 | 8.594 | 0.194 | 0.331 | 9 |
| OLMo-2 | correct | 0.871 | 7.625 | 0.182 | 0.345 | 6 |
| OLMo-2 | incorrect | 0.866 | 6.469 | 0.199 | 0.309 | 7 |
| OLMo-2 | all | 0.925 | 7.875 | 0.185 | 0.285 | 7 |

Finite H1 features identify scale-dependent cycles in the centroid dissimilarity graph.
Their
presence does not identify a continuous affective manifold or prove curvature. Lifetimes
are
in cosine-filtration units, depend on K and cohort-specific preprocessing, and lack
resampling
stability estimates. These results complement alignment and spectral diagnostics; the
design
defines no composite nonlinearity score.
Correct cohorts do not uniformly have longer-lived cycles: mean longest H1 lifetime is
0.161 correct versus 0.150 incorrect for Gemma, 0.180 versus 0.187 for Apertus, and
0.182 versus
0.199 for OLMo. Apertus's pooled cohort has the most positive H1 features per layer
(8.59) and
largest grid Betti-1 (9), despite weak union VA recovery. Topology and rating recovery
are distinct.

![Longest positive H1 lifetime at each layer, with models as panels and cohorts as lines. Layer-to-layer changes show that cycles are scale- and representation-dependent; the table reports their counts and lifetimes. No confidence interval or topological significance is implied.](plots/topology-layerwise.png)

Longest positive H1 lifetime at each layer, with models as panels and cohorts as lines. Layer-to-layer changes show that cycles are scale- and representation-dependent; the table reports their counts and lifetimes. No confidence interval or topological significance is implied. Source: [plot-ready CSV](../../outputs/exp3/analyses/three-models/plot_data/topology-layerwise.csv).

### 7.3 RDM contrasts and within-emotion scatter

| Model | Cohort pair | RDM Pearson r (mean) | Layerwise r range | Pairwise distance MAE (mean) |
| --- | --- | --- | --- | --- |
| Gemma-2 | correct / incorrect | 0.728 | 0.580–0.879 | 0.154 |
| Gemma-2 | correct / all | 0.844 | 0.613–0.976 | 0.098 |
| Gemma-2 | incorrect / all | 0.844 | 0.627–0.954 | 0.110 |
| Apertus | correct / incorrect | 0.535 | 0.254–0.846 | 0.179 |
| Apertus | correct / all | 0.693 | 0.336–0.966 | 0.130 |
| Apertus | incorrect / all | 0.727 | 0.477–0.946 | 0.122 |
| OLMo-2 | correct / incorrect | 0.592 | 0.383–0.815 | 0.163 |
| OLMo-2 | correct / all | 0.720 | 0.427–0.961 | 0.117 |
| OLMo-2 | incorrect / all | 0.802 | 0.633–0.939 | 0.109 |

RDM correlations compare the same emotion pairs within each model and are descriptive.
The pooled
RDM is generally closer to each subset than the subsets are to each other, consistent
with a mixed
centroid geometry. Even correlated RDMs can give different VA recovery because alignment
depends
on which emotion relationships shift, not only on global distance agreement.
For correct/incorrect RDMs, mean correlations are 0.728 (Gemma), 0.535 (Apertus), and
0.592
(OLMo), with mean absolute cosine-distance changes of 0.154–0.179. These comparisons
quantify
cohort-associated geometry changes on matched within-model emotion support.

| Model | Median incorrect/correct scatter | Emotion-layer cells ratio>1 | Ratio range |
| --- | --- | --- | --- |
| Gemma-2 | 1.039 | 67.3% | 0.743–1.648 |
| Apertus | 1.018 | 57.6% | 0.559–1.684 |
| OLMo-2 | 1.018 | 56.1% | 0.707–1.611 |

Scatter is mean squared residual norm around the gold-emotion centroid after that
cohort's
projection; ratios pair the same layer and emotion. Ratios avoid comparing raw
activation units
across models, but still include distinct fitted neutral transforms. They do not
establish that
dispersion causes classification errors. Full emotion-level scatter remains in
`metrics.csv`.
Incorrect scatter is only modestly higher at the median (1.8–3.9%), with both increases
and
decreases at individual emotion/layer cells. Large VA contrasts therefore should not be
reduced
to a uniform increase in within-emotion dispersion.

### 7.4 Basic spectral and coordinate checks

Full method/setting spectra, eigengaps, near-tie flags and native coordinates are saved.
Near ties
use the implementation's absolute 1e−8 threshold; conditional subspace checks use three
symmetric
perturbations (seed 0, magnitude 1e−7). Empty stability CSVs mean no tied group was
tested, not
evidence that all leading coordinates are stable. No independent spectral significance
test,
prompt-bootstrap stability estimate, or residual-variance elbow is present. Harmonics
and
horseshoe-like native patterns can be inspected in the original layerwise figures but do
not
independently establish curvature.

| Model | Method | Successful cohort/layer/setting spectra | Spectra with any near tie |
| --- | --- | --- | --- |
| Gemma-2 | MDS | 126 | 0 |
| Gemma-2 | Isomap | 497 | 3 |
| Gemma-2 | DM | 2982 | 0 |
| Apertus | MDS | 96 | 0 |
| Apertus | Isomap | 379 | 0 |
| Apertus | DM | 2274 | 0 |
| OLMo-2 | MDS | 96 | 0 |
| OLMo-2 | Isomap | 383 | 0 |
| OLMo-2 | DM | 2298 | 0 |

| Model | Cohort | Layer | Isomap setting | Tied modes (zero-based) | Min principal cosine | Max projector distance | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-2 | correct | 22 | k3 | 8 / 9 | 1.000000000 | 3.73e-07 | [saved checks](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/geometry/correct/layer_022/isomap/k3/decomposition/subspace_stability.csv) |
| Gemma-2 | correct | 24 | k3 | 8 / 9 / 10 | 1.000000000 | 1.51e-06 | [saved checks](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/geometry/correct/layer_024/isomap/k3/decomposition/subspace_stability.csv) |
| Gemma-2 | correct | 25 | k3 | 8 / 9 | 1.000000000 | 2.02e-07 | [saved checks](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/geometry/correct/layer_025/isomap/k3/decomposition/subspace_stability.csv) |

Only Gemma correct Isomap k=3 has near ties, at layers 22, 24 and 25, in near-zero modes
8–10 rather than the leading two coordinates. All nine saved perturbation checks
succeed;
principal cosines round to one and maximum projector distance is 1.51×10⁻⁶. This
supports numerical
stability of those null subspaces under the specified tiny perturbations, not robustness
to
new samples, alternative graphs, or target-selection changes.

## 8. Cross-method CKA at separately selected best layers

Every method chooses its own highest CV-R² layer for the relevant target, with
earliest-layer
tie-breaking. CKA uses centered **native** coordinates on the same emotion order, not
aligned NRC
coordinates. All 18 required 3×3 matrices are available (162 cells, symmetric,
nondegenerate
diagonals=1). The compact table gives the three unique off-diagonal entries in each
matrix;
figures display every matrix, ordered MDS/Isomap/DM on a shared [0,1] color scale.

| Model | Cohort | Dim | Layers MDS/Iso/DM | MDS–Iso | MDS–DM | Iso–DM |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma-2 | correct | 1 | 23/23/24 | 0.994 | 0.977 | 0.986 |
| Gemma-2 | correct | 2 | 23/21/25 | 0.898 | 0.931 | 0.924 |
| Gemma-2 | incorrect | 1 | 24/24/40 | 0.989 | 0.895 | 0.883 |
| Gemma-2 | incorrect | 2 | 24/24/20 | 0.988 | 0.872 | 0.858 |
| Gemma-2 | all | 1 | 24/24/24 | 0.996 | 0.970 | 0.969 |
| Gemma-2 | all | 2 | 24/24/23 | 0.989 | 0.936 | 0.939 |
| Apertus | correct | 1 | 13/14/13 | 0.891 | 0.968 | 0.945 |
| Apertus | correct | 2 | 13/20/14 | 0.810 | 0.921 | 0.847 |
| Apertus | incorrect | 1 | 13/3/7 | 0.822 | 0.908 | 0.897 |
| Apertus | incorrect | 2 | 7/7/7 | 0.859 | 0.922 | 0.905 |
| Apertus | all | 1 | 13/13/13 | 0.973 | 0.976 | 0.980 |
| Apertus | all | 2 | 7/13/13 | 0.806 | 0.856 | 0.950 |
| OLMo-2 | correct | 1 | 13/8/13 | 0.924 | 0.986 | 0.939 |
| OLMo-2 | correct | 2 | 13/24/9 | 0.811 | 0.900 | 0.829 |
| OLMo-2 | incorrect | 1 | 10/10/9 | 0.960 | 0.965 | 0.965 |
| OLMo-2 | incorrect | 2 | 4/10/10 | 0.851 | 0.793 | 0.952 |
| OLMo-2 | all | 1 | 7/11/7 | 0.922 | 0.973 | 0.952 |
| OLMo-2 | all | 2 | 8/6/10 | 0.893 | 0.921 | 0.939 |

![Full 1D CKA matrices for every model and cohort, with selected layers and K annotated. High similarity indicates shared emotion geometry across methods despite separate layer selection. The table supplies precise pairwise values; CKA is descriptive and does not measure NRC recovery or validate a nonlinear manifold.](plots/best-layer-cka-1d.png)

Full 1D CKA matrices for every model and cohort, with selected layers and K annotated. High similarity indicates shared emotion geometry across methods despite separate layer selection. The table supplies precise pairwise values; CKA is descriptive and does not measure NRC recovery or validate a nonlinear manifold. Source: [plot-ready CSV](../../outputs/exp3/analyses/three-models/plot_data/best-layer-cka-1d.csv).

![Full 2D CKA matrices for every model and cohort, with selected layers and K annotated. High similarity indicates shared emotion geometry across methods despite separate layer selection. The table supplies precise pairwise values; CKA is descriptive and does not measure NRC recovery or validate a nonlinear manifold.](plots/best-layer-cka-2d.png)

Full 2D CKA matrices for every model and cohort, with selected layers and K annotated. High similarity indicates shared emotion geometry across methods despite separate layer selection. The table supplies precise pairwise values; CKA is descriptive and does not measure NRC recovery or validate a nonlinear manifold. Source: [plot-ready CSV](../../outputs/exp3/analyses/three-models/plot_data/best-layer-cka-2d.csv).

One-dimensional CKA ranges from 0.822 to 0.996; it equals squared Pearson coordinate
correlation
and ignores sign. Two-dimensional agreement ranges from 0.793 to 0.989. The lowest 2D
comparison
is OLMo incorrect MDS–DM (0.793), while Isomap–DM there remains 0.952. Gemma's union
MDS–Isomap
agreement is especially high (0.989 in 2D), consistent with their close VA peaks.
Overall, nonlinear
methods preserve much of the baseline organization; CKA does not establish that their
differences
encode arousal or that a native axis is psychological valence.

## 9. Native maps and reproducible visual inspection

The following compact views show the descriptive VA-peak native coordinates for each
method and
cohort. They are chosen for inspection, not performance estimation. Each panel retains
neutral
and draws horizontal/vertical references through its **embedded neutral point**, with
native
units and independent axis limits. There is no NRC overlay and no implied cross-model
coordinate
alignment. Original full-layer maps remain linked in the provenance index.

![Gemma-2 native VA-peak maps, with cohorts as rows and methods as columns. Neutral reference lines make its relative position visible. Emotion separations describe these selected maps; the fidelity tables and axis correlations qualify any apparent VA organization.](plots/gemma2-va-peak-native-maps.png)

Gemma-2 native VA-peak maps, with cohorts as rows and methods as columns. Neutral reference lines make its relative position visible. Emotion separations describe these selected maps; the fidelity tables and axis correlations qualify any apparent VA organization. Source: [plot-ready CSV](../../outputs/exp3/analyses/three-models/plot_data/gemma2-va-peak-native-maps.csv).

![Apertus native VA-peak maps, with cohorts as rows and methods as columns. Neutral reference lines make its relative position visible. Emotion separations describe these selected maps; the fidelity tables and axis correlations qualify any apparent VA organization.](plots/apertus-va-peak-native-maps.png)

Apertus native VA-peak maps, with cohorts as rows and methods as columns. Neutral reference lines make its relative position visible. Emotion separations describe these selected maps; the fidelity tables and axis correlations qualify any apparent VA organization. Source: [plot-ready CSV](../../outputs/exp3/analyses/three-models/plot_data/apertus-va-peak-native-maps.csv).

![OLMo-2 native VA-peak maps, with cohorts as rows and methods as columns. Neutral reference lines make its relative position visible. Emotion separations describe these selected maps; the fidelity tables and axis correlations qualify any apparent VA organization.](plots/olmo2-va-peak-native-maps.png)

OLMo-2 native VA-peak maps, with cohorts as rows and methods as columns. Neutral reference lines make its relative position visible. Emotion separations describe these selected maps; the fidelity tables and axis correlations qualify any apparent VA organization. Source: [plot-ready CSV](../../outputs/exp3/analyses/three-models/plot_data/olmo2-va-peak-native-maps.csv).

Across these maps, positive and negative emotion groups are often separated along a
leading
coordinate, consistent with the strong valence correlations. Neutral occupies different
positions
by cohort, and the second-coordinate ordering varies across methods. In OLMo's union,
the DM map
still separates positive/negative groups while its arousal prediction is anticorrelated
(rA=−0.703);
this illustrates why a visually organized map is insufficient evidence for VA recovery.
Small
DM coordinate scales reflect diffusion time, not less psychological variability.

## 10. Robustness, limits and direct answer to EXP3

**Supported descriptive conclusions.** Correct examples have higher depth-averaged VA
recovery in
all nine model–method comparisons, while incorrect examples still encode recoverable
affective
relationships. DM's VA mean exceeds MDS in every cohort; Isomap's advantage is
conditional on model
and cohort. Recovery is concentrated in particular depth regions rather than sustained
uniformly.
The pooled union often weakens recovery despite its larger sample size. Methods'
selected native
geometries remain strongly related by CKA.

**Qualified interpretation of nonlinearity.** Nonlinear-method alignment gains are
suggestive,
but joint gains are often driven by valence, not arousal, and several gains reverse
under nested
layer selection. Native fidelity, diffusion-energy retention, eigenspectra and H1 cycles
describe
different properties and do not constitute independent proof of a nonlinear VA manifold.
The
positive spectrum consistently needs more than two dimensions for high mass retention.

**What cannot be claimed.** No above-chance significance, method-superiority test,
causal correctness
effect, significant first layer, stable significant emergence interval,
architecture/instruction-tuning
effect, or matched-support cross-model ranking follows from these runs. The design
specifies
separate maximum-statistic families for aggregate A and layerwise R², within
model/target across
methods/cohorts (and layers for the latter); neither family was completed. Correction
would not
extend across the entire model panel. An alignment-against-chance p-value, even if
completed,
would not test descriptive method/cohort differences.

**Material limitations.** There is one balanced seed, no prompt-sampling intervals, and
no saved
replication across cohort draws. K is only 16 or 19; equal emotion weights coexist with
unequal
centroid precision. Correct/incorrect transforms differ, texts are not length matched,
and Gemma's
incorrect subset contains many parser-invalid responses. The representations and neutral
fits are
transductive, targets are word-level NRC norms, and valence has substantially greater
target
variance than arousal. Sparse graph failures restrict the candidate grid. Cosine
nonmetricity,
negative MDS mass, selection optimism and truncated-coordinate stress limit native-map
inference.

**Direct answer.** The saved outputs support correctness-associated affective centroid
geometry and
conditional benefits from nonlinear embeddings. They do not yet establish a nonlinear
two-axis VA
manifold. The most defensible pattern is strong valence organization, less reliable
arousal,
and additional centroid structure beyond two dimensions. Completing the specified
permutation
families and a separately persisted common-support comparison would address the
principal missing
inferential and cross-model evidence; this report does not rerun those experiments.

## 11. Source index and reproducibility

This report reads canonical saved outputs and creates only report-specific plots and
their tidy
source tables. Canonical results are unchanged. The Markdown lives under `analyses/` as
requested;
Embedded report figures live in `plots/`; plot-ready CSVs remain under `outputs/`.

| Model | Run provenance | Integrity | Folds | Inference completion | Original plots |
| --- | --- | --- | --- | --- | --- |
| Gemma-2 | [manifest](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/manifest.json) | [checkpoint hashes](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/checkpoints/run.json) | [fold schedule](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/validation/fold_schedule.json) | [1D status](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/inference/1d/status.json); [VA status](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/inference/2d/status.json) | [all original figures](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/plots) |
| Apertus | [manifest](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/manifest.json) | [checkpoint hashes](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/checkpoints/run.json) | [fold schedule](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/validation/fold_schedule.json) | [1D status](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/inference/1d/status.json); [VA status](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/inference/2d/status.json) | [all original figures](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/plots) |
| OLMo-2 | [manifest](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/manifest.json) | [checkpoint hashes](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/checkpoints/run.json) | [fold schedule](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/validation/fold_schedule.json) | [1D status](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/inference/1d/status.json); [VA status](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/inference/2d/status.json) | [all original figures](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/plots) |

The following index converts each compact result family into the corresponding tables
above.
Counts refer to canonical plot-ready tables, including failed candidate records where
applicable.
Raw pairwise-distance tables are reproduction backing for fidelity, not independent
inferential
observations; all-layer coordinate and Betti tables are similarly summarized rather than
pasted
as hundreds of thousands of Markdown rows.

| Canonical CSV | Gemma rows | Apertus rows | OLMo rows | Report coverage | CSV links |
| --- | --- | --- | --- | --- | --- |
| scores | 756 | 576 | 576 | §§3–4: layer curves, peaks, axis recovery | [Gemma-2](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/plot_data/scores.csv) / [Apertus](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/plot_data/scores.csv) / [OLMo-2](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/plot_data/scores.csv) |
| aggregates | 18 | 18 | 18 | §3: every A and ΔA | [Gemma-2](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/plot_data/aggregates.csv) / [Apertus](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/plot_data/aggregates.csv) / [OLMo-2](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/plot_data/aggregates.csv) |
| selected_layer_scores | 18 | 18 | 18 | §4.1: every nested-selected score | [Gemma-2](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/plot_data/selected_layer_scores.csv) / [Apertus](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/plot_data/selected_layer_scores.csv) / [OLMo-2](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/plot_data/selected_layer_scores.csv) |
| cka | 54 | 54 | 54 | §8: every matrix and unique pair | [Gemma-2](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/plot_data/cka.csv) / [Apertus](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/plot_data/cka.csv) / [OLMo-2](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/plot_data/cka.csv) |
| contrasts | 1,260 | 960 | 960 | §3.3: cohort/method differences; underlying layerwise deltas | [Gemma-2](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/plot_data/contrasts.csv) / [Apertus](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/plot_data/contrasts.csv) / [OLMo-2](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/plot_data/contrasts.csv) |
| paired_errors | 20,160 | 18,240 | 18,240 | §4.3: emotion-specific improvements/regressions | [Gemma-2](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/plot_data/paired_errors.csv) / [Apertus](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/plot_data/paired_errors.csv) / [OLMo-2](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/plot_data/paired_errors.csv) |
| metrics | 176,799 | 135,105 | 136,461 | §§5–7: all 27 metric names, including scatter and graph diagnostics | [Gemma-2](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/plot_data/metrics.csv) / [Apertus](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/plot_data/metrics.csv) / [OLMo-2](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/plot_data/metrics.csv) |
| fidelity_aggregates | 1,056 | 1,056 | 1,056 | §5: fixed-setting trustworthiness/graph-ratio coverage | [Gemma-2](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/plot_data/fidelity_aggregates.csv) / [Apertus](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/plot_data/fidelity_aggregates.csv) / [OLMo-2](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/plot_data/fidelity_aggregates.csv) |
| persistence | 15,246 | 16,512 | 16,512 | §7.2: H0/H1 counts/lifetimes | [Gemma-2](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/plot_data/persistence.csv) / [Apertus](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/plot_data/persistence.csv) / [OLMo-2](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/plot_data/persistence.csv) |
| betti | 25,452 | 19,392 | 19,392 | §7.2: Betti maxima and layerwise cycles | [Gemma-2](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/plot_data/betti.csv) / [Apertus](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/plot_data/betti.csv) / [OLMo-2](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/plot_data/betti.csv) |
| spectra | 57,680 | 52,231 | 52,763 | §§6,7.4: dimensionality, eigenvalues, near ties | [Gemma-2](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/plot_data/spectra.csv) / [Apertus](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/plot_data/spectra.csv) / [OLMo-2](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/plot_data/spectra.csv) |
| coordinates | 127,554 | 115,476 | 116,484 | §9: selected native maps; full-layer archive linked above | [Gemma-2](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/plot_data/coordinates.csv) / [Apertus](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/plot_data/coordinates.csv) / [OLMo-2](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/plot_data/coordinates.csv) |
| graph_ratios | 59,640 | 64,809 | 65,493 | §7.1: graph-level ratio diagnostics | [Gemma-2](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/plot_data/graph_ratios.csv) / [Apertus](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/plot_data/graph_ratios.csv) / [OLMo-2](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/plot_data/graph_ratios.csv) |
| failures | 98 | 70 | 14 | §7.1: every failure type and cohort | [Gemma-2](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/plot_data/failures.csv) / [Apertus](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/plot_data/failures.csv) / [OLMo-2](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/plot_data/failures.csv) |
| rdm_contrasts | 15,120 | 16,416 | 16,416 | §7.3: matched-pair RDM agreement and differences | [Gemma-2](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/plot_data/rdm_contrasts.csv) / [Apertus](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/plot_data/rdm_contrasts.csv) / [OLMo-2](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/plot_data/rdm_contrasts.csv) |
| held_out_predictions | 12,096 | 10,944 | 10,944 | §4: OOF axis/error evidence; predictions retained | [Gemma-2](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/plot_data/held_out_predictions.csv) / [Apertus](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/plot_data/held_out_predictions.csv) / [OLMo-2](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/plot_data/held_out_predictions.csv) |
| composition | 144 | 171 | 171 | §2: invalid contributions and source composition | [Gemma-2](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/plot_data/composition.csv) / [Apertus](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/plot_data/composition.csv) / [OLMo-2](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/plot_data/composition.csv) |
| support | 28 | 28 | 28 | §2: all 28 emotion counts and eligibility | [Gemma-2](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/plot_data/support.csv) / [Apertus](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/plot_data/support.csv) / [OLMo-2](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/plot_data/support.csv) |
| counts | 504 | 504 | 504 | §2: pre/post sampling support and split diagnostics | [Gemma-2](../../outputs/exp3/gemma2/rq3-analysis-full-v1-descriptive/plot_data/counts.csv) / [Apertus](../../outputs/exp3/apertus/rq3-analysis-full-v1-descriptive/plot_data/counts.csv) / [OLMo-2](../../outputs/exp3/olmo2/rq3-analysis-full-bs8-v1-descriptive/plot_data/counts.csv) |

Additional reproduction backing: each run's `metrics/pairwise_distances.csv` and
`plot_data/pairwise_distances.csv` store the distinct emotion-pair reference/embedded
distances;
`validation/` stores OOF predictions, transforms, settings and nested selected layers;
`preprocessing/` stores neutral transforms and centroids; `geometry/` stores
decomposition
metadata, eigenpairs and conditional subspace-stability CSVs. `metrics/emergence.json`
stores
descriptive peaks, configured ±0.1 transitions/declines and unavailable significance
summaries;
`metrics/timing.json` records execution timing. These are provenance/diagnostics, not
additional
independent results. No setting or failed layer was filled by rerunning extraction or
embeddings.

| Report figure | Self-contained plot-ready CSV | PNG |
| --- | --- | --- |
| va-layerwise-recovery | [data](../../outputs/exp3/analyses/three-models/plot_data/va-layerwise-recovery.csv) | [image](plots/va-layerwise-recovery.png) |
| valence-layerwise-recovery | [data](../../outputs/exp3/analyses/three-models/plot_data/valence-layerwise-recovery.csv) | [image](plots/valence-layerwise-recovery.png) |
| va-axis-correlations | [data](../../outputs/exp3/analyses/three-models/plot_data/va-axis-correlations.csv) | [image](plots/va-axis-correlations.png) |
| centroid-spectral-dimensions | [data](../../outputs/exp3/analyses/three-models/plot_data/centroid-spectral-dimensions.csv) | [image](plots/centroid-spectral-dimensions.png) |
| matched-diffusion-diagnostics | [data](../../outputs/exp3/analyses/three-models/plot_data/matched-diffusion-diagnostics.csv) | [image](plots/matched-diffusion-diagnostics.png) |
| topology-layerwise | [data](../../outputs/exp3/analyses/three-models/plot_data/topology-layerwise.csv) | [image](plots/topology-layerwise.png) |
| best-layer-cka-1d | [data](../../outputs/exp3/analyses/three-models/plot_data/best-layer-cka-1d.csv) | [image](plots/best-layer-cka-1d.png) |
| best-layer-cka-2d | [data](../../outputs/exp3/analyses/three-models/plot_data/best-layer-cka-2d.csv) | [image](plots/best-layer-cka-2d.png) |
| gemma2-va-peak-native-maps | [data](../../outputs/exp3/analyses/three-models/plot_data/gemma2-va-peak-native-maps.csv) | [image](plots/gemma2-va-peak-native-maps.png) |
| apertus-va-peak-native-maps | [data](../../outputs/exp3/analyses/three-models/plot_data/apertus-va-peak-native-maps.csv) | [image](plots/apertus-va-peak-native-maps.png) |
| olmo2-va-peak-native-maps | [data](../../outputs/exp3/analyses/three-models/plot_data/olmo2-va-peak-native-maps.csv) | [image](plots/olmo2-va-peak-native-maps.png) |

To rebuild from the same canonical tables, run from the repository root:

```bash
python analyses/exp3/build_report.py
```

The builder uses the project's shared progress/logging utilities. Logs are centralized
under
`logs/three-models-exp3-analysis-report/`. It validates model/run/emotion identities,
cohort sample
partitions, per-emotion balance, prompt-token positions, unique complete score cells,
aggregate
reconstruction and source-table checkpoint hashes before rendering. It does not validate
the
entire multi-gigabyte activation bundle or rerun scientific computations.
