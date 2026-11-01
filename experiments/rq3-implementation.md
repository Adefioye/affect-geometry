# Running correctness-conditioned geometry (RQ3)

The implementation follows `docs/rq3.md` with two independently runnable stages. Cohort sampling
and neutral projection happen only in stage 1. Stage 2 reads the saved sample states and transforms.
The former combined `configs/rq3_geometry.json` has been replaced by the configurations below.

## Stage 1: extract, project, and publish all three models

```sh
python script/extract_rq3_activations.py validate
python script/extract_rq3_activations.py run
# Retry publishing already completed local bundles without loading models:
python script/extract_rq3_activations.py upload
```

Configuration: `configs/rq3/extract_activations.json`. It contains the three model IDs, RQ2 source
paths, extraction precision/batching, one sampling seed, projection storage settings, budgets, and
one Hugging Face destination per model. `--no-upload` prepares local bundles only.

The script validates complete RQ2 predictions, draws balanced cohorts once, and extracts each
selected sample's final non-padding classification-prompt token at every transformer block.
It exports **sample-level matrices**, not just centroids, for correct, incorrect, and all:

- `last_token/layer_000.npy`: original uncentered last-token residual states.
- `neutral_projected/layer_000.npy`: `(x - grand_mean) @ (I - components.T @ components)`.
- `preprocessing/layer_000.npz`: fitted means/components/variances, raw and projected centroids,
  normalized centroids, similarities and cosine distances.
- `samples.csv`: exact row IDs, gold labels, correctness/parser audit, prompt hashes and token
  positions (when supplied by the classification source).
- `status/layer_000.json`: explicit support/failure information.

The neutral transform is fitted independently for each model/cohort/block at threshold 0.5.
The all cohort's raw rows are the exact union of the subsets, but its projected rows use its own
transform; they are **not** concatenated subset projections. Saved projected sample states are
not unit-normalized. Downstream geometry normalizes centroids. Centroids are computed from the
saved projected arrays so their storage precision is accounted for. Unsupported projections keep
the raw states and failure records; no substitute derivative is fabricated.

Local bundles live at `outputs/rq3/<model>/rq3-activations-v1/activation_bundle/`.
Complete source manifests and checksums travel with each bundle. Original extraction caches remain
local and can be reused. Local layer checkpoints make projection/export resumable.

The local `hf_snapshot` analysis exports are insufficient RQ2 inputs: their `sample_outcomes.csv`
lacks compatibility-parser and prompt provenance fields. Supply the original `manifest.json` and
`dataset/{train,validation,test}.parquet` under each configured classification run.

## Stage 2: one analysis job/config per model

```sh
python script/run_rq3_geometry.py run --config configs/rq3/gemma2.json
python script/run_rq3_geometry.py run --config configs/rq3/apertus.json
python script/run_rq3_geometry.py run --config configs/rq3/olmo2.json
```

Each config contains only that model's activation source, analysis settings, budgets, and publishing
settings. The saved cohort draw and seed come from the bundle; stage 2 cannot choose another draw.
It does not load a language model, need the original RQ2 files, fit neutral PCA, or project the samples
again. Primary per-model geometry consumes persisted projected samples and preprocessing checkpoints.
Raw arrays remain part of bundle validation; cross-model recentering uses saved raw centroids.

A valid local bundle is preferred. If it is absent, the script downloads the configured dataset
folder, resolving `revision` to an exact commit before downloading. An existing broken local bundle
fails validation instead of silently changing inputs. The source commit/checksums are recorded.
Set `activation_source.revision` to the upload receipt's commit to pin a particular remote release.

The supplied analysis configs select **full runs**: every block layer (`layers: null`), all three
cohorts, both dimensions, and 2,000 permutations (`smoke: false`). Settings are frozen in the config.
At the user's request, real-data pilot verification is not a prerequisite; `pilot_verified: false`
is retained honestly in provenance. No real-data pilot or full analysis has been executed to verify
runtime. Analysis budgets are seven days and 64 GiB peak CPU RAM per model; these are cooperative
limits, not GPU memory reservations or runtime estimates. Full analysis requires a bundle built
from a full-population RQ2 source.

To finish all descriptive analyses and plots without permutation testing:

```sh
python script/run_rq3_geometry.py run --config configs/rq3/gemma2.json --skip-permutations --no-upload
python script/run_rq3_geometry.py run --config configs/rq3/apertus.json --skip-permutations --no-upload
```

This retains every configured layer and cohort and writes a separate run with `-descriptive`
appended to the configured run ID. Existing full-run outputs and permutation checkpoints remain
available. The descriptive run replays saved activations, computes the non-permutation metrics,
and produces plot-ready CSVs, plots, and the run inventory. Both permutation families are marked
`skipped`; p-values, first significant layers, and significant intervals are unavailable. Peak
layers, abrupt changes, and declines remain descriptive summaries. This mode is not a smoke run.

Validate local or Hub inputs without running metrics or uploading results:

```sh
python script/run_rq3_geometry.py validate --config configs/rq3/gemma2.json
```

Validation checks the model, all artifact checksums, cohort/sample identities, raw/projected
matrix shapes, full-population provenance, and NRC coverage. The complete stage-1 bundle is required,
including samples, preprocessing and manifests; bare activation arrays are insufficient. Unsupported
cohort/layer projection statuses remain explicit. No language model or original RQ2 files are needed.

Replay plots from an existing result directory:

```sh
python script/run_rq3_geometry.py report --run-dir outputs/rq3/gemma2/rq3-analysis-full-v1
```

After all three independent model jobs finish, produce common-support comparisons:

```sh
python script/run_rq3_geometry.py compare --comparison-id rq3-comparison-v1 --compare-runs \
  outputs/rq3/gemma2/rq3-analysis-full-v1 \
  outputs/rq3/apertus/rq3-analysis-full-v1 \
  outputs/rq3/olmo2/rq3-analysis-full-v1
```

## Upload completed analysis without rerunning it

```sh
python script/upload_rq3_analysis.py --config configs/rq3/gemma2.json
python script/upload_rq3_analysis.py --config configs/rq3/apertus.json
python script/upload_rq3_analysis.py --config configs/rq3/olmo2.json
```

This standalone script needs only the completed local analysis folder and its original configuration;
it does not download activations or compute metrics. It verifies model/configuration identity and
all output checksums, then uploads every recorded file: plots, CSVs, plot_data, geometry, validation,
inference, reports, manifests and checkpoints. Unexpected uncheckpointed files cause an error rather
than silent omission. Report regeneration verifies the existing run and refreshes its checksums.
`HF_TOKEN` is loaded from `affect-geometry/.env`; an exported token takes precedence.

## Hugging Face organization

Defaults resolve `{username}` from the authenticated Hugging Face account (`HF_TOKEN` or the
existing Hub login). They create private dataset repositories when absent:

- `{username}/rq3-gemma2`
- `{username}/rq3-apertus`
- `{username}/rq3-olmo2`

Use an explicit namespace in the configs to publish to an organization. Existing repository
visibility is preserved. Analysis and activation configs must name the same repository per model.
Both stages upload by default; use `--no-upload` for offline work. Credentials are never stored
in artifacts. Upload receipts with repository, folder and commit are saved next to each local run.

Each model dataset has this layout:

```text
activations/rq3-activations-v1/
  manifest.json
  cohorts/
    support.csv  counts.csv  unresolved.csv
    correct/    # samples.csv, last_token/, neutral_projected/, preprocessing/, status/
    incorrect/  # same files, separate fitted projection
    all/        # same files, union draw and its own fitted projection
  checkpoints/
analyses/rq3-analysis-full-v1/
  manifest.json
  cohorts/  activations/  preprocessing/  geometry/
  validation/  inference/  metrics/  plot_data/  plots/  reports/  checkpoints/
comparisons/rq3-comparison-v1/
  manifest.json  common_support/  model_support.csv  plot_data/  checkpoints/
```

All checksum-covered files, including `plot_data`, plots and analysis checkpoints, are uploaded.
Only completed, verified stages can be published. An existing remote path with different manifest
provenance is rejected; use a new run ID. Comparisons are uploaded to all three model datasets.
Operational logs stay under `logs/<model>-<run>-extract/` or `logs/<model>-<run>-analysis/`.

## Frozen policies

- Eligibility is strictly more than 100 completed compatibility-correct examples. Sample IDs are
  sorted before the seeded draw; all is the exact post-downsampling union. Zero support and
  unresolved inference remain explicit. Source split, duplicate group and text length are preserved.
- Neutral projection is fitted separately for each cohort/block/model at the existing 0.5 variance
  threshold. The equal-weight non-neutral centroid mean is retained. No auxiliary PCA is applied.
- Only cosine dissimilarities are used. Union kNN ties are broken by emotion ID. Disconnected
  candidates are rejected. Zero-distance edges are preserved. MDS never pads unsupported dimensions.
- Diffusion maps use self-loops of 1, stationary-normalized Markov eigenvectors, integer times,
  and decreasing absolute diffusion contribution. The configured local-scale rank is explicit.
  Alpha=1 is primary; optional alpha=0/0.5 geometry/fidelity diagnostics are saved separately from
  the primary alignment candidate set.
- Hyperparameters use nested leave-one-emotion-out folds. Descriptive full-data coordinates use
  the setting with the best full-set LOEO score. Outer predictions never use that descriptive
  selection. Exact setting-score ties use the lexicographically first setting identifier.
- Full-data Procrustes and pooled held-out R² have separate fields. Valence and VA remain separate.
  Failed folds invalidate a score. Failed layers invalidate equal-layer primary aggregates.
- The same NRC-row permutation applies across every member of a model/target family, including
  all layers. Complete VA rows stay together. Every permutation refits transformations and
  rating-dependent selection. Per-permutation checkpoints save null values, settings and fitted
  outer transformations. Aggregate and layerwise maximum-statistic families are separate.
- Secondary nested layer-and-setting selection is saved separately in
  `metrics/selected_layer_scores.csv` and `validation/selected_layers/`. Each outer fold selects
  both layer and setting using only inner training ratings; ties choose the earliest layer.
  It does not replace `A_M` or receive a permutation p-value. Primary permutation families remain
  the prespecified aggregate and layerwise families. Incomplete layer support is recorded explicitly.
- Best-layer CKA is descriptive, uses native coordinates and the earliest exactly tied layer, and
  retains unsupported matrix cells. No CKA selection occurs from an incomplete layer panel.
- A stable significant interval means at least the configured number of successive evaluated
  layers with family-corrected p ≤ .05. Abrupt transitions use the configured absolute CV-R² change;
  declines use its negative threshold. Smoke runs have no significant-layer declarations.
- Near-tied spectral groups receive principal-angle/projector checks under three symmetric
  perturbations (seed 0, magnitude 1e-7; near-tie tolerance 1e-8). These are numerical diagnostics,
  not evidence of manifold curvature. H0/H1 uses a Z2 clique filtration through triangles;
  surviving intervals have infinite deaths and the filtration bound is saved.
- Cross-model comparisons separately recenter common supported emotion centroids and reapply each
  saved cohort neutral projection. They preserve each model's primary sampled IDs, report emotion
  and sample-set differences, and save common-support held-out scores. They are descriptive.

## Implementation map

| Specification | Implementation |
| --- | --- |
| RQ2 source audit and NRC mapping | `data/classification_source.py` |
| Balanced cohorts | `analysis/cohorts.py` |
| Prompt-state capture and provenance | `activations/classification_prompt.py` |
| Cohort-specific preprocessing | `analysis/geometry_preprocessing.py` |
| Strict MDS and Isomap | `embeddings/geometry.py` |
| Diffusion Maps | `embeddings/diffusion_maps.py` |
| Nested alignment and transformations | `analysis/geometry_validation.py` |
| Permutations and correction families | `analysis/geometry_inference.py` |
| Fidelity, spectra, subspace checks, layer means | `analysis/nonlinearity.py` |
| H0/H1 diagrams and Betti curves | `analysis/topology.py` |
| Aggregates, paired contrasts, CKA, emergence | `analysis/geometry_comparisons.py` |
| Cross-model common support | `analysis/cross_model_geometry.py` |
| Atomic writes and checkpoint checksums | `persistence/correctness_geometry.py` |
| Two-stage orchestration/configuration | `experiments/rq3_stages.py` |
| Portable raw/projected activation bundles | `activations/rq3_bundle.py` |
| Hugging Face upload/download | `persistence/rq3_hub.py` |
| Scientific analysis orchestration | `experiments/correctness_geometry.py` |
| Configuration and cooperative budgets | `experiments/rq3_config.py`, `experiments/rq3_budget.py` |
| CSV-based plots and run inventory | `reporting/correctness_geometry.py` |

## Artifacts and replay

Runs use `outputs/rq3/<model>/<run-id>/` and shared comparisons use
`outputs/rq3/comparisons/<run-id>/`. Operational logs use `logs/<model>-<run-id>/`.
Reusable activations have their own model-specific run directory and are referenced by checksum.
Preprocessing and decomposition stages are atomic and resumable. A completed run verifies all
artifact checksums before skipping work. Changed source/configuration/code provenance requires a
new run ID. Checkpoint corruption invalidates the affected stage; it is not silently accepted.

`geometry/` holds full decompositions and 1D/2D checkpoint references; `validation/` holds outer
folds, fitted transformations, selected settings and per-emotion predictions. `inference/` holds
shared permutation schedules, resumable per-permutation transformations, null scores and family
membership. `metrics/` preserves failures and coverage. `plot_data/` contains independent tidy CSVs;
all essential plots consume these tables. Reports inventory results without claiming scientific
conclusions. Figures distinguish full-data maps from held-out predictions and mark unsupported cells.

## Verification

```sh
python -m pytest tests/rq3 -q
python -m pytest tests -q
```

Tests cover balanced partition identities and the eligibility boundary, neutral retention and
projection invariants, strict MDS rank, curved and disconnected Isomap, diffusion stochasticity and
spectral coordinates, row-order invariance, held-out-target isolation, complete permutation refits,
cache corruption and replay, real prompt-token capture with an injected model, NRC coverage,
known H0/H1 examples, and a synthetic nine-cohort end-to-end run. The synthetic run is explicitly
non-inferential and is not a substitute for an accelerator pilot on the real model panel.
