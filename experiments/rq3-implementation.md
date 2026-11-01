# Running correctness-conditioned geometry (RQ3)

The implementation follows `docs/rq3.md`. Primary results use three models, the single saved
balanced correct/incorrect draw and its exact union, all three embedding methods, and 1D valence /
2D VA. Existing RQ1/RQ2 behavior remains available. RQ3's strict decomposition API lives in
`embeddings/geometry.py`; legacy MDS/Isomap entry points retain their older defaults.

## Entry point

Run from the project directory with the project installed (`pip install -e '.[dev]'`):

```sh
python script/run_rq3_geometry.py validate --config configs/rq3_geometry.json
python script/run_rq3_geometry.py extract --config configs/rq3_geometry.json
python script/run_rq3_geometry.py run --config configs/rq3_geometry.json
python script/run_rq3_geometry.py report --run-dir outputs/rq3/gemma2/rq3-pilot-v1
```

`validate` checks all three sources and NRC coverage. `extract` reconstructs and verifies saved
classification prompts, then captures post-block final-nonpadding states without generating
answers. `run` requires checksum-verified prompt states and runs the three cohorts for every
model. `report` rebuilds figures from persisted tables. No command reruns classification.

The supplied JSON is explicitly a **non-inferential smoke/pilot configuration**, with two layers
and two permutations. Its source paths match the RQ2 run-directory convention; set them to the
actual complete RQ2 runs on the execution machine. The locally downloaded `hf_snapshot` analysis
exports are insufficient: `sample_outcomes.csv` lacks the compatibility-parser and prompt
provenance fields required here. Supply the original `manifest.json` and
`dataset/{train,validation,test}.parquet` files. Do not substitute strict correctness.

NRC v2.1's existing checksum-pinned exact-label mapping includes neutral. It is checked at runtime.
If that mapping is replaced, provide its corresponding provenance and reference checksum.

Before a full run, use a new run ID, set `smoke: false`, `layers: null`, and `permutations: 2000`.
Review/freeze the grids and numerical policies, inspect pilot timings and peak memory, and set
`settings_frozen` and `pilot_verified` to true. Set runtime and memory budgets for the host.
Limits are checked between layers and permutations, so one active work unit may exceed a budget.
Full-run p-values are never emitted for smaller smoke schedules. Source/model access and sufficient
model memory are still required for real extraction; the offline test suite does not download models.

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
| Nine-cohort orchestration | `experiments/correctness_geometry.py` |
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
