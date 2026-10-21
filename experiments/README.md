# Experiment orchestration

This directory contains entry points that coordinate complete, reproducible experiments.
Reusable scientific computation remains in `affect_geometry`; these scripts select a configuration
and invoke the existing stages in the required order.

## Last-token experiment

`01_run_last_token.sh` runs the complete GoEmotions pipeline using only the last non-special content
token from every selected example. Its configuration is `configs/last_token.json`, whose
`activations.poolings` value is exactly `["last_token"]`.

From the repository root:

```bash
experiments/01_run_last_token.sh --dry-run
experiments/01_run_last_token.sh
```

The script creates one UTC run ID and passes it to every stage. To choose the ID explicitly:

```bash
experiments/01_run_last_token.sh --run-id 20261019T153045Z
```

If an already completed stage should not be repeated, reuse the same run ID and select the first
stage that still needs to run:

```bash
experiments/01_run_last_token.sh \
  --run-id 20261019T153045Z \
  --start-at rdm
```

`--start-at` assumes all required artifacts from earlier stages already exist. It does not repair a
partially written stage; restart at the stage that was interrupted.

## What it runs

1. `prepare` downloads GoEmotions, keeps examples having exactly one emotion label, and stores the
   combined train/validation/test sample table and dataset metadata.
2. `extract` loads the configured causal language model in inference mode, captures the post-block
   residual stream at every decoder layer, takes only the last non-special content token, and writes
   one float16 memory-mapped matrix per layer. It does not compute or store mean pooling.
3. `rdm` reduces each layer to at most 64 PCA components and constructs crossnobis, RBF-MMD, energy,
   sliced-Wasserstein, cosine, and Euclidean emotion-level RDMs.
4. `embed` creates classical-MDS and Isomap embeddings in 1D, 2D, and 3D; computes intrinsic
   spectra through 10 dimensions; saves Isomap geodesic distances; and automatically runs the
   Isomap neighbor grid `[3, 4, 5, 6, 8, 10, 12]`. The principal Isomap coordinates use `k=5`;
   the sweep is diagnostic and does not choose a replacement value automatically.
5. `analyze` computes embedding quality, V–A recovery, intrinsic-dimensionality summaries,
   geodesic/direct-distance diagnostics, agreement across distance notions, neighborhood changes,
   and cross-layer similarity.
6. `plot` reads the self-contained tables in `plot_data/` and renders the configured figures.

Artifacts are written to `outputs/<model-slug>/<run-id>/`. Structured stage logs are written to
`logs/<model-slug>__<run-id>/`. The script does not upload anything to Hugging Face; uploads remain
an explicit post-run action through the scripts in `script/`.

## LT-NeutralProjected experiment

`03_run_lt_neutral_projected.py` runs the last-token neutral-projected variant with
`configs/lt_neutral_projected.json`. GoEmotions contains **27 emotions plus neutral (28 nodes)**;
every exploratory plot includes all of them.

```bash
python experiments/03_run_lt_neutral_projected.py --dry-run
python experiments/03_run_lt_neutral_projected.py --show
```

`--show` opens the completed `exploration.html` gallery in a browser. On a headless machine,
open that file later with its adjacent `plots/` directory. The gallery filters by layer, distance,
method, or dimension; click a plot to see it at full size. It requires no server or external assets.
All 3D plots are saved as static views.

To reuse sample activations from an existing last-token run without copying them or loading a model:

```bash
python experiments/03_run_lt_neutral_projected.py \
  --source-run outputs/<model-slug>/<last-token-run-id> \
  --run-id neutral-projected-study --show
```

The source must contain `manifest.json`, `dataset/samples.csv`, `activations/metadata.json`, and
`activations/last_token/layer_*.npy`. The configured model must match the source. The source run
is left unchanged, while its sample table and provenance are saved in the new output run.

The stages are `prepare`, `reference`, `extract`, `rdm`, `embed`, `analyze`, and `plot`.
With `--source-run`, preparation and extraction are skipped. Resume with the **same run ID,
config, source, and lexicon options**, adding `--start-at <stage>`. For example, to regenerate
figures from the saved plot-ready CSVs:

```bash
python experiments/03_run_lt_neutral_projected.py \
  --run-id neutral-projected-study --start-at plot --show
```

The reference stage validates the repository-local `data/reference/goemotions_nrc_vad.csv`
and snapshots it with provenance into `references/goemotions_nrc_vad.csv` under the run.
This CSV contains dominance alongside V/A scores that match `goemotions_nrc_va.csv` exactly;
all three axes retain their native `[-1, 1]` scale. Both the 2D V–A and 3D V–A–D evaluations use
the run-local snapshot. **The default reference stage needs no internet, ZIP, or full lexicon.**
Set `analysis.vad_reference` in the config or pass `--vad-reference /path/to/reference.csv`
to select another prepared reference.

To rebuild a run's reference from an official v2.1 ZIP or English TSV instead, use:

```bash
python experiments/03_run_lt_neutral_projected.py \
  --nrc-vad-lexicon /path/to/NRC-VAD-Lexicon-v2.1.zip --show
```

In that mode, the original archive/TSV and its SHA-256 provenance are saved under the run's
`outputs/` directory. See [reference documentation](../data/reference/README.md).

The transformation follows the existing LT-NeutralProjected definition: subtract the equal-weight
mean of the **non-neutral emotion centroids**, fit covariance PCA on **within-neutral sample
variation**, and project out the smallest leading neutral subspace reaching 50% cumulative variance
(configurable). The neutral sample mean is used to fit PCA; it is not an additional translation
of the projected vectors. The neutral node remains in every RDM and embedding.

All six distance metrics use the **full hidden dimension after neutral projection**. Cosine and
Euclidean use the exact projected centroids; crossnobis, RBF-MMD, energy, and sliced-Wasserstein
use full-width projected sample activations. There is **no PCA after projection** for any metric.
The config sets `rdm.pca_components` to `null`; the neutral-projected RDM stage also enforces this
behavior if a reused config specifies a numeric value. PCA is used only to identify the neutral
directions removed by the representation transformation. Distribution metrics retain the seeded
per-class sample cap; crossnobis uses all samples in the selected split.

The embedding stage retains the original experiment's 1D/2D/3D classical MDS and Isomap,
eigenspectra through ten dimensions, geodesic distances, and Isomap neighbor sweep. The main
Isomap uses `k=5`, increasing it only if necessary for connectivity; exported coordinates record
the actual `k`. The sweep does not select a replacement automatically.

Exploration plots show **embedding coordinates only**, translated so neutral is the origin.
Their colors and kNN background regions use the GoEmotions positive/negative/ambiguous/neutral
taxonomy independently of NRC scores. 2D plots include quadrant boundaries through neutral,
inspired by [Choi–Weber, Figure 2](https://arxiv.org/html/2604.07382v1#S4.F2).
The orientation remains the unaligned eigenvector orientation: signs and quadrant assignments
are descriptive, and the axes are not interpreted as valence or arousal. Translation does not
change pairwise distances. These presentation choices do not alter the geometry used for scoring.

Analysis retains embedding quality, V–A recovery, intrinsic dimension, geodesic/direct ratios,
distance and neighborhood agreement, and cross-layer similarity. It adds neutral projection
diagnostics, quadrant counts, and **separate 3D VAD Procrustes recovery**, per-axis correlations,
and refitted label-permutation tests. Recovery scores are in-sample exploratory summaries.
The exploration figures never overlay reference points or aligned reference coordinates.

Artifacts are under `outputs/lt_neutral_projected/<model-slug>/<run-id>/`:

- `activations/last_token/`: full-width float32 neutral-projected sample activations,
  with completion and transformation metadata for uploading. Source activations stay unchanged.
- `checkpoints/neutral_projection/`: centering vectors, removed PCA directions, spectra, sample
  counts, and projected centroids. No post-projection PCA checkpoints are produced.
- `analyses/`: standard diagnostics, `neutral_projection.csv`, `neutral_quadrant_counts.csv`,
  `vad_recovery.csv`, and `vad_alignment.csv`.
- `plot_data/exploration_coordinates.csv`: self-contained coordinates, labels, sample counts,
  taxonomy groups, quadrant assignments, neutral translations, graph neighbors, model/run,
  transformation parameters, and source context. It contains no NRC scores or aligned coordinates.
- Other `plot_data/*.csv`: the separate plot-ready summaries consumed by plotting functions.
- `plots/exploration/`: one 1D/2D/3D figure per layer, distance, and embedding method.
- `exploration.html` and `plot_manifest.json`: browsable gallery and CSV/figure mappings.

Logs stay in `logs/<model-slug>__<run-id>/run.log`. Neither runner uploads artifacts.

## Comprehensive LT-NeutralProjected analysis

`04_analyze_lt_neutral_projected.py` audits the frozen full-space projected run and writes
`analyses/lt_neutral_projected/README.md`, embedding 30 report figures and numerical tables.
The analysis directory includes local `figures/`, `tables/`, `plot_data/`, `metadata/`,
`source/` and `plots/` subdirectories, plus the original exploration gallery. Canonical generated
artifacts are also retained under the source run's `report/` directory in `outputs/`.

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python experiments/04_analyze_lt_neutral_projected.py
```

Use `--render-only` to redraw the report from its saved audit tables, or `--bundle-only` to
refresh the local package without rerunning computations or plots. The analysis includes
4,999 shared-label permutation fits with family-level selection checks, held-out-label axis
scores, 999-draw eigengap tests, metric and graph sensitivity, reference RSA, and 200 balanced
sample draws. Sample-split checks condition on the original all-split projection. The narrative
is curated for the documented frozen run and default settings; the scientific computations are
reusable through `affect_geometry.analysis.projected_audit`.
