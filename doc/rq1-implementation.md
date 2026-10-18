# Implementation guide

## Scope

This repository implements a layerwise affect-geometry experiment using the Hugging Face
`google-research-datasets/go_emotions` simplified configuration and `Qwen/Qwen3-0.6B`.
Only examples with exactly one label are retained. Neutral is included by default, giving the
standard 28 labels. The raw comment is passed to the tokenizer; no chat template or correctness
filter is applied.

The default run extracts the post-block residual stream from every decoder layer through native
PyTorch forward hooks. It saves both:

- `last_token`: the final non-padding, non-special content token; and
- `mean`: the mean over all non-padding, non-special content tokens.

If a tokenizer produces an empty content mask, the code safely falls back to its attended tokens.
The causal language-model head is not run because its logits are not needed.

## Install and run

Python 3.10 or newer is required. Transformers 4.51 or newer is specified because that is the
minimum version recommended for Qwen3.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m run --config configs/default.json
```

The stages can be run independently, which is preferable for a long extraction:

```bash
python -m prepare --config configs/default.json
python -m extract --config configs/default.json
python -m rdm --config configs/default.json
python -m embed --config configs/default.json
python -m analyze --config configs/default.json
python -m plot --config configs/default.json
```

`device: "auto"` selects CUDA, then Apple MPS, then CPU. Automatic dtype selects bfloat16 on
compatible CUDA devices, float16 on other accelerators, and float32 on CPU. Reduce `batch_size`
if extraction exhausts accelerator memory.

## Pipeline and conventions

### Dataset and activations

`prepare` downloads the official simplified splits, preserves their names, filters
`len(labels) == 1`, and saves sample IDs, texts, labels, counts, and dataset fingerprints.
`extract` writes one NumPy memory-mapped matrix per layer and pooling. Row order exactly matches
`dataset/samples.csv`.

The default `rdm.split` is `all`, so train, validation, and test examples contribute. Set it to a
single split to restrict an analysis. This is an unsupervised descriptive analysis; no probe
accuracy or model-correctness selection is used.

### Preprocessing

A deterministic stratified sample of at most 256 examples per emotion fits a separate PCA for
each layer and pooling. The default retained dimension is 64. The fitted mean, components, and
variance spectrum are saved as checkpoints. All available examples are then projected through
that PCA for Crossnobis, cosine, and Euclidean RDMs. The capped stratified sample is used for
MMD, Energy, and Sliced Wasserstein because their pairwise sample computations are substantially
more expensive. `rdms/distribution_sample.csv` records every selected row.

Raise `max_samples_per_emotion` to use more examples for the distribution distances. Setting it
above the largest class size makes the cap inactive.

### Dissimilarities

Six concrete matrices implement the four conceptual notions requested:

1. **Crossnobis**: repeated stratified folds, fold-specific pooled Ledoit–Wolf shrinkage precision
   estimated from training samples only, and the average train-fold/test-fold Mahalanobis
   cross-product. Its raw squared estimate is signed and can be negative under noise.
2. **RBF-MMD**: unbiased MMD² using the average of three RBF kernels. The base bandwidth is the
   global median pairwise distance and the default multipliers are 0.5, 1, and 2.
3. **Energy**: the unbiased two-sample energy statistic based on Euclidean sample distances.
4. **Sliced Wasserstein**: mean one-dimensional Wasserstein-1 distance over shared seeded random
   unit projections.
5. **Cosine centroid distance**.
6. **Euclidean centroid distance**.

For Crossnobis, MMD, and Energy, both the signed raw estimate and
`sqrt(max(raw, 0))` are saved. MDS and Isomap use the latter. This retains negative estimates for
diagnostics while giving the embedding algorithms a nonnegative dissimilarity. Wasserstein,
cosine, and Euclidean matrices are used directly.

Every RDM is saved as a compact `.npz` and a tidy pairwise `.csv` with its complete experiment
context.

### MDS and Isomap

Classical metric MDS uses double centering and an eigendecomposition. Isomap builds a
union-symmetric k-nearest-neighbor graph from the precomputed RDM, increases `k` only when needed
for connectivity, computes all-pairs shortest paths, then applies classical MDS. The requested
1D, 2D, and 3D coordinates are exported. The eigenspectrum and Isomap geodesic matrix are also
retained.

The neighbor sweep records requested and actually used `k`, two-dimensional stress, and residual
variance. The main Isomap run records its actual connected-graph `k` on every coordinate row.

### Analyses

The implementation reports:

- V–A recovery: similarity Procrustes R², aligned-axis Pearson correlations, an optional seeded
  permutation p-value, and point-level aligned coordinates;
- MDS versus Isomap: normalized metric Stress-1, Shepard Spearman correlation, trustworthiness,
  continuity, performance differences, and geodesic/direct-distance ratios;
- intrinsic dimensionality: full classical-MDS eigenvalues, dimensions reaching 80/90/95% of
  positive spectral mass, participation ratio, negative eigenvalue fraction, Isomap residual
  variance through ten dimensions, and a residual-variance elbow;
- distance agreement: upper-triangle RDM Spearman correlations, embedding Procrustes R², per-label
  k-nearest-neighbor overlap, and labels whose neighborhoods change most; and
- layerwise emergence: all quality, V–A, and dimension metrics by layer plus all-pairs cross-layer
  Procrustes similarity for two-dimensional embeddings.

The NRC VAD v2.1 terms exactly match all 28 GoEmotions labels. Their centered `[-1, 1]` valence and
arousal values, including neutral, are stored in `data/reference/goemotions_nrc_va.csv` with the
source term used for each label.

## Output layout

Everything generated by a run is below `outputs/<run_id>/`:

```text
dataset/       filtered samples, labels, counts, fingerprints
activations/   layer matrices under last_token/ and mean/
checkpoints/   layerwise PCA parameters
rdms/          NPZ matrices, tidy pairwise CSVs, sampling record
embeddings/    coordinates, spectra, geodesics, Isomap sweeps
analyses/      statistical summary and point/pair-level CSVs
plot_data/     self-contained tidy CSVs used directly by plotting
plots/         embedding, layerwise, intrinsic, and agreement figures
manifest.json  full configuration, git commit, Python and package versions
```

Important portable tables in `plot_data/` are `coordinates.csv`, `layerwise_metrics.csv`,
`va_alignment.csv`, `intrinsic_spectrum.csv`, `isomap_residual_variance.csv`,
`distance_agreement.csv`, `neighborhood_change.csv`, `mds_isomap_comparison.csv`,
`geodesic_ratios.csv`, `geodesic_pair_ratios.csv`, and `cross_layer_similarity.csv`. Plotting reads
these CSVs rather than hidden in-memory results. `coordinates.csv` includes NRC valence and arousal,
so it can recreate the colored 1D/2D/3D figures elsewhere without the source dataset.

## Validation and practical notes

Run the offline scientific tests with:

```bash
pytest -q
ruff check src tests
```

The tests cover token pooling, every RDM's symmetry and diagonal, variance-only distribution
changes, exact Euclidean MDS recovery, connected Isomap geometry, Procrustes invariance, and a
synthetic end-to-end RDM/embedding/analysis run. They require neither Hugging Face access nor a
model download.

The complete default activation store is large: its rough size is
`samples × 28 layers × 1024 features × 2 bytes × 2 poolings`, before downstream artifacts. Run
one pooling at a time or use a smaller dataset split if local storage is constrained. The model
and dataset download occur only in the `prepare`/`extract` stages; all later stages operate on
local artifacts.
