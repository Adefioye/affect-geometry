# Experiment orchestration

This directory contains shell entry points that coordinate complete, reproducible experiments.
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
