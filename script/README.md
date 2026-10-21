# Hugging Face upload scripts

These scripts archive completed `outputs/<model-slug>/<run-id>/` artifacts in Hugging Face dataset
repositories. The model is a required argument because it is part of the local path. The default
remote directory mirrors the same model/run hierarchy:

```text
<model-slug>/<run-id>/
```

The canonical run ID is a compact UTC datetime, `YYYYMMDDTHHMMSSZ`; for example,
`20261018T153045Z`. This form sorts chronologically, is unambiguous across RunPod machines, and is
safe in both local and Hub paths.

`python -m run --config configs/default.json` generates this ID once and records the resolved ID in
`manifest.json`. If you run stages separately, create one ID and pass the same `--run-id` to every
stage. The value passed to the upload script must be that resolved ID, not `auto`.

The scripts create missing repositories as **private** by default. An existing repository keeps
its current visibility. Use `--public` only when the artifacts and source data are appropriate for
public release.

## Setup

Install a current Hub client and authenticate without placing tokens in command history:

```bash
python -m pip install -U huggingface_hub
hf auth login
```

`HF_TOKEN` may be supplied through the environment instead. For large RunPod uploads,
`huggingface_hub` uses Xet automatically. `HF_XET_HIGH_PERFORMANCE=1` can improve throughput on a
machine with fast local storage and networking.

Always run `--dry-run` first. It performs local validation and reports the selected file count and
total size without contacting Hugging Face. Both scripts also require the supplied model and run ID
to match `manifest.json`, preventing a mislabeled upload.

## Upload compact run results

`upload_run_outputs.py` uploads:

- `manifest.json`
- `dataset/`
- `checkpoints/`
- `rdms/`
- `embeddings/`
- `analyses/`
- `plot_data/`
- `references/` when present, plus `representation.json`, `plot_manifest.json`, and
  `exploration.html` when present.

It excludes `activations/` and the regenerable `plots/` directory by default.

```bash
python script/upload_run_outputs.py \
  --model Qwen/Qwen3-0.6B \
  --run-id 20261018T153045Z \
  --repo-id YOUR_USER/affect-geometry-runs \
  --dry-run

python script/upload_run_outputs.py \
  --model Qwen/Qwen3-0.6B \
  --run-id 20261018T153045Z \
  --repo-id YOUR_USER/affect-geometry-runs
```

Add `--include-plots` to upload all generated PNGs. Add `--allow-partial` only when intentionally
archiving a run that has not completed the RDM, embedding, analysis, and plot-data stages.

## Upload model activations

`upload_model_activations.py` uploads `manifest.json`, `dataset/`, and `activations/` to a
dedicated dataset repository:

```bash
python script/upload_model_activations.py \
  --model Qwen/Qwen3-0.6B \
  --run-id 20261018T153045Z \
  --repo-id YOUR_USER/affect-geometry-activations \
  --dry-run

python script/upload_model_activations.py \
  --model Qwen/Qwen3-0.6B \
  --run-id 20261018T153045Z \
  --repo-id YOUR_USER/affect-geometry-activations
```

Before uploading, the script requires `activations/metadata.json` and verifies every configured
pooling has the expected number of layer files and that every array has the expected sample and
hidden dimensions. This prevents uploading preallocated but incomplete activation stores.

The `dataset/` directory is included because `dataset/samples.csv` defines the row-to-sample
mapping for every activation matrix.

## Common options

```text
--output-root PATH     Local output root; defaults to outputs
--model NAME           Required model name; locates and verifies the run
--path-in-repo PATH    Override the generated model/run directory
--private              Create a missing repository as private (default)
--public               Create a missing repository as public
--dry-run              Validate and print the upload plan only
```

Run the scripts only after the corresponding pipeline stage completes. In particular, do not
upload activation `.npy` files while extraction is still writing them.

## Extract and upload mean-pooled activations

`extract_mean_pooled_activations.py` reproduces the model and tokenizer settings from a reference
last-token run. It copies that run's dataset artifacts byte-for-byte, extracts only mean-pooled
activations, validates every layer, and then uploads the new timestamped run. The project's `.env`
file is loaded without overriding credentials already exported in the environment.

Using a reference already on Hugging Face:

```bash
python script/extract_mean_pooled_activations.py \
  --reference-path-in-repo qwen--qwen3-0.6b/REFERENCE_RUN_ID \
  --run-id NEW_MEAN_RUN_ID
```

Using a local reference run and retaining the result locally only:

```bash
python script/extract_mean_pooled_activations.py \
  --reference-run outputs/qwen--qwen3-0.6b/REFERENCE_RUN_ID \
  --run-id NEW_MEAN_RUN_ID \
  --no-upload
```

The local and remote activation path is `<model-slug>/<run-id>/activations/mean/`. The default
destination repository is `kokolamba/affect-geometry-activations`; use `--repo-id` to override it.

If the reference did not record a model revision, supply `--model-revision COMMIT_ID`.
The new run pins that commit and records that the reference model version could not be verified.
A supplied revision cannot override a different revision already recorded in the reference.

## Run the representation comparison

`run_representation_comparison.py` accepts local activation runs or downloads them from Hugging
Face. Its default output root is exactly:

```text
outputs/representation_comparison/
```

For remotely stored runs:

```bash
python script/run_representation_comparison.py \
  --last-token-path-in-repo qwen--qwen3-0.6b/LAST_TOKEN_RUN_ID \
  --mean-path-in-repo qwen--qwen3-0.6b/MEAN_RUN_ID
```

For local runs:

```bash
python script/run_representation_comparison.py \
  --last-token-run outputs/qwen--qwen3-0.6b/LAST_TOKEN_RUN_ID \
  --mean-run outputs/qwen--qwen3-0.6b/MEAN_RUN_ID
```

The command validates and aligns sample IDs, evaluates the configured representation variants,
and writes representations, neutral-PCA checkpoints, cosine RDMs, MDS coordinates, ranked tables,
plot-ready CSVs, and plots beneath `outputs/representation_comparison/`.
