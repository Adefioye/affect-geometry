# Hugging Face upload scripts

These scripts archive completed `outputs/<run-id>/` artifacts in Hugging Face dataset
repositories. Remote directories are organized by the model recorded in `manifest.json` and by
run ID:

```text
models/<model-slug>/runs/<run-id>/
```

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
total size without contacting Hugging Face.

## Upload compact run results

`upload_run_outputs.py` uploads:

- `manifest.json`
- `dataset/`
- `checkpoints/`
- `rdms/`
- `embeddings/`
- `analyses/`
- `plot_data/`

It excludes `activations/` and the regenerable `plots/` directory by default.

```bash
python script/upload_run_outputs.py \
  --run-id qwen3_0.6b_goemotions \
  --repo-id YOUR_USER/affect-geometry-runs \
  --dry-run

python script/upload_run_outputs.py \
  --run-id qwen3_0.6b_goemotions \
  --repo-id YOUR_USER/affect-geometry-runs
```

Add `--include-plots` to upload all generated PNGs. Add `--allow-partial` only when intentionally
archiving a run that has not completed the RDM, embedding, analysis, and plot-data stages.

## Upload model activations

`upload_model_activations.py` uploads `manifest.json`, `dataset/`, and `activations/` to a
dedicated dataset repository:

```bash
python script/upload_model_activations.py \
  --run-id qwen3_0.6b_goemotions \
  --repo-id YOUR_USER/affect-geometry-activations \
  --dry-run

python script/upload_model_activations.py \
  --run-id qwen3_0.6b_goemotions \
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
--model NAME           Verify the model recorded in manifest.json
--path-in-repo PATH    Override the generated model/run directory
--private              Create a missing repository as private (default)
--public               Create a missing repository as public
--dry-run              Validate and print the upload plan only
```

Run the scripts only after the corresponding pipeline stage completes. In particular, do not
upload activation `.npy` files while extraction is still writing them.
