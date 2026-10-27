# Independent single-label classification

Run commands from `affect-geometry`. Install the project with `pip install -e '.[classification,dev]'`
in the target GPU environment. Supply Hugging Face access through the standard environment or
login mechanism; LLaMA/Gemma checkpoint access must already be granted. No secrets go in config
or output files. This pipeline has no dependency on the local Choi–Weber code.

## Pilot and complete runs

```sh
python -m affect_geometry.classification run --model gemma2 --pilot-limit 12 --run-dir outputs/rq2_classification/gemma2/pilot
python -m affect_geometry.classification run --model gemma2 --run-dir outputs/rq2_classification/gemma2/full
```

Other model keys: `llama2`, `apertus`, `olmo2`. Run each in a separate process/run directory.
`--model-revision` and `--dataset-revision` resolve to immutable Hub commit hashes; omitted values
resolve the current default branch on the first run, and resume reuses stored commits.
`--dtype` defaults to bfloat16; `--device` defaults to auto. The per-model configs use batch size 8 with left padding and attention masks.
Override with `--batch-size`; direct runs without a config default to 1. Each response is
checkpointed separately. Failed batches fall back to individual inference with retries. Generation uses greedy decoding with a ten-token cap. Overlength inputs fail
visibly rather than being silently truncated. Validate GPU compatibility and native templates in
the training-only pilot before committing to full runs.

Rerun the same command/directory to resume. Completed sample IDs are skipped, failures are retried,
and altered settings or implementation hashes cannot silently mix into the run. A SQLite ledger
commits each attempt transactionally; Parquet snapshots are written at the end of processing.
Interruptions retain committed attempts; rerun to finish and export. Exhausted failures produce
partial reports and a nonzero exit. Each full dataset should contain 45,446 single-label rows,
including neutral, with original split sizes 36,308 / 4,548 / 4,590.

## Reports and comparisons

```sh
python -m affect_geometry.classification report --run-dir outputs/rq2_classification/gemma2/full
python -m affect_geometry.classification compare --run-dir outputs/rq2_classification/comparison --compare-runs outputs/rq2_classification/gemma2/full outputs/rq2_classification/llama2/full
```

`analyses/summary.csv` contains all dataset/split/emotion combinations, counts, explicit denominator
rates, Wilson intervals, and compatibility diagnostics. `sample_outcomes.csv` supports inspecting
raw failures. `sample_indexes.json` includes every split/emotion/Boolean outcome group, including
empty groups. Confusion counts include an INVALID column; label metrics use zero for undefined
precision/recall/F1 and macro averages cover all 28 labels. Figures read their source CSVs from
`plot_data/`. Comparisons require identical completed source populations and generate paired
bootstrap differences, grouping identical texts together. No causal claims follow from these
four architecturally different models.

## Publish a separate dataset per model

```sh
python -m affect_geometry.classification publish --run-dir outputs/rq2_classification/gemma2/full --repo-id YOUR_NAMESPACE/rq2-goemotions-single-label-gemma2
```

Publication defaults to private; add `--public` deliberately when appropriate. Pilot and partial
runs cannot be published as complete releases. Original splits are retained. The publisher saves
a model-specific card, manifest, and reports and verifies the exact remote revision against the
local records. `publication.json` records the verified revision. Choose distinct repositories for
different models; never repurpose another model's existing repository.

`correct`/`valid_label` are primary strict outcomes; `compat_correct`/`compat_valid_label` are
complementary first-word outcomes. Raw outputs, rendered prompts, token IDs and finish reasons
remain available regardless of parser acceptance. Infrastructure errors retain null correctness
until resolved. No activation extraction, geometry analysis, probe training, or model fine-tuning
runs in this pipeline.

## Verification boundary

Offline tests exercise filtering, all parser examples, scoring, denominator arithmetic, persisted
error recovery, resume rejection, Parquet types, sample reports, and plots with injected model
responses. Real checkpoint generation and authenticated Hub publication require separate target-
environment integration validation; unit tests are not a claim that either has been performed.

## Per-model configurations: local evaluation first, upload later

Run from the `affect-geometry` project root after installing the project. Each JSON selects
exactly one model; there is no all-model inference launcher:

| Model | Config | Local run directory |
| --- | --- | --- |
| LLaMA-2-7B | `configs/classification/llama2.json` | `outputs/rq2_classification/llama2/full` |
| Gemma-2-9B | `configs/classification/gemma2.json` | `outputs/rq2_classification/gemma2/full` |
| Apertus | `configs/classification/apertus.json` | `outputs/rq2_classification/apertus/full` |
| OLMo-2 | `configs/classification/olmo2.json` | `outputs/rq2_classification/olmo2/full` |

Select one configuration:

```sh
python -m affect_geometry.classification run --config configs/classification/gemma2.json
```

Explicit CLI flags override config values. For a separate pilot:

```sh
python -m affect_geometry.classification run --config configs/classification/gemma2.json --pilot-limit 12 --run-dir outputs/rq2_classification/gemma2/pilot
```

Paths in configs are relative to the working directory (use the project root). Defaults resolve
Hub revisions on first run and store immutable commits for resumption. Use a new run directory
when changing settings. Evaluation downloads source data/checkpoints as needed but **never uploads
predictions or reports**. Each attempted response is committed to the local SQLite checkpoint;
after processing, train/validation/test Parquet files are saved in `dataset/`, metrics and example
indexes in `analyses/`, plot source tables in `plot_data/`, and figures in `plots/`.

Only after reviewing local outputs, invoke the dedicated uploader:

```sh
python script/upload_classification.py --config configs/classification/gemma2.json --repo-id YOUR_NAMESPACE/rq2-goemotions-single-label-gemma2
```

Or select a previously saved run directly:

```sh
python script/upload_classification.py --run-dir outputs/rq2_classification/gemma2/full --repo-id YOUR_NAMESPACE/rq2-goemotions-single-label-gemma2
```

The uploader loads no model and performs no inference. It uploads the split datasets (including
raw model responses and rendered prompts), metrics, sample outcome tables/indexes, plot-ready
CSVs, figures, manifest and available audit files. It rejects pilot/partial runs, missing required
metrics, mismatched model configs, and repositories belonging to a different run. Default
visibility is private; `--public` explicitly selects public visibility. It verifies the remote
dataset records and stores the verified Hub revision locally in `publication.json`.

Batch size is recorded in the manifest; changing it requires a new run directory. Existing
batch-size-1 runs are not silently merged with batch-size-8 predictions. Validate batched
versus single-example predictions on the target GPU during the pilot.
