# Running M03

M03 implements the current label-free scope in [m-03-docs.md](../../m-03-docs.md).
Run commands from the `affect-geometry/` repository root. The checked-in configuration is
[configs/m03/gemma2.json](../../configs/m03/gemma2.json). Implementation details and artifact
conventions are in [METHODS.md](METHODS.md); saved files are described in [OUTPUTS.md](OUTPUTS.md).

## Install and test

Use Python 3.10 or newer. Install the CUDA-compatible PyTorch build for your GPU before the
editable project install if your environment does not already supply it.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,m03]'
export MPLCONFIGDIR="$PWD/outputs/m03_lens_comparison/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/outputs/m03_lens_comparison/.cache"

python -m pytest tests/m03 -q
python -m ruff check src/affect_geometry/m03 tests/m03
python -m pytest -q
```

Run the complete offline integration smoke test:

```bash
python -m affect_geometry.m03 smoke --run-id smoke-01
```

This instantiates a random three-block Gemma2 model, synthetic frozen translators and
Jacobian matrices, and invented comments. It executes the pilot, unlabeled reference
calibration, canonical and both alternate-prompt variants, every prediction rule, grouped
bootstrap metrics, sensitivity checks, complete-score export, and plots. Its output has
a `SYNTHETIC.json` marker under `outputs/m03_lens_comparison/synthetic--tiny-gemma2/smoke-01/`.
These outputs validate software execution; they are not Gemma-2-9B research results.

## Prepare the actual EXP2 population

The defaults reuse the existing local pinned EXP2 snapshot. If that directory is absent,
the runner downloads the exact baseline dataset revision from its configured private
Hugging Face repository. For a different local copy, set `baseline_root` in a configuration
copy before preparation. Preserve its `manifest.json` and `data/*.parquet` or
`dataset/*.parquet` structure.

For Gemma access, accept Google's model license with the Hugging Face account used for the
run and provide `HF_TOKEN` through the environment or the project's existing `.env` file.
The runner never prints the token. The account also needs access to the private baseline.

```bash
RUN_ID=m03-gemma2-01
python -m affect_geometry.m03 prepare --config configs/m03/gemma2.json --run-id "$RUN_ID"
```

Preparation uses CPU only. It validates all 45,446 single-label source records, original
split counts, model/data provenance, exact prompt and label order, source identities, strict
greedy parsing, and resolved inference status. It freezes the 64 training pilot IDs, 4,590
test IDs, optional reference IDs, duplicate audit, configuration, and implementation hash.
No classification accuracy is used to choose settings.

Recorded implementation checks are in [VERIFICATION.md](VERIFICATION.md).

## GPU technical pilot

The default device is CUDA and precision is bfloat16, matching EXP2. This environment must
fit the 9B backbone, frozen lens matrices, and extraction workspace. Exact memory and speed
are measured on the real GPU rather than estimated from the training hardware listed in
the lens artifact card. The runner fails immediately if CUDA is requested but unavailable.

```bash
python -m affect_geometry.m03 pilot --config configs/m03/gemma2.json --run-id "$RUN_ID"
```

The pilot loads immutable artifact revisions, checks tensor shapes and the tuned unembedding
hash, records exact source-layer mappings, verifies native final logits, and compares
independent candidate replay with shared prefixes and separate examples with batching.
It extracts all pilot scores without reporting gold-label accuracy. Check `pilot_checks.json`,
`freeze.json`, and `runtime/*.json` before the full extraction.

Batch and projection sizes are technical choices, without gold-label feedback. If the pilot
runs out of memory, create a configuration copy with smaller `prefix_batch_size` and
`projection_batch_size`, or `lens_device: "cpu"` to stream the readout matrices from host memory.
Use a new run ID and repeat preparation and the pilot. These settings are part of the frozen
cache identity; the runner does not silently change them or drop layers.

## Extract and evaluate

With the default configuration, extract both canonical and reversed-menu prompts, then
evaluate every declared method and reconstruct the figures:

```bash
python -m affect_geometry.m03 extract --config configs/m03/gemma2.json --run-id "$RUN_ID"
python -m affect_geometry.m03 analyze --config configs/m03/gemma2.json --run-id "$RUN_ID" --export-scores
python -m affect_geometry.m03 plot --config configs/m03/gemma2.json --run-id "$RUN_ID"
python -m affect_geometry.m03 status --config configs/m03/gemma2.json --run-id "$RUN_ID"
```

The native candidate baseline scores every canonical label with the same token paths and
EOS event as each lens. The original strict greedy baseline is joined by exact source IDs.
The six headline lens settings are fixed before test metrics are computed. Full layer/beta
scans are descriptive; the best test point is never promoted into the headline table.

To run preparation through plotting in one process, loading the model once:

```bash
python -m affect_geometry.m03 run --config configs/m03/gemma2.json --run-id "$RUN_ID"
```

The `run` command also exports the long complete-label score tables. For later exports from
a completed cache, without loading the backbone:

```bash
python -m affect_geometry.m03 export-scores --config configs/m03/gemma2.json --run-id "$RUN_ID" --variant canonical
python -m affect_geometry.m03 export-scores --config configs/m03/gemma2.json --run-id "$RUN_ID" --variant alternate_menu
```

`analyze` and `plot` need no GPU. `analyze` without `--export-scores` skips the long score-table
export; all compact score shards, prediction tables, metrics, and plot CSVs are still retained.

## Optional label-free analyses

Freeze optional settings in a configuration copy **before** preparation, using a new run ID.

- `calibrated: true`: run the Belrose-style diagnostic with up to 2,048 unique unlabeled
  training comments. It is a threshold statistic, not an alternative top-1 classifier.
- `length_normalized: true`: evaluate label+EOS score divided by label+EOS token count, with
  a matched native candidate baseline. Label-token-only sensitivity is always included.
- `alternate_prompt: true`: recompute the fixed, placeholder-free second formulation on a
  deterministic test sample of at most `pilot_size` comments. The reverse-menu check is
  enabled by default on the entire test population.

The supplied [all-label-free config](../../configs/m03/gemma2-all-label-free.json) enables
all three optional analyses and inherits the remaining pinned `Config` defaults. Use it
for every stage with a fresh run ID. To execute all of them in one process:

```bash
python -m affect_geometry.m03 run --config configs/m03/gemma2-all-label-free.json --run-id m03-gemma2-all-01
```

When calibration is enabled, after preparation and the pilot the separate stages are:

```bash
python -m affect_geometry.m03 extract --config configs/m03/gemma2-all-label-free.json --run-id "$RUN_ID" --phase reference
python -m affect_geometry.m03 calibrate --config configs/m03/gemma2-all-label-free.json --run-id "$RUN_ID"
python -m affect_geometry.m03 extract --config configs/m03/gemma2-all-label-free.json --run-id "$RUN_ID" --phase test
```

Use the same all-label-free configuration for preparation and the pilot too, with a new
run ID. Thresholds must be frozen before test extraction. The one-process `run`
command performs these stages automatically when calibration is enabled.

## Resume and interpretation

Repeat `extract` with the same config and run ID to resume: complete sample shards are
validated and reused; missing samples are recomputed. An unfinished comment batch replays
its original completed batchmates to preserve numerical batch context, keeping their saved
shards unchanged; fully complete batches do not invoke the backbone. It never reports a full-population
metric when any required shard is missing or malformed. A changed config, implementation,
extraction software version, prompt, token path, artifact, or layer mapping is rejected.

`freeze.json` records actual lens coverage. Logit readout includes every post-block state;
tuned and Jacobian readout include every supported fitted source layer. Unsupported depths
are reported. The final native baseline does not require a fitted translator.

The default test-history statement is conservative because prior EXP2 outcomes were available.
Set `test_history` to the actual inspection/selection history before preparation. Current
intervals are paired, pointwise 95% duplicate-text-group bootstrap intervals with 2,000
replicates, not simultaneous bands or significance tests over a searched layer range.

Validation-selected layers, validation-selected beta, negative-beta extensions, fitting new
lenses, and replication on models lacking compatible artifacts remain outside this run.
