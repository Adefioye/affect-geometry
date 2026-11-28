# Run EXP5: oracle-guided neuron-community transfer

The [scientific specification](../docs/exp5.md) is implemented by
`python -m affect_geometry.neurons`. Model weights remain frozen. The default study uses
OLMo-2-1124-7B-Instruct, up to 50 strict-correct training comments per emotion, up to 20
communities, strength 0.25, and the full single-label GoEmotions test population.

**Run order:** start with `allenai/OLMo-2-1124-7B-Instruct` on the A40.
**Git branch:** use `circuits` for EXP5 implementation and execution.

## Environment and inputs

From the repository root, switch to `circuits`, then install in an isolated environment
with a suitable PyTorch build for the target GPU:

```bash
git switch circuits
pip install -e '.[neurons,dev]'
```

The `neurons` extra adds Accelerate, SentencePiece, python-igraph 0.11.8, and leidenalg 0.10.2.
It does not change the project's Python >=3.10 requirement or install Goodfire's entire package.
The pinned Goodfire census reference and MIT notice are retained under `third_party/`;
the adapted Curveball implementation is tested against that reference.

Edit [configs/neuron_transfer.json](../configs/neuron_transfer.json):

- Set `classification_dir` to a completed local EXP2 snapshot containing `manifest.json` and
  `dataset/train.parquet`, `dataset/validation.parquet`, and `dataset/test.parquet`. The validation
  file is read for source integrity only; it is not used for discovery, selection, or tuning.
- Use an explicit new `run_id`. Model and dataset revisions are inherited from the saved manifest
  and must be immutable 40-character commit hashes. Match the EXP2 dtype.
- Set `device` for the execution host. The default `score_batch_size=32` targets the planned A40
  run: all 28 baseline label continuations are scored together, while edited diagnostics use two.
  Attribution and recipient generation each process one comment per pass. Run the smoke stage
  on the target GPU before scaling; a changed setting requires a new configuration/run identity.
- Set `test_previously_inspected=true` if the test examples/outcomes informed method choices.
  `allow_partial_source=true` is for development fixtures only and marks results exploratory.

The initial OLMo adapter and the Llama adapter are exercised with tiny randomly initialized
models. Full checkpoint execution still requires the pinned EXP2 source, model access, sufficient
memory, and the target-device smoke checks. No GPU-hour estimate is inferred from community count.

## Stages

Run stages in this order with the same config. `prepare`, `graph`, `freeze`, and `report` do not
load model weights. The other stages require the model execution environment.

```bash
python -m affect_geometry.neurons --config configs/neuron_transfer.json --stage prepare
python -m affect_geometry.neurons --config configs/neuron_transfer.json --stage smoke
python -m affect_geometry.neurons --config configs/neuron_transfer.json --stage attribute
python -m affect_geometry.neurons --config configs/neuron_transfer.json --stage graph
python -m affect_geometry.neurons --config configs/neuron_transfer.json --stage select
python -m affect_geometry.neurons --config configs/neuron_transfer.json --stage freeze
python -m affect_geometry.neurons --config configs/neuron_transfer.json --stage test
python -m affect_geometry.neurons --config configs/neuron_transfer.json --stage report
```

`--stage all` executes this sequence. Separate stages let you inspect training support,
`smoke_profile.json`, and `compute_budget.json` before starting the full fixed test evaluation.
The budget records actual shortlist/population sizes, condition-count bounds, and a rough linear
projection from measured smoke timings that excludes donor work. It is not a GPU-hour guarantee.
Changes after inspecting test interventions
require a new run and fresh confirmation data for a confirmatory claim.

| Stage | Output/behavior |
| --- | --- |
| prepare | Audits EXP2; saves deterministic discovery, selection, smoke, and test IDs; excludes donor duplicates; derives wrong-label alternatives from all valid-wrong training predictions |
| smoke | Replays EXP2; checks complete paths, prompt logits, no-op/self-donor identity and signed finite differences; records component timings, memory and numerical tolerance |
| attribute | Computes sparse signed top-k records; transactionally accumulates full-neuron corpus means for the unclustered control |
| graph | Counts sparse layer tiles; applies the pinned combined-row Curveball null and stabilized excess score; saves all Leiden candidates |
| select | Uses training-only gold/rival margin ablations and random groups; selects up to 20 nonredundant communities and the primary ID |
| freeze | Saves communities, controls, donor vocabulary/rules, test IDs, a configuration fingerprint, and compute bounds |
| test | Reproduces baselines; fixes donor renderings; runs community, random, unclustered, and wrong-label arms; scores held-out correct-case ablations |
| report | Writes full-population/common-coverage metrics, label support, primary paired intervals, donor reuse, and a CSV-driven figure |

No supported communities is a completed discovery finding. `all` then writes a finding report and
skips test interventions. It does not fabricate a community or lower thresholds.

## Resume and artifacts

Artifacts are under `outputs/neuron-transfer/<model-slug>/<run_id>/`; logs are under
`logs/<model-slug>__<run_id>/run.log`, using the shared project logger/progress utilities.

Model stages use per-stage transactional SQLite records. Interrupting a stage leaves completed
conditions available for reuse. Donor caches are keyed by donor, exact matched rendering, and the
required coordinate union. Completed stage artifacts are checksummed. Changing source bytes, code,
software, configuration, or predecessor artifacts invalidates reuse; use a new run ID for a revision.

Important files include `cohorts.csv`, `cohort_support.csv`, `smoke_profile.json`, `label_paths.json`,
`attribution.parquet`, `importance_means.parquet`, `graph_edges.parquet`, `graph_null_manifest.json`,
`communities.json`, `community_effects.parquet`, `community_support.csv`, `shortlist.json`,
`controls.json`, `frozen.json`, `compute_budget.json`, `donor_pairs.parquet`, `predictions.parquet`, `path_scores.parquet`,
`edit_norms.parquet`, `test_ablation_effects.parquet`, `metrics.csv`, `primary_contrasts.csv`,
`per_label_metrics.csv`, `donor_usage.csv`, `score_diagnostics.csv`, `heldout_ablation_summary.csv`,
`edit_norm_summary.csv`, and `report.md`.

Baseline scores contain all 28 canonical paths. Edited conditions contain only gold and a frozen
rival: they do not provide edited 28-way membership accuracy. Invalid responses and donor abstentions
remain in the primary denominator. Held-out ablations measure score changes, not generated label flips.
The report keeps the oracle qualifier and identifies exploratory comparisons.

Finite differences use a fixed activation-scale grid and the realized native-dtype edit size.
The most stable adjacent pair is selected without looking at the analytical gradient, then checked
for sign and numerical agreement. Every trial is saved; an unresolved check stops the smoke stage.

## Verification

```bash
pytest -q tests/test_neuron_transfer_native.py tests/test_neuron_transfer_graph.py tests/test_neuron_transfer_pipeline.py
```

These tests use actual tiny Llama/OLMo2 models for native scoring, gradients and patch behavior,
the pinned census source for Curveball parity, real Leiden clustering, and a deterministic synthetic
policy fixture for stage orchestration and report generation. They do not measure GoEmotions repair.
