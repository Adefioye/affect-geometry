# Run EXP5: oracle-guided neuron-community transfer

The [scientific specification](../docs/exp5.md) is implemented by
`python -m affect_geometry.neurons`. Model weights remain frozen. The default study uses
OLMo-2-1124-7B-Instruct, up to 50 strict-correct training comments per emotion, up to 20
communities, strength 0.25, and 100 held-out GoEmotions comments from each saved EXP2 outcome
(strict correct and strict incorrect).

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
  run and batches all 28 canonical continuations together for the baseline and each transfer
  arm. The training and held-out ablation diagnostics score the gold and fixed rival paths.
  Attribution and recipient generation each process one comment per pass. Run the smoke stage
  on the target GPU before scaling. A changed setting requires a new run identity.
- `test_per_outcome=100` orders each saved outcome with a fixed seed and checks candidates until
  100 per outcome agree with the unedited EXP5 baseline. The incorrect stratum includes wrong
  labels and invalid outputs.
- Set `test_previously_inspected=true` if the test examples/outcomes informed method choices.
  `allow_partial_source=true` is for development fixtures only and marks results exploratory.

The initial OLMo adapter and the Llama adapter are exercised with tiny randomly initialized
models. Full checkpoint execution still requires the pinned EXP2 source, model access, sufficient
memory, and the target-device smoke checks. No GPU-hour estimate is inferred from community count.

## Stages

Run stages in this order with the same config. `prepare`, `graph`, and `report` do not
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
`smoke_profile.json`, and `compute_budget.json` before starting the fixed held-out evaluation.
The budget records actual shortlist/sample sizes, condition-count bounds, and a rough linear
projection from measured smoke timings that excludes donor work. It is not a GPU-hour guarantee.
Changes after inspecting test interventions
require a new run and fresh confirmation data for a confirmatory claim.

| Stage | Output/behavior |
| --- | --- |
| prepare | Audits EXP2 and saves deterministic discovery, selection, smoke, and held-out candidate IDs. Excludes donor duplicates and derives wrong-label alternatives from all valid-wrong training predictions. |
| smoke | Replays EXP2; checks complete paths, prompt logits, no-op/self-donor identity and signed finite differences; records component timings, memory and numerical tolerance |
| attribute | Computes sparse signed top-k records; transactionally accumulates full-neuron corpus means for the unclustered control |
| graph | Counts sparse layer tiles; applies the pinned combined-row Curveball null and stabilized excess score; saves all Leiden candidates |
| select | Uses training-only gold/rival margin ablations and random groups; selects up to 20 nonredundant communities and the primary ID |
| freeze | Screens candidates in seeded order, records EXP2 agreement, and freezes test IDs and unedited baselines alongside communities, controls, donor rules, a configuration fingerprint, and compute bounds. |
| test | Evaluates strict generation and 28-way membership top-1 in each transfer arm, using frozen unedited baselines and matched donor renderings. Scores held-out correct-case ablations. |
| report | Writes both accuracy metrics for the held-out sample and common-coverage subset, per-label recall, paired intervals, donor reuse, and a CSV-driven figure |

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

Strict accuracy is the primary endpoint. It scores the decoded greedy response with the EXP2
strict parser. Membership top-1 accuracy is secondary. It selects the canonical label with the
highest total log probability for the label tokens followed by the native stop token. Ties use
canonical label order. Each edited transfer arm scores all 28 paths under the same edit used for
generation. Invalid generated responses contribute to both metrics, and donor abstentions reuse
the unedited scores and generation. Held-out ablations measure changes in the gold–rival margin,
the gold label-token score, and the gold stop-token score.

`predictions.parquet` records both predictions and correctness flags. `path_scores.parquet`
contains the 28 continuations for each transfer condition. `metrics.csv` includes paired baseline
and arm accuracies, their changes, and descriptive 95% duplicate-group bootstrap intervals on
the held-out sample. `primary_contrasts.csv` contains the adjusted strict-accuracy contrasts.
`plot_data/accuracy_deltas.csv` contains the data for both panels of the accuracy figure.
Runs saved before membership scoring require a new run ID to evaluate the missing edited paths.

Saved EXP2 correctness defines the sampling strata. If correctness changes under the unedited
EXP5 runner, the next candidate in that stratum is checked. Response or token changes that
preserve correctness are recorded and the candidate can still be included. Paired metrics and
held-out correct-case ablations use the unedited EXP5 baseline, which uses the same
single-comment generation as edits. `heldout_screening.csv` records every screened candidate
and `test.parquet` contains the final cohort.
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
