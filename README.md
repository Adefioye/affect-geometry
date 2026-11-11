# Affect Geometry

A reproducible pipeline for extracting every-layer Qwen residual-stream activations from
single-label GoEmotions examples, constructing representational dissimilarity matrices (RDMs),
and comparing MDS and Isomap affect manifolds.

See [the EXP1 implementation guide](docs/exp1-implementation.md) for installation, commands, outputs, and
the mathematical conventions used by the implementation.

## Setup

On a Linux x86-64 machine, bootstrap Miniconda and the complete development environment with one
command:

```bash
./bootstrap_dev_env.sh
source ~/.bashrc
conda activate affect-geometry
```

The bootstrap is idempotent: it reuses an existing Conda installation, otherwise runs
`install_miniconda.sh`, makes Conda immediately available to the setup process, and then runs
`setup_dev_env.sh`. The latter creates the `affect-geometry` environment, installs this project
from `pyproject.toml`, and runs its validation checks.

The two steps can still be run separately when needed:

```bash
./install_miniconda.sh
source ~/.bashrc
./setup_dev_env.sh
conda activate affect-geometry
```

For a specific CUDA build of PyTorch, provide its wheel index when setting up the environment:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 ./bootstrap_dev_env.sh
```

Run the complete experiment with:

```bash
python -m run --config configs/default.json
```

The complete-pipeline command generates one compact UTC run ID, such as
`20261018T153045Z`. All generated artifacts are written below:

```text
outputs/<model-slug>/<run-id>/
```

For example, the default model writes to
`outputs/qwen--qwen3-0.6b/20261018T153045Z/`. To execute stages separately, generate one run ID
and pass that same value to every stage:

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
python -m prepare --config configs/default.json --run-id "$RUN_ID"
python -m extract --config configs/default.json --run-id "$RUN_ID"
python -m rdm --config configs/default.json --run-id "$RUN_ID"
python -m embed --config configs/default.json --run-id "$RUN_ID"
python -m analyze --config configs/default.json --run-id "$RUN_ID"
python -m plot --config configs/default.json --run-id "$RUN_ID"
```

You can also replace `"auto"` in the config with one explicit run ID. A shared explicit ID is
required for separate-stage commands so they all read and write the same run.

Every command shows descriptive `tqdm` progress and appends structured events to one centralized
file identified by model and run ID, for example:

```text
logs/qwen--qwen3-0.6b__20261018T153045Z/run.log
```

## EXP2: independent emotion classification

The independent classification runner evaluates all single-label GoEmotions examples, including
neutral, for LLaMA-2-7B, Gemma-2-9B, Apertus-8B-Instruct-2509, and OLMo-2-1124-7B-Instruct.
It saves raw responses and strict correctness, with Choi–Weber first-word scoring only as a
complementary diagnostic. No Choi–Weber scripts are imported or executed.

See [classification commands and outputs](experiments/exp2-classification.md) and
[the EXP2 protocol](docs/exp2.md). Begin with a training-only pilot before a full GPU run.

### EXP3: correctness-conditioned affective geometry

The EXP3 pipeline adds balanced correctness cohorts, verified classification-prompt activations,
strict MDS/Isomap and Diffusion Maps, nested held-out-rating evaluation, NRC-row permutation
families, centroid topology, selected-layer CKA, and reproducible CSV-based figures.
Start with the [implementation and run guide](experiments/exp3-implementation.md).
Use [extract_exp3_activations.py](script/extract_exp3_activations.py) with
[the extraction config](configs/exp3/extract_activations.json) to publish raw and neutral-projected
sample states for all three cohorts/models. Then run [run_exp3_geometry.py](script/run_exp3_geometry.py)
with [Gemma2](configs/exp3/gemma2.json), [Apertus](configs/exp3/apertus.json), or
[OLMo2](configs/exp3/olmo2.json). Each model's activations and all analysis artifacts, including
`plot_data`, share its existing `{username}/rq3-<model>` dataset. The analysis configs select all
layers and 2,000 permutations. Use
[upload_exp3_analysis.py](script/upload_exp3_analysis.py) to upload a completed run independently.

## EXP4: valence-direction transfer

See [EXP4](docs/exp4.md) for centroid-PC1 extraction from saved EXP3 last-token neutral-projected
activations and frozen directional patching on SST, IMDb, DynaSent R1 and R2.
The standalone configuration is `configs/valence_transfer.json`.
