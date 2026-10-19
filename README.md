# Affect Geometry

A reproducible pipeline for extracting every-layer Qwen residual-stream activations from
single-label GoEmotions examples, constructing representational dissimilarity matrices (RDMs),
and comparing MDS and Isomap affect manifolds.

See [`doc/rq1-implementation.md`](doc/rq1-implementation.md) for installation, commands, outputs, and
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
