#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="${ENV_NAME:-affect-geometry}"
PY_VER="${PY_VER:-3.11}"
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# --- sanity checks -----------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "❌ 'conda' was not found in PATH."
  echo "   Install Miniconda and restart the shell before running this script."
  exit 1
fi

# --- accept Anaconda channel terms ------------------------------------------
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# --- create environment if needed -------------------------------------------
if conda run -n "$ENV_NAME" python --version >/dev/null 2>&1; then
  echo "ℹ️  Conda environment '$ENV_NAME' already exists; skipping creation."
else
  echo "📦 Creating Conda environment '$ENV_NAME' with Python $PY_VER..."
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi

# --- install project dependencies -------------------------------------------
echo "⬆️  Updating packaging tools in '$ENV_NAME'..."
conda run -n "$ENV_NAME" \
  python -m pip install --upgrade pip setuptools wheel

# Set TORCH_INDEX_URL when a machine needs a specific CPU or CUDA wheel index.
# Example:
# TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 ./setup_dev_env.sh
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  echo "🔥 Installing PyTorch from '$TORCH_INDEX_URL'..."
  conda run -n "$ENV_NAME" \
    python -m pip install --index-url "$TORCH_INDEX_URL" "torch>=2.1"
fi

echo "📦 Installing affect-geometry with development dependencies..."
conda run -n "$ENV_NAME" \
  python -m pip install -e "${PROJECT_ROOT}[dev]"

# --- verify environment ------------------------------------------------------
echo "🔍 Checking dependency consistency..."
conda run -n "$ENV_NAME" python -m pip check

echo "🧪 Verifying required imports..."
conda run -n "$ENV_NAME" python -c \
  "import datasets, matplotlib, numpy, pandas, pyarrow, scipy, sklearn, torch, tqdm, transformers"

echo "🧪 Running the test suite..."
conda run -n "$ENV_NAME" \
  python -m pytest -q "$PROJECT_ROOT/tests"

# --- done -------------------------------------------------------------------
echo
echo "✅ Environment ready."
echo "   • Activate: conda activate $ENV_NAME"
echo "   • Python:   python --version"
echo "   • Run:      python -m run --config configs/default.json"
