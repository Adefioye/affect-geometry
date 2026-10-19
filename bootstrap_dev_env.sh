#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MINICONDA_PREFIX="${MINICONDA_PREFIX:-$HOME/miniconda}"

if command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
  echo "ℹ️  Using existing Conda installation: $CONDA_BIN"
elif [[ -x "$MINICONDA_PREFIX/bin/conda" ]]; then
  CONDA_BIN="$MINICONDA_PREFIX/bin/conda"
  echo "ℹ️  Using existing Miniconda installation: $CONDA_BIN"
else
  if [[ "$MINICONDA_PREFIX" != "$HOME/miniconda" ]]; then
    echo "❌ install_miniconda.sh currently installs only to '$HOME/miniconda'."
    echo "   Unset MINICONDA_PREFIX or install Conda at the requested location first."
    exit 1
  fi

  echo "📦 Miniconda was not found; installing it first..."
  "$PROJECT_ROOT/install_miniconda.sh"
  CONDA_BIN="$MINICONDA_PREFIX/bin/conda"

  if [[ ! -x "$CONDA_BIN" ]]; then
    echo "❌ Miniconda installation completed without producing '$CONDA_BIN'."
    exit 1
  fi
fi

# conda init affects future shells. Adding its bin directory here makes Conda
# available immediately to setup_dev_env.sh in this bootstrap process.
export PATH="$(dirname -- "$CONDA_BIN"):$PATH"

echo "🛠️  Creating and validating the development environment..."
"$PROJECT_ROOT/setup_dev_env.sh"

echo
echo "✅ Bootstrap complete."
echo "   Restart the shell or run: source ~/.bashrc"
echo "   Activate with:           conda activate ${ENV_NAME:-affect-geometry}"
