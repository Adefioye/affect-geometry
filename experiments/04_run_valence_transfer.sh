#!/usr/bin/env bash

set -eo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate affect-geometry

# The Hub calls load HF_TOKEN from the project .env when it is unset.
unset HF_TOKEN
export HF_HOME="$PWD/outputs/valence-transfer/huggingface-cache"
export PYTHONUNBUFFERED=1
python -c 'from affect_geometry.classification.environment import load_hf_token; import sys; sys.exit(None if load_hf_token() else "Set HF_TOKEN in the project .env")'

config_path="${1:-configs/valence_transfer.json}"
for stage in select evaluate; do
    printf '\nEXP4 stage %s\n' "$stage"
    python experiments/04_run_valence_transfer.py "$stage" --config "$config_path"
done

printf '\nEXP4 complete\n'
