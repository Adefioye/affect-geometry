#!/usr/bin/env bash
set -uo pipefail
cd /workspace/affect-geometry
export PATH=/root/miniconda/envs/affect-geometry/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HOME=/workspace/affect-geometry/outputs/exp3/hf_cache
printf 'Started descriptive analysis: %s\n' "$(date -u --iso-8601=seconds)"
python -u script/run_exp3_geometry.py run --config outputs/exp3/olmo2/exp3-activations-bs8-v1/analysis_config.json --skip-permutations --no-upload
run_status=$?
printf '%s\n' "$run_status" > logs/olmo2-exp3-analysis-full-bs8-v1-descriptive-analysis/exit_code
printf 'Finished: %s | exit code: %s\n' "$(date -u --iso-8601=seconds)" "$run_status"
exit "$run_status"
