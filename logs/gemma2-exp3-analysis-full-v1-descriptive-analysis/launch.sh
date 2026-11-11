#!/usr/bin/env bash
set -uo pipefail
cd /workspace/affect-geometry
export PATH=/root/miniconda/envs/affect-geometry/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HOME=/workspace/affect-geometry/outputs/exp3/hf_cache
printf 'Started descriptive analysis: %s\n' "$(date -u --iso-8601=seconds)" | tee -a '/workspace/affect-geometry/logs/gemma2-exp3-analysis-full-v1-descriptive-analysis/console.log'
python script/run_exp3_geometry.py run --config configs/exp3/gemma2.json --skip-permutations --no-upload 2>&1 | tee -a '/workspace/affect-geometry/logs/gemma2-exp3-analysis-full-v1-descriptive-analysis/console.log'
run_status=${PIPESTATUS[0]}
printf '%s\n' "$run_status" > '/workspace/affect-geometry/logs/gemma2-exp3-analysis-full-v1-descriptive-analysis/exit_code'
printf 'Finished: %s | exit code: %s\n' "$(date -u --iso-8601=seconds)" "$run_status" | tee -a '/workspace/affect-geometry/logs/gemma2-exp3-analysis-full-v1-descriptive-analysis/console.log'
exit "$run_status"
