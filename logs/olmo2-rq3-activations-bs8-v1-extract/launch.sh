#!/usr/bin/env bash
set -uo pipefail
cd /workspace/affect-geometry
export HF_HOME=/workspace/affect-geometry/outputs/rq3/hf_cache
export PYTHONPATH=/workspace/affect-geometry/outputs/rq3/olmo2/memory_test/runtime:/workspace/affect-geometry/src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
printf 'Started: %s\n' "$(date -u --iso-8601=seconds)"
nice -n 10 /root/miniconda/envs/affect-geometry/bin/python -u outputs/rq3/olmo2/rq3-activations-bs8-v1/run_extraction.py 2>&1 | tee -a logs/olmo2-rq3-activations-bs8-v1-extract/console.log
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > logs/olmo2-rq3-activations-bs8-v1-extract/exit_code
printf 'Finished: %s | exit code: %s\n' "$(date -u --iso-8601=seconds)" "$status" | tee -a logs/olmo2-rq3-activations-bs8-v1-extract/console.log
exit "$status"
