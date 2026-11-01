#!/usr/bin/env bash
set -uo pipefail
cd /workspace/affect-geometry
export HF_HOME=/workspace/affect-geometry/outputs/rq3/hf_cache
export PYTHONPATH=/workspace/affect-geometry/outputs/rq3/olmo2/memory_test/runtime:/workspace/affect-geometry/src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
nice -n 10 /root/miniconda/envs/affect-geometry/bin/python -u outputs/rq3/olmo2/memory_test/probe.py 2>&1 | tee logs/olmo2-rq3-memory-test/console.log
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > logs/olmo2-rq3-memory-test/exit_code
exit "$status"
