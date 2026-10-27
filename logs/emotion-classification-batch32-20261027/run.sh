#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace/affect-geometry
export PATH="/root/miniconda/envs/affect-geometry/bin:$PATH"
export PYTHONUNBUFFERED=1
trap 'code=$?; echo "STOPPED: exit $code at line $LINENO. Later models will not start."; exit "$code"' ERR
for model in gemma2 llama2 apertus olmo2; do
  echo "$(date -Is) START pilot, batch size 32: $model"
  python -m affect_geometry.classification run --config "configs/classification/$model.json" --pilot-limit 64 --run-dir "outputs/rq2_classification/$model/pilot"
  echo "$(date -Is) START full, batch size 32: $model"
  python -m affect_geometry.classification run --config "configs/classification/$model.json"
  echo "$(date -Is) DONE: $model"
done
echo "$(date -Is) ALL FOUR MODELS COMPLETE"
