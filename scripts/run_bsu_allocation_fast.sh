#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset PYTHONPATH LD_LIBRARY_PATH
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
runtime=$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["python"])')
if [[ "${1:-}" == run ]]; then
  label_output="$repo/outputs/bsu_allocation_labels_600_20261019"
  label_args=("$@")
  for ((label_i=0; label_i<${#label_args[@]}; label_i++)); do
    if [[ "${label_args[$label_i]}" == --output ]]; then
      label_output="${label_args[$((label_i+1))]:?--output requires a path}"
    elif [[ "${label_args[$label_i]}" == --output=* ]]; then
      label_output="${label_args[$label_i]#--output=}"
    fi
  done
  "$runtime" -m semantic_transmission.bsu_allocation_fast "$@" 2>&1 | tee -a "$label_output/run.log"
  exit ${PIPESTATUS[0]}
fi
exec "$runtime" -m semantic_transmission.bsu_allocation_fast "$@"
