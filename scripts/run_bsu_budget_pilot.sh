#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset PYTHONPATH LD_LIBRARY_PATH
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
runtime=$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["python"])')
if [[ "${1:-}" == run ]]; then
  budget_output="$repo/outputs/bsu_key_budget_pilot_20261019"
  budget_args=("$@")
  for ((budget_i=0; budget_i<${#budget_args[@]}; budget_i++)); do
    if [[ "${budget_args[$budget_i]}" == --output ]]; then
      budget_output="${budget_args[$((budget_i+1))]:?--output requires a path}"
    elif [[ "${budget_args[$budget_i]}" == --output=* ]]; then
      budget_output="${budget_args[$budget_i]#--output=}"
    fi
  done
  mkdir -p "$budget_output"
  "$runtime" -m semantic_transmission.bsu_budget_pilot "$@" 2>&1 | tee -a "$budget_output/run.log"
  exit ${PIPESTATUS[0]}
fi
exec "$runtime" -m semantic_transmission.bsu_budget_pilot "$@"
