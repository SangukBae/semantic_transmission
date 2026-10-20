#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset PYTHONPATH LD_LIBRARY_PATH
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
runtime=$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["python"])')
if [[ "${1:-}" == train ]]; then
  allocation_args=("$@")
  allocation_output=""
  for ((allocation_i=0; allocation_i<${#allocation_args[@]}; allocation_i++)); do
    if [[ "${allocation_args[$allocation_i]}" == --output ]]; then
      allocation_output="${allocation_args[$((allocation_i+1))]:?--output requires a path}"
    elif [[ "${allocation_args[$allocation_i]}" == --output=* ]]; then
      allocation_output="${allocation_args[$allocation_i]#--output=}"
    fi
  done
  if [[ -z "$allocation_output" ]]; then
    exec "$runtime" -m semantic_transmission.bsu_allocation_training "$@"
  fi
  mkdir -p "$allocation_output"
  "$runtime" -m semantic_transmission.bsu_allocation_training "$@" 2>&1 | tee -a "$allocation_output/run.log"
  exit ${PIPESTATUS[0]}
fi
exec "$runtime" -m semantic_transmission.bsu_allocation_training "$@"
