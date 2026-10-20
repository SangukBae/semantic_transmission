#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset PYTHONPATH LD_LIBRARY_PATH PYTORCH_CUDA_ALLOC_CONF
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
case "${1:-}" in
  status|watch|--help|-h) exec python3 -m semantic_transmission.bsu_metadata_e2e "$@" ;;
esac
runtime=$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["python"])')
if [[ "${1:-}" == run ]]; then
  metadata_output="$repo/outputs/bsu_metadata_e2e_20261019_v2"
  metadata_args=("$@")
  for ((metadata_i=0; metadata_i<${#metadata_args[@]}; metadata_i++)); do
    if [[ "${metadata_args[$metadata_i]}" == --output ]]; then
      metadata_output="${metadata_args[$((metadata_i+1))]:?--output requires a path}"
    elif [[ "${metadata_args[$metadata_i]}" == --output=* ]]; then
      metadata_output="${metadata_args[$metadata_i]#--output=}"
    fi
  done
  mkdir -p "$metadata_output"
  "$runtime" -m semantic_transmission.bsu_metadata_e2e "$@" 2>&1 | tee -a "$metadata_output/run.log"
  exit ${PIPESTATUS[0]}
fi
exec "$runtime" -m semantic_transmission.bsu_metadata_e2e "$@"
