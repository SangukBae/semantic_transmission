#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2
runtime_python=$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["channel_python"])')
exec "$runtime_python" -m semantic_transmission.terminal_gpu_progress \
  --log-dir .local/terminal_logs/bsu_lossless_transport -- \
  "$runtime_python" -m semantic_transmission.bsu_lossless_validation --channel "$@"
