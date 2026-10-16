#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
runtime_python="$repo/.local/dcvc_rt_env/bin/python"
exec "$runtime_python" -m semantic_transmission.terminal_gpu_progress \
  --log-dir .local/terminal_logs/bsu_dcvc_keys -- \
  "$runtime_python" -m semantic_transmission.bsu_dcvc_keys "$@"
