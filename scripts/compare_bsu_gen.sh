#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset PYTHONPATH LD_LIBRARY_PATH
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
runtime=$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["python"])')
exec "$runtime" -m semantic_transmission.terminal_gpu_progress \
  --log-dir .local/terminal_logs/bsu_ltx -- \
  "$runtime" -m semantic_transmission.bsu_ltx_comparison "$@"
