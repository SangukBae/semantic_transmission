#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
runtime_python="$repo/.local/unified_bsu_env/bin/python"
if [[ ! -x "$runtime_python" ]]; then
  printf '%s\n' 'Missing .local/unified_bsu_env. See docs/RUN_GUIDE.md#bsu-unified.' >&2
  exit 1
fi
if [[ "${1:-status}" == status || "${1:-}" == plan ]]; then
  exec "$runtime_python" -m semantic_transmission.unified_bsu_training_quality "$@"
fi
exec "$runtime_python" -m semantic_transmission.terminal_gpu_progress \
  --stop-grace-seconds 0 --log-dir .local/terminal_logs/bsu_unified_etri -- \
  "$runtime_python" -m semantic_transmission.unified_bsu_training_quality "$@"
