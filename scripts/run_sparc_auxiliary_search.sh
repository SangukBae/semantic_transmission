#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
case "${1:-}" in
  status|watch) exec python3 -m semantic_transmission.sparc_auxiliary_search "$@" ;;
esac
runtime="$repo/.local/bsu_codec_eval_env/bin/python"
exec "$runtime" -m semantic_transmission.terminal_gpu_progress --stop-grace-seconds 0 \
  --log-dir .local/terminal_logs/sparc_auxiliary -- \
  "$runtime" -m semantic_transmission.sparc_auxiliary_search "$@"
