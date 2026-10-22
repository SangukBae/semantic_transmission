#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
runtime="${SPARC_PYTHON:-$repo/.local/bsu_codec_eval_env/bin/python}"
exec "$runtime" -m semantic_transmission.terminal_gpu_progress \
  --log-dir .local/terminal_logs/sparc_uf_no_gap -- \
  "$runtime" -m semantic_transmission.sparc_uf_no_gap "$@"
