#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
runtime_python="$repo/.local/bsu_codec_eval_env/bin/python"
exec "$runtime_python" -m semantic_transmission.terminal_gpu_progress \
  --log-dir .local/terminal_logs/bsu_dcvc_validation -- \
  "$runtime_python" -m semantic_transmission.bsu_dcvc_validation "$@"
