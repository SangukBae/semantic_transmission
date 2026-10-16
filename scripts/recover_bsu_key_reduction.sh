#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
runtime_python=$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["python"])')
exec "$runtime_python" -m semantic_transmission.terminal_gpu_progress \
    --log-dir .local/terminal_logs/bsu_key_reduction_recovery -- \
    "$runtime_python" -m semantic_transmission.bsu_key_reduction_recovery "$@"
