#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONPATH="$repo/src:$repo/.local/vendor/Open-Sora" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
runtime_python=$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["python"])')
if [[ $# -eq 0 ]]; then set -- check; fi
exec "$runtime_python" -m semantic_transmission.terminal_gpu_progress \
  --log-dir .local/terminal_logs/bsu_adapter_training_7h --stop-grace-seconds 0 -- \
  "$runtime_python" -m semantic_transmission.bsu_adapter_timed_training "$@"
