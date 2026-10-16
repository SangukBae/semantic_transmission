#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTORCH_ALLOC_CONF=expandable_segments:True
# The pinned decoder uses PyTorch 2.2, which reads the CUDA-specific name.
export PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_ALLOC_CONF"
ulimit -n 8192
python=$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["python"])')
case "${1:-}" in
  run|check)
    exec "$python" -m semantic_transmission.terminal_progress -- \
      "$python" -m semantic_transmission.bsu_default "$@"
    ;;
  *) exec "$python" -m semantic_transmission.bsu_default "$@" ;;
esac
