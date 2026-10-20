#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset LD_LIBRARY_PATH PYTORCH_CUDA_ALLOC_CONF PYTHONPATH
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
python=$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["python"])')
exec "$python" -m semantic_transmission.terminal_gpu_progress \
  --log-dir .local/terminal_logs/ntscc_plus -- \
  "$python" -m semantic_transmission.ntscc_plus_validation "$@"
