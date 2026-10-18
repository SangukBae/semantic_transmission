#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset PYTHONPATH LD_LIBRARY_PATH
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
runtime="$repo/.local/envs/bsu-ltx25/bin/python"
exec "$runtime" -m semantic_transmission.terminal_gpu_progress \
  --log-dir .local/terminal_logs/bsu_ltx_receiver -- \
  "$runtime" -m semantic_transmission.ltx_receiver \
  --model-dir "$repo/.local/checkpoints/ltx25" --vendor "$repo/.local/vendor/LTX-2" "$@"
