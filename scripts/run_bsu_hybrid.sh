#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
runtime_python="$repo/.local/bsu_codec_eval_env/bin/python"
case "${1:-}" in
  profile|status|--help|-h) exec python3 -m semantic_transmission.bsu "$@" ;;
esac
if [[ ! -x "$runtime_python" ]]; then
  printf '%s\n' 'BSU runtime missing: .local/bsu_codec_eval_env/bin/python (see docs/RUN_GUIDE.md)' >&2
  exit 1
fi
exec "$runtime_python" -m semantic_transmission.terminal_gpu_progress \
  --log-dir .local/terminal_logs/bsu -- \
  "$runtime_python" -m semantic_transmission.bsu "$@"
