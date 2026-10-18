#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case "${1:-}" in
  profile|status|--help|-h) exec python3 -m semantic_transmission.bsu_gen "$@" ;;
esac
runtime_python="${BSU_GEN_PYTHON:-$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["python"])')}"
if [[ ! -x "$runtime_python" ]]; then
  printf '%s\n' 'BSU-Gen runtime missing; see docs/RUN_GUIDE.md#bsu-gen' >&2
  exit 1
fi
ulimit -n 8192
exec "$runtime_python" -m semantic_transmission.terminal_gpu_progress \
  --log-dir .local/terminal_logs/bsu_gen -- \
  "$runtime_python" -m semantic_transmission.bsu_gen "$@"
