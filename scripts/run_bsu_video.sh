#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case "${1:-}" in
  status|--help|-h) exec python3 -m semantic_transmission.bsu_video "$@" ;;
esac
runtime_python="$repo/.local/bsu_codec_eval_env/bin/python"
if [[ ! -x "$runtime_python" ]]; then
  printf '%s\n' 'BSU runtime missing. For the configured ETRI video use: bash scripts/run_bsu_etri01.sh' >&2
  exit 1
fi
exec "$runtime_python" -c 'from semantic_transmission.bsu_video import progress_main; progress_main()' \
  --log-dir .local/terminal_logs/bsu_video -- \
  "$runtime_python" -m semantic_transmission.bsu_video "$@"
