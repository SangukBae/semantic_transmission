#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
case "${1:-}" in
  watch|status|--help|-h) exec python3 -m semantic_transmission.sparc_framer "$@" ;;
esac
exec "$repo/.local/bsu_codec_eval_env/bin/python" -m semantic_transmission.sparc_framer "$@"
