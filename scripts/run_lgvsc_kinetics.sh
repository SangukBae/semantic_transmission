#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
case "${1:-}" in
  start|watch)
    exec bash "$repo/scripts/lgvsc_kinetics_session.sh" "$@"
    ;;
esac
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
ulimit -n 8192
python=$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["python"])')
if [[ "${1:-}" == status ]]; then
  exec "$python" -m semantic_transmission.lgvsc_kinetics "$@"
fi
exec "$python" -m semantic_transmission.terminal_gpu_progress --log-dir .local/terminal_logs/lgvsc_kinetics -- \
  "$python" -m semantic_transmission.lgvsc_kinetics "$@"
