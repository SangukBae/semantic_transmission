#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONPATH="$repo/src:$repo/.local/qwen35_training_probe_deps:$repo/scripts"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTORCH_ALLOC_CONF=expandable_segments:True
case "${1:-}" in
  freeze-reference|packet|score)
    exec .local/qwen35_env/bin/python scripts/evaluate_qwen35_astra_transfer.py "$@" ;;
  audit)
    exec .local/qwen35_env/bin/python scripts/audit_qwen35_astra_transfer.py ;;
  analyze)
    exec .local/qwen35_env/bin/python scripts/analyze_qwen35_astra_transfer.py ;;
  *) exec .local/qwen35_env/bin/python -m semantic_transmission.qwen35_astra_transfer "$@" ;;
esac
