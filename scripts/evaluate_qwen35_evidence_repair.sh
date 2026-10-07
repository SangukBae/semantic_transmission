#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
qwen_eval_nofile=$(ulimit -Sn)
if [[ "$qwen_eval_nofile" != unlimited ]] && (( qwen_eval_nofile < 8192 )); then
    ulimit -Sn 8192
fi
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONPATH="$repo/src:$repo/scripts:$repo/.local/qwen35_training_probe_deps"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_ALLOC_CONF=expandable_segments:True
case "${1:-}" in
    packet|score) exec .local/qwen35_env/bin/python scripts/evaluate_qwen35_evidence_repair.py "$@" ;;
    *) exec .local/qwen35_env/bin/python -m semantic_transmission.qwen35_evidence_repair_eval "$@" ;;
esac
