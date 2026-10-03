#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .local/qwen35_env/bin/python ]]; then
    echo 'Run bash scripts/bootstrap_qwen35.sh first.' >&2
    exit 1
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTORCH_ALLOC_CONF=expandable_segments:True
exec .local/qwen35_env/bin/python scripts/qwen35_faithful_caption.py --engine basic "$@"
