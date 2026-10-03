#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
profile="${1:-detail}"
if [[ $# -gt 0 ]]; then shift; fi
case "$profile" in
    detail) options=(--quantization nf4 --frames 16) ;;
    int8) options=(--quantization int8 --frames 8) ;;
    max) options=(--quantization int8 --frames 16) ;;
    *) echo 'Usage: bash scripts/run_qwen35_advanced.sh {detail|int8|max} --video ... --output ...' >&2; exit 2 ;;
esac
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTORCH_ALLOC_CONF=expandable_segments:True
exec .local/qwen35_env/bin/python scripts/qwen35_faithful_caption.py --engine advanced \
    "${options[@]}" --max-pixels 262144 --max-new-tokens 256 "$@"
