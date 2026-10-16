#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
runtime_python=$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["python"])')
case "${1:-all}" in
  all) module=semantic_transmission.bsu_validation_pipeline; if (($#)); then shift; fi ;;
  budget|tracking|adapter|adapter_native)
    stage=$1; shift
    module="semantic_transmission.bsu_${stage}_validation"
    if [[ "$stage" == adapter_native ]]; then module=semantic_transmission.bsu_adapter_native; fi
    set -- run "$@" ;;
  audit|verify|report)
    stage=$1; shift; module=semantic_transmission.bsu_motion_report
    if [[ "$stage" == report ]]; then stage=summarize; fi
    set -- "$stage" "$@" ;;
  *) module=semantic_transmission.bsu_motion_validation ;;
esac
exec "$runtime_python" -m semantic_transmission.terminal_gpu_progress \
  --log-dir .local/terminal_logs/bsu_motion_validation -- \
  "$runtime_python" -m "$module" "$@"
