#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONPATH="$PWD/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec .local/fc_auto_env/bin/python -m semantic_transmission.auto_extraction "$@"
