#!/usr/bin/env bash
# Only selects/exports frames. Never launches the video decoder.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$repo/src:$repo/.local/vendor/InternVL"
selector_python=$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["internvl_python"])')
exec "$selector_python" -m semantic_transmission.hybrid_selection select "$@"
