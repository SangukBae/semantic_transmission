#!/usr/bin/env bash
# Prepared three-video hybrid/v2 inputs -> tail17 + cached CPU FP32 T5.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/src"
core_python=$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["python"])')
exec "$core_python" -m semantic_transmission.short_video_batch "$@"
