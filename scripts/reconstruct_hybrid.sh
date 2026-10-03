#!/usr/bin/env bash
# Reuse the completed hybrid keyframes, then caption -> transmit -> reconstruct -> evaluate.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export PYTHONPATH="$repo/src"
core_python=$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["python"])')
exec "$core_python" -m semantic_transmission.hybrid_reconstruction "$@"
