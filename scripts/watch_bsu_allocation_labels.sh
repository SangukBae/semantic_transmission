#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset PYTHONPATH LD_LIBRARY_PATH
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
runtime=$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["python"])')
exec "$runtime" "$repo/scripts/watch_bsu_allocation_labels.py" "$@"
