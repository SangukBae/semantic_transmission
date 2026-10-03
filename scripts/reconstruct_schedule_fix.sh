#!/usr/bin/env bash
# Paired full-video schedule repair; preserve completed reference reconstructions.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/src"
core_python=$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["python"])')
exec "$core_python" -m semantic_transmission.schedule_repair "$@"
