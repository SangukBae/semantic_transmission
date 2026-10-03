#!/usr/bin/env bash
# Current ETRI default: hybrid keys + v2 captions + tail17 + prepared GPU T5.
# Prepared input: tv_low_08. Keep the hash-pinned experiment runners unchanged.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/src"
core_python=$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["python"])')
exec "$core_python" -m semantic_transmission.reconstruction_console "$@"
