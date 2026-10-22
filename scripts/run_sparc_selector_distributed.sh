#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
exec python3 scripts/watch_sparc_selector_distributed.py "${1:-watch}" \
  --config "${SPARC_DISTRIBUTED_CONFIG:-$repo/.local/selector_distributed_20261021/config.json}"
