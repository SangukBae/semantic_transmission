#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$repo/.local/selector_distributed_20261021/config.json" ]]; then
  exec bash "$repo/scripts/run_sparc_selector_distributed.sh" "${1:-watch}"
fi
exec bash "$repo/scripts/run_sparc_selector_v2.sh" "${1:-watch}"
