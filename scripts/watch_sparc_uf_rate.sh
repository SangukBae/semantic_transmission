#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
exec python3 -m semantic_transmission.sparc_uf_rate_watch "$@"
