#!/usr/bin/env bash
set -euo pipefail
metric_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$metric_repo"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$metric_repo/src"
exec python3 -m semantic_transmission.metric_campaign "$@"
