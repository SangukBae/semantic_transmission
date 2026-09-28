#!/usr/bin/env bash
# Recover the verified tv_low_08 selection after the 2026-09-26 caption failure.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$repo/scripts/run_etri_60s.sh" \
  --reuse-selection-from "$repo/outputs/etri_60s_tv_low_08_263507d45874" "$@"
