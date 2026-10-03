#!/usr/bin/env bash
# Reuse v2 inputs, tail17 and cached T5; record independently fixed inference noise.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$repo/scripts/reconstruct.sh" --precision-run "$@"
