#!/usr/bin/env bash
set -euo pipefail
metric_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$metric_repo"
metric_stage="${1:?stage required}"
metric_output="${2:?fresh output directory required}"
shift 2
export PYTHONNOUSERSITE=1
export PYTHONPATH="$metric_repo/src"
source "$metric_repo/scripts/metric_runtime.sh"
semtx_metric_runtime "$metric_repo"
metric_motion_python="$metric_python"
case "$metric_stage" in
  prepare|truth_audit)
    exec "$metric_motion_python" -m semantic_transmission.metric_v3_validation "$metric_stage" --output "$metric_output" "$@" ;;
  extractor_gate)
    "$metric_object_python" -m semantic_transmission.metric_v3_validation extractor_audit --part event --output "$metric_output"
    "$metric_object_python" -m semantic_transmission.metric_v3_validation extractor_audit --part object --output "$metric_output"
    exec "$metric_motion_python" -m semantic_transmission.metric_v3_validation gate_summary --output "$metric_output" ;;
  summarize)
    exec "$metric_motion_python" scripts/report_metric_v3.py --output "$metric_output" ;;
  event|object|attribution)
    "$metric_motion_python" -c 'import sys; from pathlib import Path; from semantic_transmission.metric_v3_validation import require_gates; require_gates(Path(sys.argv[1]))' "$metric_output"
    echo 'Post-gate experiment runner requires implementation after the pre-score gate; no scores written.' >&2
    exit 2 ;;
  *) echo "unknown stage: $metric_stage" >&2; exit 2 ;;
esac
