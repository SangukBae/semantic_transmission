#!/usr/bin/env bash
set -euo pipefail
metric_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$metric_repo"
metric_phase="${1:?phase required}"
metric_output="${2:?output required}"
export PYTHONNOUSERSITE=1 PYTHONPATH="$metric_repo/src"
source "$metric_repo/scripts/metric_runtime.sh"
semtx_metric_runtime "$metric_repo"
if [[ "$metric_phase" == visual || "$metric_phase" == development ]]; then
  metric_python="$metric_object_python"
fi
if [[ "$metric_phase" == declare ]]; then
  exec "$metric_python" scripts/report_metric_v5.py --output "$metric_output" --declare
elif [[ "$metric_phase" == report ]]; then
  exec "$metric_python" scripts/report_metric_v5.py --output "$metric_output"
fi
exec "$metric_python" -m semantic_transmission.metric_v5_pipeline "$metric_phase" --output "$metric_output"
