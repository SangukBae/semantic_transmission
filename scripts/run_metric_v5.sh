#!/usr/bin/env bash
set -euo pipefail
metric_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$metric_repo"
metric_phase="${1:?phase required}"
metric_output="${2:?output required}"
export PYTHONNOUSERSITE=1 PYTHONPATH="$metric_repo/src"
unset LD_LIBRARY_PATH
metric_python=/home/sangukbae/anaconda3/envs/lgvsc/bin/python
if [[ "$metric_phase" == visual || "$metric_phase" == development ]]; then
  metric_python="$metric_repo/.local/metric_v2_env/bin/python"
fi
if [[ "$metric_phase" == visual || "$metric_phase" == pixel || "$metric_phase" == development ]]; then
  export LD_LIBRARY_PATH="$metric_repo/.local/metric_v2_driver/extracted/usr/lib/x86_64-linux-gnu"
fi
if [[ "$metric_phase" == declare ]]; then
  exec "$metric_python" scripts/report_metric_v5.py --output "$metric_output" --declare
elif [[ "$metric_phase" == report ]]; then
  exec "$metric_python" scripts/report_metric_v5.py --output "$metric_output"
fi
exec "$metric_python" -m semantic_transmission.metric_v5_pipeline "$metric_phase" --output "$metric_output"
