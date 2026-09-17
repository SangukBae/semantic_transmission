#!/usr/bin/env bash
set -euo pipefail
metric_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$metric_repo"
metric_stage="${1:?stage required}"
metric_output="${2:?output required}"
shift 2
export PYTHONNOUSERSITE=1 PYTHONPATH="$metric_repo/src"
unset LD_LIBRARY_PATH
metric_driver="$metric_repo/.local/metric_v2_driver/extracted/usr/lib/x86_64-linux-gnu"
if grep -q '580.173.02' /proc/driver/nvidia/version && test -f "$metric_driver/libcuda.so.580.173.02"; then
  export LD_LIBRARY_PATH="$metric_driver"
fi
metric_python=/home/sangukbae/anaconda3/envs/lgvsc/bin/python
case "$metric_stage" in
  visual-development|visual-heldout)
    exec "$metric_repo/.local/metric_v2_env/bin/python" -m semantic_transmission.metric_v3_formal run --output "$metric_output" --part visual --split "${metric_stage#visual-}" ;;
  pixel-development|pixel-heldout)
    exec "$metric_python" -m semantic_transmission.metric_v3_formal run --output "$metric_output" --part pixel --split "${metric_stage#pixel-}" ;;
  prepare|calibrate|verify)
    exec "$metric_python" -m semantic_transmission.metric_v3_formal "$metric_stage" --output "$metric_output" "$@" ;;
  summarize)
    "$metric_python" -m semantic_transmission.metric_v3_formal summarize --output "$metric_output"
    exec "$metric_python" scripts/report_metric_v3_formal.py --output "$metric_output" ;;
  *) echo "unknown stage: $metric_stage" >&2; exit 2 ;;
esac
