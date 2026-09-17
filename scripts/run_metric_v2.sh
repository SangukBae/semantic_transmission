#!/usr/bin/env bash
# Local, reproducible entry point. Does not install or alter system drivers.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
stage="${1:?Usage: bash scripts/run_metric_v2.sh prepare|motion|object|summarize OUTPUT}"
output_dir="${2:?output directory is required}"
case "$stage" in
  object) metric_python="$repo_dir/.local/metric_v2_env/bin/python" ;;
  prepare|motion|summarize) metric_python="/home/sangukbae/anaconda3/envs/lgvsc/bin/python" ;;
  *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac
export PYTHONNOUSERSITE=1
export PYTHONPATH="$repo_dir/src"
unset LD_LIBRARY_PATH
metric_driver_dir="$repo_dir/.local/metric_v2_driver/extracted/usr/lib/x86_64-linux-gnu"
if grep -q '580.173.02' /proc/driver/nvidia/version 2>/dev/null && test -f "$metric_driver_dir/libcuda.so.580.173.02"; then
  export LD_LIBRARY_PATH="$metric_driver_dir"
fi
if test "$stage" = prepare; then
  "$metric_python" -m semantic_transmission.metric_v2_validation prepare --output "$output_dir"
  exec "$metric_python" scripts/freeze_metric_v2_extensions.py --output "$output_dir"
else
  "$metric_python" scripts/verify_metric_v2_run.py --output "$output_dir"
fi
exec "$metric_python" -m semantic_transmission.metric_v2_validation "$stage" --output "$output_dir"
