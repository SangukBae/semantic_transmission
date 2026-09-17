#!/usr/bin/env bash
# Local, reproducible entry point. Does not install or alter system drivers.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
stage="${1:?Usage: bash scripts/run_metric_v2.sh prepare|motion|object|summarize OUTPUT}"
output_dir="${2:?output directory is required}"
source "$repo_dir/scripts/metric_runtime.sh"
semtx_metric_runtime "$repo_dir"
case "$stage" in
  object) metric_python="$metric_object_python" ;;
  prepare|motion|summarize) ;;
  *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac
export PYTHONNOUSERSITE=1
export PYTHONPATH="$repo_dir/src"
if test "$stage" = prepare; then
  "$metric_python" -m semantic_transmission.metric_v2_validation prepare --output "$output_dir"
  exec "$metric_python" scripts/freeze_metric_v2_extensions.py --output "$output_dir"
else
  "$metric_python" scripts/verify_metric_v2_run.py --output "$output_dir"
fi
exec "$metric_python" -m semantic_transmission.metric_v2_validation "$stage" --output "$output_dir"
