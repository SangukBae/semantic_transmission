#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_ALLOC_CONF=expandable_segments:True
bsu_python=${LGVSC_PYTHON:-/home/sangukbae/anaconda3/envs/lgvsc/bin/python}
if (( $# == 0 )); then
    "$bsu_python" -m semantic_transmission.bsu_motion_ablation --stage plan
    "$bsu_python" -m semantic_transmission.bsu_motion_ablation --stage generate
    "$bsu_python" -m semantic_transmission.bsu_motion_ablation --stage evaluate
    exec "$bsu_python" -m semantic_transmission.bsu_motion_context_evaluation
fi
exec "$bsu_python" -m semantic_transmission.bsu_motion_ablation "$@"
