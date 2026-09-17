#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
bash scripts/bootstrap_hq.sh
bash scripts/bootstrap_internvl.sh
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONNOUSERSITE=1
base="${CONDA_BASE:-$HOME/anaconda3}"
if [[ ! -x .local/cuda-build/bin/nvcc || ! -f .local/cuda-build/include/cuda_profiler_api.h ]]; then
  "$base/bin/conda" create -y -p "$repo/.local/cuda-build" -c nvidia/label/cuda-12.1.1 \
    cuda-nvcc=12.1.105 cuda-cudart-dev=12.1.105 cuda-cccl=12.1.109 cuda-profiler-api=12.1.105
fi
"$base/envs/lgvsc/bin/python" scripts/install_apex.py
"$base/envs/lgvsc/bin/python" -m pip install --no-deps -r requirements-evaluation.txt
"$base/envs/lgvsc/bin/python" -m pip check
