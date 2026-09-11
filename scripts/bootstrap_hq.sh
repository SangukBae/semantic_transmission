#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
bash scripts/bootstrap.sh
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONNOUSERSITE=1
core="${CONDA_BASE:-$HOME/anaconda3}/envs/lgvsc/bin/python"
"$core" -m pip install --no-deps 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1%2Bcu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
"$core" -m pip check
"$core" - <<'PY'
import torch
from flash_attn import flash_attn_func
q = torch.randn(1, 64, 8, 64, device="cuda", dtype=torch.bfloat16)
y = flash_attn_func(q, q, q, causal=True)
torch.cuda.synchronize()
assert torch.isfinite(y).all()
print("FlashAttention CUDA probe passed")
PY
