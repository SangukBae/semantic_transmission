#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONNOUSERSITE=1
base="${CONDA_BASE:-$HOME/anaconda3}"
selector="$base/envs/lgvsc-internvl/bin/python"
if [[ ! -x "$selector" ]]; then
  "$base/bin/conda" create -y -n lgvsc-internvl python=3.9.19 pip
fi
"$selector" -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
"$selector" -m pip install -r environment/requirements-internvl-release.txt
"$selector" -m pip install --no-deps 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3%2Bcu123torch2.4cxx11abiFALSE-cp39-cp39-linux_x86_64.whl'
"$selector" -m pip check
"$base/envs/lgvsc/bin/python" - "$selector" <<'PY'
import json, sys
from pathlib import Path
from semantic_transmission.artifacts import write_json
path = Path('.local/settings.json')
config = json.loads(path.read_text())
config['internvl_python'] = sys.argv[1]
write_json(path, config)
PY
