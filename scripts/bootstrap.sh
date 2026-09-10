#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
export PYTHONNOUSERSITE=1
unset PYTHONPATH LD_LIBRARY_PATH
conda_base=${CONDA_BASE:-$HOME/anaconda3}
conda_bin="$conda_base/bin/conda"
if [[ ! -x "$conda_bin" ]]; then
    echo "Set CONDA_BASE to an existing Conda installation." >&2
    exit 1
fi
command -v ffmpeg >/dev/null || { echo "Install ffmpeg before bootstrapping." >&2; exit 1; }
mkdir -p .local/logs
for name in lgvsc lgvsc-channel; do
    if [[ ! -x "$conda_base/envs/$name/bin/python" ]]; then
        "$conda_bin" create -y -n "$name" python=3.10 pip --override-channels -c defaults
    fi
done
core="$conda_base/envs/lgvsc/bin/python"
channel="$conda_base/envs/lgvsc-channel/bin/python"
"$core" -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
"$core" -m pip install -r environment/requirements-local.txt -c environment/locks/lgvsc-linux-py310.txt
"$core" scripts/repair_decord_wheel.py
"$channel" -m pip install -r environment/requirements-channel.txt -c environment/locks/lgvsc-channel-linux-py310.txt
"$core" -m pip install -e . --no-deps
"$channel" -m pip install -e . --no-deps
"$core" scripts/setup_dependencies.py
"$core" - "$core" "$channel" <<'PY'
import json,sys
from pathlib import Path
Path('.local/settings.json').write_text(json.dumps({'python':sys.argv[1],'channel_python':sys.argv[2]},indent=2)+'\n')
PY
"$core" scripts/download_models.py
"$core" scripts/download_auxiliary.py
"$core" -m semantic_transmission.cli doctor
"$core" scripts/probe_environment.py
"$core" -m pip check
"$channel" -m pip check
echo "Ready. Run: source scripts/activate.sh"
