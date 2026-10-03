#!/usr/bin/env bash
# Isolated environment; existing LGVSC/InternVL environments are not modified.
set -euo pipefail
cd "$(dirname "$0")/.."
model_id="${1:-Qwen/Qwen3-VL-4B-Instruct}"
revision="${2:-ebb281ec70b05090aa6165b016eac8ec08e71b17}"
if [[ ! -x .local/fc_auto_env/bin/python ]]; then
    uv venv --python /usr/bin/python3 .local/fc_auto_env
fi
uv pip install --python .local/fc_auto_env/bin/python \
    'torch==2.12.0' 'torchvision==0.27.0' 'transformers==4.57.6' \
    'accelerate==1.12.0' 'bitsandbytes==0.48.2' 'Pillow>=10' \
    'opencv-python-headless<4.12' 'numpy<2' pytest
uv pip freeze --python .local/fc_auto_env/bin/python > .local/fc_auto_environment.lock.txt
env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE .local/fc_auto_env/bin/python - "$model_id" "$revision" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2], max_workers=4))
PY
