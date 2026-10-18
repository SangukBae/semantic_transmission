#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset PYTHONPATH LD_LIBRARY_PATH HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
base_python=$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["python"])')
runtime="$repo/.local/envs/bsu-ltx25"
vendor="$repo/.local/vendor/LTX-2"
revision=$(python3 -c 'import json; print(json.load(open("configs/bsu_gen_ltx25.json"))["upstream_revision"])')
if [[ ! -e "$vendor" ]]; then
  git clone https://github.com/Lightricks/LTX-2.git "$vendor"
  git -C "$vendor" checkout --detach "$revision"
fi
[[ $(git -C "$vendor" rev-parse HEAD) == "$revision" ]]
if [[ ! -x "$runtime/bin/python" ]]; then
  "$base_python" -m venv "$runtime"
fi
"$runtime/bin/python" -m pip install --disable-pip-version-check \
  torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
"$runtime/bin/python" -m pip install --disable-pip-version-check \
  transformers==5.14.1 'huggingface_hub>=0.34' opencv-python-headless \
  -e "$vendor/packages/ltx-core" -e "$vendor/packages/ltx-pipelines"
"$runtime/bin/python" -m pip freeze > "$runtime/bsu-environment.txt"
"$runtime/bin/python" -c 'from ltx_pipelines.distilled import DistilledPipeline; import torch; print("LTX_IMPORT_OK", torch.__version__, torch.cuda.get_device_name(0))'
"$runtime/bin/python" -m semantic_transmission.ltx_setup --model-dir "$repo/.local/checkpoints/ltx25" "$@"
