#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset PYTHONPATH LD_LIBRARY_PATH PYTORCH_CUDA_ALLOC_CONF
export PYTHONNOUSERSITE=1
base_python="${NTSCC_PLUS_BASE_PYTHON:-$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["python"])')}"
"$base_python" -c 'import sys; assert sys.version_info[:2] == (3, 10), "NTSCC+ wheel requires Python 3.10"'
vendor=.local/vendor/NTSCC_plus
revision=765c6f6c11cf3bbe249a26b624dbaf47a4f0676d
if [[ ! -d "$vendor" ]]; then
  git clone https://github.com/wsxtyrdd/NTSCC_plus.git "$vendor"
  git -C "$vendor" checkout --detach "$revision"
fi
[[ "$(git -C "$vendor" rev-parse HEAD)" == "$revision" ]]
git -C "$vendor" diff --exit-code
if [[ ! -x .local/ntscc_plus_env/bin/python ]]; then
  "$base_python" -m venv .local/ntscc_plus_env
fi
python=.local/ntscc_plus_env/bin/python
"$python" -m pip install 'torch==2.0.0+cu118' 'torchvision==0.15.1+cu118' \
  --extra-index-url https://download.pytorch.org/whl/cu118
"$python" -m pip install 'numpy==1.24.4' 'timm==0.6.13' 'compressai==1.2.4' \
  einops psutil 'gdown==5.2.0' \
  'natten @ https://github.com/SHI-Labs/NATTEN/releases/download/v0.14.6/natten-0.14.6%2Btorch200cu118-cp310-cp310-linux_x86_64.whl'
"$python" - <<'PY'
from pathlib import Path
import hashlib
import gdown
import torch
lib = Path(torch.__file__).parent / 'lib'
alias = lib / 'libnvrtc.so'
if not alias.exists():
    candidates = [p for p in lib.glob('libnvrtc-*.so.*') if not p.name.startswith('libnvrtc-builtins')]
    if len(candidates) != 1:
        raise RuntimeError('expected one bundled NVRTC library')
    alias.symlink_to(candidates[0].name)
path = Path('.local/checkpoints/ntscc_plus_ckbd2_lmbd_0.18.pth.tar')
path.parent.mkdir(parents=True, exist_ok=True)
if not path.exists():
    temporary = path.with_suffix(path.suffix + '.download')
    gdown.download(id='12zWIGHRfRVWvM3WcvpFRhawrfXwjql1y', output=str(temporary))
    if hashlib.sha256(temporary.read_bytes()).hexdigest() != '85181c01a7faab33f4c08fb6a0946032ad68bf581c8fe954f4484914dd2e36e8':
        raise RuntimeError('downloaded checkpoint hash mismatch')
    temporary.rename(path)
assert hashlib.sha256(path.read_bytes()).hexdigest() == '85181c01a7faab33f4c08fb6a0946032ad68bf581c8fe954f4484914dd2e36e8'
PY
"$python" -m pip check
"$python" -m pip freeze > .local/ntscc_plus_environment.txt
printf '%s\n' 'NTSCC+ installed. Run the key-codec validation before claiming inference success.'
