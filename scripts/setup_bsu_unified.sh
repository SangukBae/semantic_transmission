#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
base_python=$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["python"])')
if [[ ! -x .local/unified_bsu_env/bin/python ]]; then
  "$base_python" -m venv --system-site-packages .local/unified_bsu_env
fi
runtime_python="$repo/.local/unified_bsu_env/bin/python"
dcvc_vendor="$repo/.local/vendor/DCVC/DCVC-family/DCVC-RT"
if [[ -d "$repo/.local/vendor/DCVC-RT" ]]; then
  dcvc_vendor="$repo/.local/vendor/DCVC-RT"
fi
if [[ ! -f "$dcvc_vendor/src/cpp/setup.py" ]]; then
  printf '%s\n' 'Pinned DCVC-RT source missing. See docs/RUN_GUIDE.md#bsu-unified.' >&2
  exit 1
fi
"$runtime_python" -m pip install pybind11==3.1.0 wheel
# Only this independent C++ entropy extension ignores upstream's Python>=3.12
# packaging floor. Its actual byte-stream round trip is tested on Python 3.10.
"$runtime_python" -m pip install --no-build-isolation --ignore-requires-python "$dcvc_vendor/src/cpp"
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2
"$runtime_python" - "$dcvc_vendor" <<'PY'
from pathlib import Path
import sys
import numpy as np
from semantic_transmission.bsu_dcvc_keys import Codec
# DISTS locates its installed calibration file through sys.prefix even when
# imported from the inherited environment; expose that same immutable file.
calibration=Path(sys.base_prefix)/'weights.pt'
target=Path(sys.prefix)/'weights.pt'
if calibration.is_file() and not target.exists():
    target.symlink_to(calibration)
codec=Codec(Path(sys.argv[1]),Path('.local/checkpoints/dcvc_rt').resolve())
image=np.zeros((2,64,64,3),np.uint8);image[:,20:40,20:40]=180
packet=codec.encode(image,[0,1],24);indices,restored=codec.decode(packet)
assert indices==[0,1] and restored.shape==image.shape
assert np.abs(image.astype(float)-restored).mean()<5
print('PASS: actual DCVC entropy encode/decode')
PY
