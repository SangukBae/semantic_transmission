#!/usr/bin/env bash
# Source this file from any working directory.
_semtx_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
_semtx_conda=${CONDA_BASE:-$HOME/anaconda3}
source "$_semtx_conda/etc/profile.d/conda.sh"
_semtx_python="$_semtx_conda/envs/lgvsc/bin/python"
if [[ -f "$_semtx_root/.local/settings.json" ]]; then
    _semtx_python=$("$_semtx_conda/bin/python" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["python"])' \
        "$_semtx_root/.local/settings.json") || return 1
fi
conda activate "$(dirname -- "$(dirname -- "$_semtx_python")")" || return 1
export PYTHONNOUSERSITE=1
unset PYTHONPATH LD_LIBRARY_PATH
export OPENSORA_DIR="$_semtx_root/.local/vendor/Open-Sora"
export INTERNVL_DIR="$_semtx_root/.local/vendor/InternVL"
export NTSCC_DIR="$_semtx_root/.local/vendor/NTSCC_JSAC22"
export NTSCC_CKPT="$_semtx_root/.local/checkpoints"
export LGVSC_PYTHON="$_semtx_python"
unset _semtx_root _semtx_conda _semtx_python
