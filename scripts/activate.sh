#!/usr/bin/env bash
# Source this file from any working directory.
_semtx_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
_semtx_conda=${CONDA_BASE:-$HOME/anaconda3}
source "$_semtx_conda/etc/profile.d/conda.sh"
conda activate lgvsc
export PYTHONNOUSERSITE=1
unset PYTHONPATH LD_LIBRARY_PATH
export OPENSORA_DIR="$_semtx_root/.local/vendor/Open-Sora"
export INTERNVL_DIR="$_semtx_root/.local/vendor/InternVL"
export NTSCC_DIR="$_semtx_root/.local/vendor/NTSCC_JSAC22"
export NTSCC_CKPT="$_semtx_root/.local/checkpoints"
export LGVSC_PYTHON="$_semtx_conda/envs/lgvsc/bin/python"
unset _semtx_root _semtx_conda
