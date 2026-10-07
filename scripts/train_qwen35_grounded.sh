#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
# WSL's GPU allocation handles consume process file descriptors. Interactive
# terminals may inherit1024 even when the Codex runner inherits1048576.
# Raise only this child shell's soft limit, within the existing hard limit.
qwen_nofile_soft=$(ulimit -Sn)
qwen_nofile_hard=$(ulimit -Hn)
qwen_nofile_required=8192
if [[ "$qwen_nofile_soft" != unlimited ]] && (( qwen_nofile_soft < qwen_nofile_required )); then
    if [[ "$qwen_nofile_hard" != unlimited ]] && (( qwen_nofile_hard < qwen_nofile_required )); then
        printf 'Qwen GPU runtime needs an open-file hard limit of at least %s; current soft=%s hard=%s.\n' \
            "$qwen_nofile_required" "$qwen_nofile_soft" "$qwen_nofile_hard" >&2
        exit 1
    fi
    ulimit -Sn "$qwen_nofile_required"
fi
printf 'GPU_RUNTIME open_files_soft=%s hard=%s (inherited_soft=%s)\n' \
    "$(ulimit -Sn)" "$qwen_nofile_hard" "$qwen_nofile_soft" >&2
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONPATH="$repo/src:$repo/.local/qwen35_training_probe_deps:$repo/scripts"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_ALLOC_CONF=expandable_segments:True
if [[ "${1:-}" == runtime-check ]]; then
    shift
    exec .local/qwen35_env/bin/python scripts/probe_qwen35_grounded_runtime.py \
        --report "outputs/qwen35_grounded_v1/runtime_check_$(date -u +%Y%m%dT%H%M%S).json" "$@"
fi
exec .local/qwen35_env/bin/python -m semantic_transmission.qwen35_grounded_training "$@"
