#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
# No arguments prepare data only. Training requires: train ARM (or train all).
if (( $# == 0 )); then set -- prepare; fi
# Reject extra arguments before any training process can start.
if [[ "$1" == train && "${2:-}" == all ]] && (( $# != 2 )); then
    printf 'Usage: bash scripts/train_qwen35_evidence_repair.sh train all\n' >&2
    exit 2
fi
if [[ "$1" == train ]]; then
    qwen_repair_nofile=$(ulimit -Sn)
    qwen_repair_hard=$(ulimit -Hn)
    if [[ "$qwen_repair_nofile" != unlimited ]] && (( qwen_repair_nofile < 8192 )); then
        if [[ "$qwen_repair_hard" != unlimited ]] && (( qwen_repair_hard < 8192 )); then
            printf 'Qwen training requires an open-file hard limit >=8192; current=%s\n' "$qwen_repair_hard" >&2
            exit 1
        fi
        ulimit -Sn 8192
    fi
fi
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONPATH="$repo/src:$repo/scripts:$repo/.local/qwen35_training_probe_deps"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_ALLOC_CONF=expandable_segments:True
if [[ "$1" == train && "${2:-}" == all ]]; then
    # Each arm keeps its own existing completion/resume checks and GPU lock.
    # Separate processes release all model/CUDA resources between conditions.
    # set -e stops the sequence immediately if any arm fails.
    for qwen_repair_arm in caption evidence judgment preference; do
        printf 'RUN_CONDITION %s\n' "$qwen_repair_arm"
        .local/qwen35_env/bin/python -m semantic_transmission.qwen35_evidence_repair train "$qwen_repair_arm"
    done
    printf 'ALL_CONDITIONS_COMPLETE\n'
    exit 0
fi
exec .local/qwen35_env/bin/python -m semantic_transmission.qwen35_evidence_repair "$@"
