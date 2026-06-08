#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# LGVSC central path configuration.
#   Edit the values below for your machine, then:  source env.sh
# These variables are referenced by the per-stage scripts and READMEs.
# ─────────────────────────────────────────────────────────────────────────────

# Root directory that holds your raw input videos (*.mp4) and all derived
# artifacts (16x24/, frames/, *_save_dir_*/, ...). Each "dataset" is one DATA_ROOT.
export DATA_ROOT="${DATA_ROOT:-/path/to/your/dataset}"

# Clones of the upstream repos (see the dependency table in README.md).
export OPENSORA_DIR="${OPENSORA_DIR:-/path/to/Open-Sora}"        # hpcaitech Open-Sora @ bf4d6673
export INTERNVL_DIR="${INTERNVL_DIR:-/path/to/InternVL}"         # OpenGVLab InternVL
export NTSCC_DIR="${NTSCC_DIR:-/path/to/NTSCC_JSAC22}"           # wsxtyrdd NTSCC_JSAC22 (+ our patch)
export TIMESFORMER_DIR="${TIMESFORMER_DIR:-/path/to/TimeSformer}"
export DEPTH_DIR="${DEPTH_DIR:-/path/to/Depth-Anything-V2}"

# Model weights.
export INTERNVL_MODEL="${INTERNVL_MODEL:-OpenGVLab/InternVL2-8B}" # HF id or local path
export NTSCC_CKPT="${NTSCC_CKPT:-$NTSCC_DIR/checkpoints}"         # ntscc_hyperprior_quality_*.pth

# GPU
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[LGVSC] DATA_ROOT=$DATA_ROOT"
echo "[LGVSC] OPENSORA_DIR=$OPENSORA_DIR  INTERNVL_DIR=$INTERNVL_DIR  NTSCC_DIR=$NTSCC_DIR"

# ── Optional fine-grained overrides (sensible $DATA_ROOT-based defaults otherwise) ──
export NTSCC_CKPT="${NTSCC_CKPT:-$NTSCC_DIR/checkpoints}"   # SNR=10 quality_4 weight lives here
export NTSCC_MAYU="${NTSCC_MAYU:-$NTSCC_DIR/mayu}"          # SNR 0-8 weights (not in public release)
export VIDEO_FOLDER="${VIDEO_FOLDER:-$DATA_ROOT/16x24}"     # baselines (05) input folder
export TARGET_DIR="${TARGET_DIR:-$DATA_ROOT/summary}"       # downstream video-summary output
