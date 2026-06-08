#!/bin/bash
# Small batch runner for SKEM (PSSS-guided) keyframe selection.
# Source env.sh at the repo root first (exports DATA_ROOT, INTERNVL_DIR), then edit
# --method / --threshold below for each run.

# InternVL's `internvl_chat` package must be importable:
export PYTHONPATH="$INTERNVL_DIR:$PYTHONPATH"

python "$(dirname "$0")/MLM-keyframe-internvl.py" \
    --csv-path "$DATA_ROOT/16x24/videos.csv" \
    --method   internvl_diff_0.35 \
    --threshold 0.35
