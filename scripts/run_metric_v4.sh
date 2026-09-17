#!/usr/bin/env bash
set -euo pipefail
metric_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$metric_repo"
metric_phase="${1:?phase required}"
metric_output="${2:?output required}"
export PYTHONNOUSERSITE=1 PYTHONPATH="$metric_repo/src"
unset LD_LIBRARY_PATH
metric_python=/home/sangukbae/anaconda3/envs/lgvsc/bin/python
if [[ "$metric_phase" == visual || "$metric_phase" == development ]]; then
  metric_python="$metric_repo/.local/metric_v2_env/bin/python"
fi
if [[ "$metric_phase" == visual || "$metric_phase" == pixel ]]; then
  export LD_LIBRARY_PATH="$metric_repo/.local/metric_v2_driver/extracted/usr/lib/x86_64-linux-gnu"
fi
if [[ "$metric_phase" == declare ]]; then
  "$metric_python" scripts/report_metric_v4.py --output "$metric_output" --declare
  exec "$metric_python" - "$metric_output" <<'PY'
from pathlib import Path
import sys
import time
from semantic_transmission.artifacts import sha256
from semantic_transmission.metric_v3_formal import save
root=Path(sys.argv[1]); declaration=root/'input_audit_declaration.json'
if declaration.exists() or (root/'scores').exists():
    raise ValueError('Declare input audit before scoring, in a fresh run')
save(declaration,{'created_unix':time.time(),'heldout_scores_observed':0,
    'script_sha256':sha256(Path('scripts/audit_metric_v4_inputs.py')),
    'scope':'Actual PNG decode and missing-input rejection audit; technical failures only'})
PY
elif [[ "$metric_phase" == report ]]; then
  exec "$metric_python" scripts/run_metric_v4_report.py --output "$metric_output"
fi
exec "$metric_python" -m semantic_transmission.metric_v4_pipeline "$metric_phase" --output "$metric_output"
