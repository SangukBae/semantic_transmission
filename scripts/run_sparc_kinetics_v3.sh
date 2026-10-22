#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
root="${SPARC_KINETICS_V3_OUTPUT:-$repo/outputs/sparc_kinetics14_qp30_v3_20261022}"
case "${1:-watch}" in
  start|resume) action=launch ;;
  check|stop) action="$1" ;;
  watch|status)
    args=(--root "$root")
    [[ "${1:-watch}" != status ]] || args+=(--once)
    exec python3 scripts/watch_sparc_kinetics_v3.py "${args[@]}" ;;
  report)
    exec python3 -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1])/"COMPARISON.json"; print(json.dumps(json.loads(p.read_text()),ensure_ascii=False,indent=2) if p.exists() else "Comparison not complete. Use status or watch.")' "$root" ;;
  *) echo "Usage: $0 {check|start|resume|watch|status|stop|report}" >&2; exit 2 ;;
esac
exec "$repo/.local/bsu_codec_eval_env/bin/python" -m semantic_transmission.sparc_kinetics_evaluate "$action" --root "$root"
