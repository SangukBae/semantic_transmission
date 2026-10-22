#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
action=${1:-status}
if (($#)); then shift; fi
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
case "$action" in
  start|resume) action=launch ;;
  watch|status)
    root="$repo/outputs/sparc_selector_risk_v4_20261022"
    if [[ ${1:-} == --root ]]; then root=$2; shift 2; fi
    args=(--output "$root" --title 'SPARC v4 risk selector' --resume-command "bash scripts/run_sparc_selector_v4.sh resume --root $root")
    [[ "$action" == watch ]] || args+=(--once)
    exec python3 scripts/watch_sparc_selector.py "${args[@]}" "$@" ;;
  check|run|stop) ;;
  *) echo "Usage: $0 {check|start|resume|watch|status|stop} [--labels-only|--no-labels-only] [--root PATH] [--parent V3_RUN] [--source-root V2_RUN] [--runtimes JSON]" >&2; exit 2 ;;
esac
exec "$repo/.local/bsu_codec_eval_env/bin/python" -m semantic_transmission.sparc_selector_v4 "$action" "$@"
