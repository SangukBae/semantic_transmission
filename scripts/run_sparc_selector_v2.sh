#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SPARC_RUN="${SPARC_SELECTOR_OUTPUT:-$PWD/outputs/sparc_selector_cnn_attention_v2_20261020}"
SPARC_PARENT="${SPARC_SELECTOR_PARENT:-$PWD/outputs/sparc_selector_v1_20261020}"
SPARC_PYTHON="$PWD/.local/bsu_codec_eval_env/bin/python"
# A released v3 route preserves the immutable v2 labels but replaces its optimizer.
if [[ -f "$SPARC_RUN/.distributed/training_route.json" ]]; then
  case "${1:-watch}" in
    start|resume|stop)
      action=launch
      [[ "${1:-watch}" != stop ]] || action=stop
      exec env PYTHONPATH="$PWD/src" PYTHONNOUSERSITE=1 "$SPARC_PYTHON" \
        -m semantic_transmission.sparc_selector_dispatch "$action" \
        --root "$SPARC_RUN" --plan "$SPARC_RUN/.distributed/plan.json"
      ;;
  esac
fi
case "${1:-watch}" in
  watch|status)
    title="SPARC CNN + temporal Attention selector v2"
    [[ ! -f "$SPARC_RUN/.distributed/training_route.json" ]] || title="SPARC budget selector v3 / preserved v2 measurements"
    args=(--output "$SPARC_RUN" --title "$title"
          --resume-command "bash scripts/run_sparc_selector_v2.sh resume")
    [[ "${1:-watch}" != status ]] || args+=(--once)
    exec python3 scripts/watch_sparc_selector.py "${args[@]}"
    ;;
  start|resume)
    mkdir -p "$SPARC_RUN"
    [[ -f "$SPARC_PARENT/protocol.json" && -f "$SPARC_PARENT/sources.json" ]] || {
      echo "Missing frozen supervision manifest" >&2; exit 1;
    }
    if ! flock -n "$SPARC_RUN/run.lock" true; then
      echo "Already running; use: bash scripts/run_sparc_selector_v2.sh watch"
      exit 0
    fi
    if ! flock -n "$SPARC_PARENT/run.lock" true; then
      echo "Parent selector is still running; wait for its safe stop." >&2; exit 1
    fi
    nohup env PYTHONPATH="$PWD/src" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 \
      MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      "$SPARC_PYTHON" -m semantic_transmission.sparc_selector_train_v2 \
      --output "$SPARC_RUN" --parent "$SPARC_PARENT" \
      >> "$SPARC_RUN/run.log" 2>&1 < /dev/null &
    echo "$!" > "$SPARC_RUN/launcher.pid"
    echo "Launched PID $!; monitor: bash scripts/run_sparc_selector_v2.sh watch"
    ;;
  stop)
    python3 - "$SPARC_RUN" <<'PY'
import json,os,signal,sys
from pathlib import Path
root=Path(sys.argv[1])
v=json.loads((root/'status.json').read_text())
pid=v['pid']
args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
if (b'semantic_transmission.sparc_selector_train_v2' not in args
        or str(root).encode() not in args):
    raise SystemExit('Process identity mismatch; no signal sent')
os.kill(pid,signal.SIGTERM)
print('Stop requested; current measurement/optimizer update is committed before exit.')
PY
    ;;
  *) echo "Usage: $0 {start|resume|watch|status|stop}" >&2; exit 2 ;;
esac
