#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SPARC_RUN="${SPARC_SELECTOR_OUTPUT:-$PWD/outputs/sparc_selector_v1_20261020}"
SPARC_PYTHON="$PWD/.local/bsu_codec_eval_env/bin/python"
case "${1:-watch}" in
  watch|status)
    args=(--output "$SPARC_RUN")
    [[ "${1:-watch}" != status ]] || args+=(--once)
    exec python3 scripts/watch_sparc_selector.py "${args[@]}"
    ;;
  start|resume)
    mkdir -p "$SPARC_RUN"
    if [[ ! -f "$SPARC_RUN/sources.json" ]]; then
      echo "Missing source manifest; acquisition must finish first." >&2
      exit 1
    fi
    # Test the actual lock, not a stale PID file.
    if ! flock -n "$SPARC_RUN/run.lock" true; then
      echo "Already running. Use: bash scripts/run_sparc_selector.sh watch"
      exit 0
    fi
    nohup env PYTHONPATH="$PWD/src" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 \
      MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
      "$SPARC_PYTHON" -m semantic_transmission.sparc_selector_train \
      --output "$SPARC_RUN" >> "$SPARC_RUN/run.log" 2>&1 < /dev/null &
    echo "$!" > "$SPARC_RUN/launcher.pid"
    echo "Launched PID $!; monitor: bash scripts/run_sparc_selector.sh watch"
    ;;
  stop)
    python3 - "$SPARC_RUN" <<'PY'
import json,os,signal,sys
from pathlib import Path
root=Path(sys.argv[1])
v=json.loads((root/'status.json').read_text())
pid=v['pid']
args=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
if 'semantic_transmission.sparc_selector_train' not in args or str(root) not in args:
    raise SystemExit('Process identity mismatch; no signal sent')
os.kill(pid,signal.SIGTERM)
print('Stop requested; exits at a safe work boundary and retains resume state.')
PY
    ;;
  *) echo "Usage: $0 {start|resume|watch|status|stop}" >&2; exit 2 ;;
esac
