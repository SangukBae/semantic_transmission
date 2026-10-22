#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
session=${LGVSC_OFFICIAL_SESSION:-lgvsc_official}
if [[ ! "$session" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo 'LGVSC_OFFICIAL_SESSION must contain letters, digits, _ or -.' >&2
  exit 2
fi
terminal=(tmux -L semantic-transmission)
case "${1:-}" in
  start)
    shift
    if "${terminal[@]}" has-session -t "=$session" 2>/dev/null; then
      echo "Session $session exists. Use watch, or a new LGVSC_OFFICIAL_SESSION." >&2
      exit 1
    fi
    # Complete all read-only validation before creating a persistent session.
    bash "$repo/scripts/run_lgvsc_official.sh" check "$@"
    printf -v worker '%q ' env BSU_PROGRESS=live bash "$repo/scripts/run_lgvsc_official.sh" run "$@"
    "${terminal[@]}" new-session -d -s "$session" -c "$repo" -x 180 -y 24 "$worker" \; \
      set-option -w -t "$session:" remain-on-exit on
    echo "Started $session. Progress: bash scripts/run_lgvsc_official.sh watch"
    ;;
  watch)
    [[ $# == 1 ]] || { echo 'watch takes no run options' >&2; exit 2; }
    exec "${terminal[@]}" attach-session -r -t "=$session"
    ;;
  stop)
    [[ $# == 1 ]] || { echo 'stop takes no run options' >&2; exit 2; }
    "${terminal[@]}" send-keys -t "$session:0.0" C-c
    echo 'Interrupt requested. Check status before restarting; completed stages are retained.'
    ;;
  *) echo 'Usage: start [run options] | watch | stop' >&2; exit 2 ;;
esac
