#!/usr/bin/env bash
# A persistent terminal for the existing run command; watching never starts work.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
session=${LGVSC_KINETICS_SESSION:-lgvsc_kinetics}
if [[ ! "$session" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo 'LGVSC_KINETICS_SESSION must contain only letters, digits, _ or -.' >&2
  exit 2
fi
if ! command -v tmux >/dev/null; then
  echo 'tmux is required for start/watch. Foreground run and status remain available.' >&2
  exit 1
fi
terminal=(tmux -L semantic-transmission)
case "${1:-}" in
  start)
    shift
    if "${terminal[@]}" has-session -t "=$session" 2>/dev/null; then
      echo "Session $session already exists; use watch to inspect it." >&2
      exit 1
    fi
    # Quote each argument for tmux's shell, including paths containing spaces.
    printf -v worker '%q ' env BSU_PROGRESS=live bash "$repo/scripts/run_lgvsc_kinetics.sh" run "$@"
    "${terminal[@]}" new-session -d -s "$session" -c "$repo" -x 180 -y 24 "$worker" \; \
      set-option -w -t "$session:" remain-on-exit on
    echo "Started persistent session: $session"
    echo 'Progress: bash scripts/run_lgvsc_kinetics.sh watch'
    echo 'Detach from the progress screen with Ctrl+b, then d. The experiment keeps running.'
    ;;
  watch)
    if [[ $# != 1 ]]; then
      echo 'watch takes no run options; set LGVSC_KINETICS_SESSION for another session.' >&2
      exit 2
    fi
    if ! "${terminal[@]}" has-session -t "=$session" 2>/dev/null; then
      echo "No persistent session: $session" >&2
      echo 'Saved status: bash scripts/run_lgvsc_kinetics.sh status' >&2
      exit 1
    fi
    # Read-only clients cannot send Ctrl+C to the experiment.
    exec "${terminal[@]}" attach-session -r -t "=$session"
    ;;
  *) echo 'Usage: bash scripts/lgvsc_kinetics_session.sh {start [run options...]|watch}' >&2; exit 2 ;;
esac
