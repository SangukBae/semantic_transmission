#!/usr/bin/env bash
set -euo pipefail
action="${1:-watch}"
case "$action" in check|start|resume|watch|status|stop) ;; *) echo 'Use check/start/resume/watch/status/stop' >&2; exit 2 ;; esac
args=(-o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10)
[[ "$action" != watch ]] || args+=(-tt)
exec ssh "${args[@]}" semantic-pc "bash /home/sangukbae/ETRI/Semantic/semantic_transmission_sparc/scripts/run_sparc_webvid53.sh $action"
