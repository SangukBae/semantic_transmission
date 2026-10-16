#!/usr/bin/env bash
# Local WSL entry: execute on the configured GPU PC and retrieve final videos.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
action=${1:-run}
if [[ $# -gt 1 ]]; then
  printf '%s\n' 'Usage: bash scripts/run_bsu_etri01.sh [run|check|prepare|status|fetch]' >&2
  exit 2
fi
case "$action" in run|check|prepare|status|fetch) ;; *)
  printf '%s\n' 'Usage: bash scripts/run_bsu_etri01.sh [run|check|prepare|status|fetch]' >&2; exit 2 ;;
esac
remote_host=${BSU_REMOTE_HOST:-semantic-pc}
remote_repo=${BSU_REMOTE_REPO:-/home/sangukbae/ETRI/Semantic/semantic_transmission}
output_name=bsu_etri01_latest_20261016_v1
if [[ -x "$repo/.local/bsu_codec_eval_env/bin/python" && "$action" != fetch ]]; then
  exec bash scripts/run_bsu_video.sh "$action"
fi
printf -v remote_command 'cd %q && bash scripts/run_bsu_video.sh %q' "$remote_repo" "$action"
if [[ "$action" != fetch ]]; then
  ssh_flags=(-o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3)
  if [[ -t 0 && -t 1 ]]; then ssh_flags+=(-t); fi
  ssh "${ssh_flags[@]}" "$remote_host" "$remote_command"
fi
if [[ "$action" == run || "$action" == fetch ]]; then
  printf -v check_command 'cd %q && bash scripts/run_bsu_video.sh status' "$remote_repo"
  result=$(ssh -o BatchMode=yes "$remote_host" "$check_command")
  python3 -c 'import json,sys; v=json.loads(sys.stdin.read()); assert v["status"]=="COMPLETE_RAW_VIDEO_BSU" and v["completed_artifacts_verified"]' <<< "$result"
  local_output="$repo/outputs/$output_name/delivered"
  mkdir -p "$local_output"
  scp -q "$remote_host:$remote_repo/outputs/$output_name/bsu/delivery/refined/reconstruction.mp4" "$local_output/"
  scp -q "$remote_host:$remote_repo/outputs/$output_name/source_vs_bsu.mp4" "$local_output/"
  scp -q "$remote_host:$remote_repo/outputs/$output_name/RESULT.json" "$local_output/"
  python3 - "$local_output" <<'PY'
import hashlib, json, pathlib, sys
root=pathlib.Path(sys.argv[1]); result=json.loads((root/'RESULT.json').read_text())
for name, field in [('reconstruction.mp4','movie_sha256'),('source_vs_bsu.mp4','comparison_sha256')]:
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=result[field]:
        raise SystemExit('Downloaded video hash mismatch: '+name)
print('LOCAL_VERIFIED_RESULT: '+str(root/'reconstruction.mp4'))
print('LOCAL_COMPARISON: '+str(root/'source_vs_bsu.mp4'))
PY
fi
