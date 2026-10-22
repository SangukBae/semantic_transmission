#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
root="${SPARC_FINETUNE_OUTPUT:-$repo/outputs/sparc_selector_webvid_finetune_20261022_v2}"
case "${1:-watch}" in
  start|resume) action=launch ;;
  stop|check) action="$1" ;;
  watch|status)
    args=(--root "$root")
    [[ "${1:-watch}" != status ]] || args+=(--once)
    exec python3 scripts/watch_sparc_selector_finetune.py "${args[@]}" ;;
  *) echo "Usage: $0 {start|resume|watch|status|stop|check}" >&2; exit 2 ;;
esac
exec "$repo/.local/bsu_codec_eval_env/bin/python" -m semantic_transmission.sparc_selector_finetune "$action" --root "$root"
