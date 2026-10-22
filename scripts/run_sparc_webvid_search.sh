#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
root="${SPARC_WEBVID_SEARCH_OUTPUT:-$repo/outputs/sparc_selector_webvid_long_20261022}"
case "${1:-watch}" in
  start|resume)
    exec "$repo/.local/bsu_codec_eval_env/bin/python" -m semantic_transmission.sparc_selector_webvid_search launch --root "$root" ;;
  stop)
    exec "$repo/.local/bsu_codec_eval_env/bin/python" -m semantic_transmission.sparc_selector_webvid_search stop --root "$root" ;;
  watch|status)
    args=(--root "$root")
    [[ "${1:-watch}" != status ]] || args+=(--once)
    exec python3 scripts/watch_sparc_webvid_search.py "${args[@]}" ;;
  *) echo "Usage: $0 {start|resume|watch|status|stop}" >&2; exit 2 ;;
esac
