#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
export PYTHONPATH="$repo/src" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
root="${SPARC_WEBVID53_OUTPUT:-$repo/outputs/sparc_webvid53_qp30_selector_v3_20261022}"
module=semantic_transmission.sparc_webvid_evaluate
case "${1:-watch}" in
  start|resume) action=launch; module=semantic_transmission.sparc_webvid_recovery ;;
  stop) action=stop; module=semantic_transmission.sparc_webvid_recovery ;;
  check) action=check ;;
  watch|status)
    args=(--root "$root")
    [[ "${1:-watch}" != status ]] || args+=(--once)
    exec python3 scripts/watch_sparc_webvid53.py "${args[@]}" ;;
  *) echo "Usage: $0 {check|start|resume|watch|status|stop}" >&2; exit 2 ;;
esac
exec "$repo/.local/bsu_codec_eval_env/bin/python" -m "$module" "$action" --root "$root"
