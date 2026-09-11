#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
core=$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["python"])')
exec 9>.local/etri_run.lock
if ! flock -n 9; then
  echo "이미 ETRI 영상 생성 명령이 실행 중입니다." >&2
  exit 1
fi
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
exec "$core" -m semantic_transmission.continue_etri "$@"
