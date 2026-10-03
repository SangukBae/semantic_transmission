#!/usr/bin/env bash
# Installs a separate environment and downloads official, revision-pinned weights.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .local/qwen35_env/bin/python ]]; then
    uv venv --python /usr/bin/python3 .local/qwen35_env
fi
uv pip install --python .local/qwen35_env/bin/python -r environment/requirements-qwen35.lock.txt
uv pip check --python .local/qwen35_env/bin/python
uv pip freeze --python .local/qwen35_env/bin/python > .local/qwen35_environment.lock.txt
env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
    .local/qwen35_env/bin/python scripts/qwen35_caption.py download
bash scripts/run_qwen35_caption.sh doctor
