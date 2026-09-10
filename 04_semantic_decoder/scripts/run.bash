#!/usr/bin/env bash
set -euo pipefail
# Paths and CSV parsing are handled by the installed research package.
exec "${LGVSC_PYTHON:-python}" -m semantic_transmission.decoder_runner "$@"
