#!/usr/bin/env bash
# Source this helper, then call semtx_metric_runtime REPOSITORY.
semtx_metric_runtime() {
  local metric_runtime_repo="$1" metric_runtime_kernel metric_runtime_driver
  metric_python="${LGVSC_PYTHON:-}"
  if [[ -z "$metric_python" ]]; then
    if [[ -f "$metric_runtime_repo/.local/settings.json" ]]; then
      metric_python=$(python3 - "$metric_runtime_repo/.local/settings.json" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    print(json.load(stream)['python'])
PY
      ) || return
    else
      metric_python="${CONDA_BASE:-$HOME/anaconda3}/envs/lgvsc/bin/python"
    fi
  fi
  metric_object_python="${LGVSC_METRIC_PYTHON:-$metric_runtime_repo/.local/metric_v2_env/bin/python}"
  unset LD_LIBRARY_PATH
  metric_runtime_kernel=$(uname -r)
  if [[ "${metric_runtime_kernel,,}" == *microsoft* ]]; then
    if [[ -n "${LGVSC_DRIVER_LIBRARY:-}" && "$LGVSC_DRIVER_LIBRARY" != /usr/lib/wsl/lib ]]; then
      echo 'WSL must use its Windows-provided CUDA driver, not a copied Linux libcuda.' >&2
      return 1
    fi
    if [[ -d /usr/lib/wsl/lib ]]; then
      export LD_LIBRARY_PATH=/usr/lib/wsl/lib
    fi
  elif [[ -n "${LGVSC_DRIVER_LIBRARY:-}" ]]; then
    [[ -d "$LGVSC_DRIVER_LIBRARY" ]] || { echo 'LGVSC_DRIVER_LIBRARY is not a directory.' >&2; return 1; }
    export LD_LIBRARY_PATH="$LGVSC_DRIVER_LIBRARY"
  else
    # Preserve the original workstation workaround only for that exact driver.
    metric_runtime_driver="$metric_runtime_repo/.local/metric_v2_driver/extracted/usr/lib/x86_64-linux-gnu"
    if grep -q '580.173.02' /proc/driver/nvidia/version 2>/dev/null &&
       [[ -f "$metric_runtime_driver/libcuda.so.580.173.02" ]]; then
      export LD_LIBRARY_PATH="$metric_runtime_driver"
    fi
  fi
}
