import json
import os
from pathlib import Path
import subprocess


HELPER = Path(__file__).resolve().parents[1] / 'scripts/metric_runtime.sh'


def runtime(tmp_path, **overrides):
    env = {k: v for k, v in os.environ.items()
           if k not in ('LGVSC_PYTHON', 'LGVSC_METRIC_PYTHON', 'LGVSC_DRIVER_LIBRARY')}
    env.update(overrides)
    return subprocess.run(
        ['bash', '-c', 'set -euo pipefail; source "$1"; semtx_metric_runtime "$2"; '
         'printf "%s\\n" "$metric_python" "$metric_object_python" "${LD_LIBRARY_PATH-}"',
         'runtime-test', str(HELPER), str(tmp_path)],
        env=env, capture_output=True, text=True,
    )


def test_machine_settings_and_explicit_interpreters(tmp_path):
    (tmp_path / '.local').mkdir()
    (tmp_path / '.local/settings.json').write_text(json.dumps({'python': '/new user/env/bin/python'}))
    result = runtime(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[:2] == [
        '/new user/env/bin/python', str(tmp_path / '.local/metric_v2_env/bin/python')]
    result = runtime(tmp_path, LGVSC_PYTHON='/override/core', LGVSC_METRIC_PYTHON='/override/metric')
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[:2] == ['/override/core', '/override/metric']


def test_new_conda_location_without_local_settings(tmp_path):
    result = runtime(tmp_path, CONDA_BASE='/new disk/miniconda3')
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == '/new disk/miniconda3/envs/lgvsc/bin/python'


def test_wsl_rejects_copied_linux_driver(tmp_path):
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    uname = fake_bin / 'uname'
    uname.write_text('#!/bin/sh\necho 6.6-microsoft-standard-WSL2\n')
    uname.chmod(0o755)
    path = str(fake_bin) + os.pathsep + os.environ['PATH']
    result = runtime(tmp_path, PATH=path, LD_LIBRARY_PATH='/old/linux/driver',
                     LGVSC_DRIVER_LIBRARY='/old/linux/driver')
    assert result.returncode != 0
    assert 'Windows-provided CUDA driver' in result.stderr
    result = runtime(tmp_path, PATH=path, LD_LIBRARY_PATH='/old/linux/driver')
    assert result.returncode == 0, result.stderr
    assert '/old/linux/driver' not in result.stdout
