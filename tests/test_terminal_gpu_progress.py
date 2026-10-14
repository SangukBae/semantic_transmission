import subprocess
import sys

import pytest

from semantic_transmission import terminal_gpu_progress as gpu
from test_terminal_progress import terminal


def test_gpu_numeric_rows_and_multidevice():
    rows=gpu.gpu_rows('0, NVIDIA GPU, 81, 8192, 16384\n1, Other GPU, 0, 0, 8192\n')
    assert rows==['GPU 사용률 : GPU 0 81% | GPU 1 0%',
        'GPU 메모리 : GPU 0 8.00/16.00 GiB (50.0%) | GPU 1 0.00/8.00 GiB (0.0%)']


@pytest.mark.parametrize('text',['','0, NVIDIA, N/A, 0, 16384','0, GPU, 101, 0, 10',
    '0, GPU, 0, 11, 10','0, GPU, nan, 0, 10','0, GPU, 0, 0, 0'])
def test_invalid_measurement_never_becomes_zero(text):
    with pytest.raises(ValueError):gpu.gpu_rows(text)


def test_sampler_drops_stale_measurement_and_bounds_driver_timeout(monkeypatch):
    monitor=gpu.Monitor(.01);calls=[]
    def query(*args,**kwargs):
        calls.append(kwargs['timeout'])
        if len(calls)==1:return subprocess.CompletedProcess(args,0,'0, GPU, 55, 1024, 2048')
        monitor.stop.set()
        raise subprocess.TimeoutExpired('nvidia-smi',2)
    monkeypatch.setattr(gpu.subprocess,'run',query)
    monitor.sample()
    assert calls==[2,2] and all('조회 불가' in row for row in monitor.rows)


@pytest.mark.parametrize('exit_code',[0,7])
def test_real_tty_gpu_refresh_during_silent_worker_and_failure_log(tmp_path,monkeypatch,exit_code):
    fake=tmp_path/'nvidia-smi';fake.write_text('#!/bin/sh\nprintf "0, FakeGPU, 77, 4096, 16384\\n"\n')
    fake.chmod(0o755)
    import os
    monkeypatch.setenv('PATH',str(tmp_path)+os.pathsep+os.environ['PATH'])
    code=f'import time; print("detail for the full log",flush=True); time.sleep(1.2); raise SystemExit({exit_code})'
    command=[sys.executable,'-m','semantic_transmission.terminal_gpu_progress','--log-dir',str(tmp_path/'logs'),
        '--',sys.executable,'-u','-c',code]
    status,screen=terminal(command)
    assert status==exit_code and '\x1b[8F' in screen
    assert 'GPU 0 77%' in screen and '4.00/16.00 GiB (25.0%)' in screen
    assert 'detail for the full log' in next((tmp_path/'logs').glob('*.log')).read_text()
    if exit_code==0:assert 'detail for the full log' not in screen
    else:assert '실패' in screen
