import json
from pathlib import Path
import subprocess
import sys

import pytest

from semantic_transmission import bsu_video as video
from semantic_transmission.artifacts import sha256, write_json


def test_raw_pipeline_stops_legacy_jobs_before_channel_or_decoder():
    stages = [name for name, _, _ in video.sender_jobs()]
    assert stages == ['input-audit', 'semantic-clips', 'caption', 'flow', 'send']
    assert not set(stages) & {'channel', 'receive', 'prepare-text', 'reconstruct', 'evaluate'}


def test_worker_failure_preserves_log_and_exit_status(tmp_path):
    log = tmp_path / 'worker.log'
    with pytest.raises(subprocess.CalledProcessError) as error:
        video.launch([sys.executable, '-c', 'print("retained evidence", flush=True); raise SystemExit(7)'], log)
    assert error.value.returncode == 7
    assert 'retained evidence' in log.read_text()
    assert json.loads(log.with_suffix('.resource.json').read_text())['returncode'] == 7


def test_sender_trajectory_matches_existing_policy_without_decoder(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image
    from semantic_transmission import bsu_sequence_refinement as previous
    from semantic_transmission.packets import accounting
    from semantic_transmission.wire import pack
    run = tmp_path / 'run'
    tx = run / 'transmitter'
    tx.mkdir(parents=True)
    source = tmp_path / 'source'
    source.mkdir()
    background = np.random.default_rng(5).integers(0, 256, (160, 288, 3), dtype=np.uint8)
    for index in range(12):
        Image.fromarray(np.roll(background, index, axis=1)).save(source / f'{index}.png')
    packet = pack(dict(video=dict(frames=12, fps=24, width=288, height=160),
                       keyframes=[dict(index=0), dict(index=11)]))
    (tx / 'metadata.bin').write_bytes(packet)
    (tx / 'visual.c64').write_bytes(bytes(128000 * 8))
    video.trajectory_sender(run, source, tmp_path / 'new')
    assert not (run / 'output_audit.json').exists()
    monkeypatch.setattr(previous, 'paths_for_run', lambda _: (run, None, None))
    previous.prepare(dict(base='unused', base_channel_uses=128000 + accounting(len(packet), bits_per_symbol=4)['complex_channel_uses'],
        digital_bits_per_symbol=4, extra_rate_ratio=.05, mode='combined', source_frames=str(source)), tmp_path / 'old')
    assert json.loads((tmp_path / 'new/sender.json').read_text()) == json.loads((tmp_path / 'old/sender.json').read_text())
    assert (tmp_path / 'new/trajectory.bin').read_bytes() == (tmp_path / 'old/trajectory.bin').read_bytes()


def test_status_rejects_changed_delivered_video(tmp_path):
    paths = [tmp_path / name for name in ('video.mp4', 'comparison.mp4', 'receiver.json')]
    for path in paths:
        path.write_bytes(b'original')
    write_json(tmp_path / 'RESULT.json', dict(status='COMPLETE_RAW_VIDEO_BSU',
        movie=str(paths[0]), movie_sha256=sha256(paths[0]),
        comparison=str(paths[1]), comparison_sha256=sha256(paths[1]),
        receiver_result=str(paths[2]), receiver_result_sha256=sha256(paths[2])))
    assert video.status(tmp_path)['completed_artifacts_verified']
    paths[0].write_bytes(b'changed')
    with pytest.raises(ValueError, match='completed artifact changed'):
        video.status(tmp_path)


def test_full_video_verification_rejects_truncation(tmp_path):
    movie = tmp_path / 'short.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=black:s=64x32:r=24:d=1',
                    '-c:v', 'libx264', '-threads', '1', str(movie)], check=True)
    source = dict(frames=24, fps=24, width=64, height=32)
    assert video.verify_movie(movie, source)['decoded_to_eof']
    with pytest.raises(ValueError, match='full-timeline video differs'):
        video.verify_movie(movie, dict(source, frames=48))


def test_ssh_hangup_stops_nested_worker_and_preserves_log(tmp_path):
    import os
    import signal
    import time
    ready = tmp_path / 'worker.pid'
    worker = f'import os,time; from pathlib import Path; Path({str(ready)!r}).write_text(str(os.getpid())); print("worker ready",flush=True); time.sleep(60)'
    controller = f'''import signal,sys
from pathlib import Path
from semantic_transmission.bsu_video import launch
def stop(*_): raise KeyboardInterrupt('stopped')
signal.signal(signal.SIGTERM,stop)
launch([sys.executable,'-c',{worker!r}],Path({str(tmp_path / 'worker.log')!r}))
'''
    command = [sys.executable, '-c', 'from semantic_transmission.bsu_video import progress_main; progress_main()',
        '--display', 'plain', '--log-dir', str(tmp_path / 'logs'), '--', sys.executable, '-c', controller]
    env = dict(os.environ, PYTHONPATH=str(video.REPO / 'src'), PYTHONNOUSERSITE='1')
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic()+8
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert ready.exists()
        pid = int(ready.read_text())
        process.send_signal(signal.SIGHUP)
        output = process.communicate(timeout=8)[0].decode()
        assert process.returncode == 143
        assert '중단됨' in output
        assert not Path(f'/proc/{pid}').exists()
        assert 'worker ready' in (tmp_path / 'worker.log').read_text()
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=8)
