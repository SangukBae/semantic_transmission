import errno
import fcntl
import io
import json
import os
from pathlib import Path
import pty
import select
import signal
import struct
import subprocess
import sys
import termios
import time

import pytest

from semantic_transmission.terminal_progress import Display, Progress, fit


REPO = Path(__file__).resolve().parents[1]


def environment():
    return dict(os.environ, PYTHONPATH=str(REPO/'src'), TERM='xterm-256color', BSU_PROGRESS='auto')


def invoke(tmp_path, code, *options):
    return [sys.executable, '-m', 'semantic_transmission.terminal_progress',
            '--log-dir', str(tmp_path/'logs'), *options, '--', sys.executable, '-u', '-c', code]


def terminal(command, *, cwd=REPO, columns=100, timeout=15):
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 24, columns, 0, 0))
    child = subprocess.Popen(command, cwd=cwd, env=environment(), stdout=slave, stderr=slave,
                             stdin=subprocess.DEVNULL)
    os.close(slave); chunks = []; deadline = time.monotonic()+timeout
    try:
        while time.monotonic() < deadline:
            if select.select([master], [], [], .1)[0]:
                try:
                    part = os.read(master, 65536)
                except OSError as error:
                    if error.errno == errno.EIO:
                        break
                    raise
                if not part:
                    break
                chunks.append(part)
        else:
            raise AssertionError('terminal worker timed out')
        return child.wait(timeout=5), b''.join(chunks).decode('utf-8')
    finally:
        if child.poll() is None:
            child.terminate(); child.wait(timeout=5)
        os.close(master)


def test_global_count_does_not_follow_per_video_completion():
    p = Progress()
    p.feed('BATCH 2/159 (1.3%) webvid_test SNR=6 mode=combined elapsed=7s\n')
    p.feed('Qwen 의미 변화 순차 판정: 50.0% (8/16) | 경과 2분 | ETA 2분\r')
    assert p.overall.startswith('2/159 (1.3%)')
    assert '8/16 (50.0%)' in p.stage and 'ETA 2분' in p.stage
    p.feed('전체 작업 100% 완료\n')
    assert p.overall.startswith('2/159')


def test_nested_window_counter_and_split_crlf():
    p = Progress(); p.feed('WINDOW 2/7 reference 3,')
    p.feed('000/6,000 (50.0%) elapsed=2s ETA=2s\r'); p.feed('\n')
    assert p.stage == 'WINDOW 2/7 reference 3000/6000 (50.0%) | ETA 2s'
    assert p.overall == '전체 건수 확인 중'


def test_resume_retains_verified_completion_while_visiting_old_jobs():
    p = Progress()
    p.feed('BSU WebVid reconstruction — delivered MP4\n')
    p.feed('Status: PARTIAL; complete video/SNR pairs 22/159\n')
    p.feed('BATCH 0/159 (0.0%) old SNR=6 mode=combined\nRESUME: verified\n')
    assert p.overall.startswith('22/159')
    p.feed('Status: PARTIAL; complete video/SNR pairs 23/159\n')
    p.feed('BATCH 2/159 (1.3%) new SNR=6 mode=combined\n')
    assert p.overall.startswith('23/159')


def test_preflight_resume_unknown_loading_and_error_visibility():
    p = Progress()
    p.feed('PREFLIGHT 1/53 (1.9%) video\n')
    assert p.stage.startswith('PREFLIGHT 1/53')
    p.feed('START base_123; log=/tmp/base.log; elapsed=0s, ETA unknown\n')
    assert p.stage == 'base_123 (진행률 미확정)'
    p.feed('Loading model weights\nRESUME: complete hashes verified\n')
    assert '재사용' in p.state
    p.feed('RuntimeError: worker failed', final=True)
    assert 'RuntimeError' in p.state and 'worker failed' in p.tail[-1]


def test_last_summary_replaces_old_incremental_tables():
    p = Progress()
    p.feed('BSU WebVid reconstruction old\nStatus: PARTIAL\n6\tbase\t1\n')
    p.feed('BSU WebVid reconstruction new\nStatus: COMPLETE\n6\tbase\t53\n')
    assert p.summary == ['BSU WebVid reconstruction new', 'Status: COMPLETE', '6\tbase\t53']


def test_fixed_rows_clear_stale_numbers_and_clip_korean_width():
    out = io.StringIO(); display = Display(out)
    display.draw(['1000/1000']*6); display.draw(['1/2']*6)
    assert '\x1b[6F' in out.getvalue()
    assert out.getvalue().count('\x1b[2K') == 12
    assert fit('가나다abc', 7) == '가나다'
    assert fit('\x1b[31m가\x1b[0m\t나', 6) == '가 나'


def test_redirected_output_is_plain_and_complete_log_retains_bytes(tmp_path):
    data = 'PREFLIGHT 1/2 (50.0%) 검사\rPREFLIGHT 2/2 (100.0%) 완료\nwarning detail\n'
    code = f'import os; os.write(1,{data.encode()!r})'
    result = subprocess.run(invoke(tmp_path, code), env=environment(), capture_output=True, timeout=10)
    assert result.returncode == 0 and b'\x1b[' not in result.stdout
    assert 'warning detail' in result.stdout.decode()
    assert next((tmp_path/'logs').glob('*.log')).read_bytes() == data.encode()


def test_real_tty_repaints_in_place_and_preserves_all_details(tmp_path):
    code = '''import time
print('BATCH 1/3 (33.3%) video SNR=10 mode=combined',flush=True)
for n in range(8):
 print(f'QUALITY {n+1}/8 ({100*(n+1)/8}%)',flush=True)
 print('detail only in full log',flush=True)
 time.sleep(.06)
print('BATCH 2/3 (66.7%) next SNR=10 mode=combined',flush=True)
'''
    status, screen = terminal(invoke(tmp_path, code))
    assert status == 0 and '\x1b[6F' in screen and '2/3 (66.7%)' in screen
    assert 'detail only in full log' not in screen
    log = next((tmp_path/'logs').glob('*.log')).read_text()
    assert log.count('detail only in full log') == 8 and 'QUALITY 8/8' in log
    assert '명령 정상 종료' in screen


def test_failed_tty_worker_keeps_exit_code_and_error_tail(tmp_path):
    code = "import sys; print('BATCH 0/3 (0.0%) video'); print('fatal quality input mismatch', file=sys.stderr); sys.exit(7)"
    status, screen = terminal(invoke(tmp_path, code))
    assert status == 7 and '실패 (종료 코드 7)' in screen
    assert 'fatal quality input mismatch' in screen and '0/3 (0.0%)' in screen
    assert '100.0%' not in screen


def test_plain_override_in_tty_and_unique_attempt_logs(tmp_path):
    for _ in range(2):
        status, screen = terminal(invoke(tmp_path, "print('worker detail')", '--display', 'plain'))
        assert status == 0 and '\x1b[' not in screen and 'worker detail' in screen
    assert len(list((tmp_path/'logs').glob('*.log'))) == 2


def test_interrupt_reaches_worker_group_and_allows_cleanup(tmp_path):
    marker = tmp_path/'cleanup.json'
    code = f'''import json, signal, subprocess, sys, time
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
def stop(sig,frame):
 child.wait(timeout=5)
 Path({str(marker)!r}).write_text(json.dumps(dict(grandchild_exit=child.returncode)))
 sys.exit(130)
signal.signal(signal.SIGINT,stop)
print('READY_FOR_INTERRUPT',flush=True)
time.sleep(60)
'''
    wrapper = subprocess.Popen(invoke(tmp_path, code), env=environment(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            logs=list((tmp_path/'logs').glob('*.log'))
            if logs and 'READY_FOR_INTERRUPT' in logs[0].read_text():
                break
            time.sleep(.05)
        else:
            raise AssertionError('worker did not start')
        # Allow the grandchild to finish interpreter initialization too.
        time.sleep(.15); wrapper.send_signal(signal.SIGINT)
        stdout,_=wrapper.communicate(timeout=8)
        assert wrapper.returncode==130 and '중단됨' in stdout.decode()
        assert json.loads(marker.read_text())['grandchild_exit'] != 0
    finally:
        if wrapper.poll() is None:
            wrapper.terminate(); wrapper.communicate(timeout=8)


def test_launcher_keeps_arguments_and_status_remains_machine_readable(tmp_path):
    import shutil
    package=tmp_path/'src/semantic_transmission';package.mkdir(parents=True)
    (package/'__init__.py').write_text('')
    shutil.copyfile(REPO/'src/semantic_transmission/terminal_progress.py',package/'terminal_progress.py')
    (package/'bsu_default.py').write_text("import json,sys\nprint(json.dumps({'action':sys.argv[1:]}))\n")
    (tmp_path/'scripts').mkdir();script=tmp_path/'scripts/run_bsu_webvid.sh'
    shutil.copyfile(REPO/'scripts/run_bsu_webvid.sh',script)
    (tmp_path/'.local').mkdir();(tmp_path/'.local/settings.json').write_text(json.dumps(dict(python=sys.executable)))
    code,screen=terminal(['bash',str(script),'run','--root','path with spaces'],cwd=tmp_path)
    assert code==0 and '\x1b[6F' in screen
    log=next((tmp_path/'.local/terminal_logs').glob('*.log'))
    assert json.loads(log.read_text())['action']==['run','--root','path with spaces']
    status=subprocess.run(['bash',str(script),'status'],cwd=tmp_path,capture_output=True,text=True,timeout=5)
    assert status.returncode==0 and json.loads(status.stdout)==dict(action=['status'])
