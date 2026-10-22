"""Read-only fixed-row progress for the detached four-video UF experiment."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from .terminal_gpu_progress import Monitor
from .terminal_progress import Display, duration

DEFAULT = Path(__file__).resolve().parents[2]/'outputs/kinetics_001_004_uf_rate_20261020'


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError,ValueError,OSError):
        return {}


def process_alive(pid):
    if type(pid) is not int or pid <= 0:
        return False
    try:
        state = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]
        return state != 'Z'
    except OSError:
        return False


def rows(root, gpu):
    state = read(root/'status.json')
    launcher = read(root/'launcher.json')
    status = state.get('status','PREFLIGHT')
    pid = state.get('pid',launcher.get('pid'))
    if not process_alive(pid) and status in ('RUNNING','PREFLIGHT'):
        status = 'STOPPED: process absent; inspect controller log'
    done,total = state.get('completed_trials',0),state.get('total_trials',36)
    current = f"{state.get('clip','001–004')} / {state.get('trial','input/model verification')}"
    stage = state.get('stage','preflight')
    active = Path(state['active_root']) if state.get('active_root') else None
    log = state.get('log') or launcher.get('log',str(root/'controller.log'))
    if active:
        if stage == 'reference':
            child = read(active/'status.json')
            stage = 'SPARC '+child.get('stage','prepare')
            log = str(active/'logs'/f"{child.get('stage','prepare')}.log")
        progress = read(active/'worker_progress.json')
        if progress.get('stage') == stage.removeprefix('SPARC '):
            n,t = progress.get('done',0),progress.get('total',0)
            stage += f' {n}/{t}'+(f' ({100*n/t:.1f}%)' if t else '')
    started = state.get('started_unix',launcher.get('started_unix',time.time()))
    output = [
        f'전체 진행 : {done}/{total} ({100*done/total:.1f}%) 조건 완료',
        f'현재 작업 : {current}',
        f'단계 진행 : {stage}',
        f'경과 시간 : {duration(time.time()-started)} | ETA 미확정',
        *gpu,
        f'실행 상태 : {status} | PID {pid} | exit {state.get("exit_code","-")}',
        f'상세 로그 : {log}',
        f'전체 로그 : {launcher.get("log",str(root/"controller.log"))}',
        '재개 명령 : bash scripts/run_sparc_uf_rate.sh run (완료 단계 해시 검증)',
    ]
    if state.get('error'):
        output.append('오류 내용 : '+state['error'])
    return output,status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--once',action='store_true')
    args = p.parse_args()
    if args.once:
        import subprocess
        from .terminal_gpu_progress import gpu_rows
        try:
            data = subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
                                            '--format=csv,noheader,nounits'],text=True,timeout=3)
            gpu = gpu_rows(data)
        except (OSError,ValueError,subprocess.SubprocessError):
            gpu = ['GPU 사용률 : 조회 불가','GPU 메모리 : 조회 불가']
        print('\n'.join(rows(args.output,gpu)[0]))
        return
    display = Display(sys.stdout)
    try:
        with Monitor() as monitor:
            while True:
                output,status = rows(args.output,monitor.rows)
                if sys.stdout.isatty():
                    display.draw(output)
                else:
                    print('\n'.join(output),flush=True)
                if status.startswith('COMPLETE_') or status in ('FAILED','INTERRUPTED') or status.startswith('STOPPED'):
                    break
                time.sleep(1)
    except KeyboardInterrupt:
        print('\n모니터만 종료했습니다. 실험은 계속 실행됩니다.',flush=True)


if __name__ == '__main__':
    main()
