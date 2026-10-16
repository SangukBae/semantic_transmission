"""Fixed terminal progress with an independent, bounded NVIDIA sampler."""
import argparse
import csv
import io
import math
import os
from pathlib import Path
import subprocess
import threading
import time

from . import terminal_progress as terminal


def gpu_rows(text):
    utilization, memory = [], []
    for row in csv.reader(io.StringIO(text)):
        if len(row) != 5:
            raise ValueError('unexpected nvidia-smi columns')
        index, name, load, used, total = [s.strip() for s in row]
        load, used, total = float(load), float(used), float(total)
        if not (0 <= load <= 100 and 0 <= used <= total and total > 0):
            raise ValueError('invalid GPU measurement')
        utilization.append(f'GPU {index} {load:.0f}%')
        memory.append(f'GPU {index} {used/1024:.2f}/{total/1024:.2f} GiB ({100*used/total:.1f}%)')
    if not utilization:
        raise ValueError('no NVIDIA device')
    return ['GPU 사용률 : '+' | '.join(utilization), 'GPU 메모리 : '+' | '.join(memory)]


class Monitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.rows = ['GPU 사용률 : 확인 중', 'GPU 메모리 : 확인 중']
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            started = time.monotonic()
            try:
                result = subprocess.run(['nvidia-smi',
                    '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2, check=True)
                self.rows = gpu_rows(result.stdout)
            except (OSError, ValueError, subprocess.SubprocessError):
                # Never display an old sample as current, or confuse unavailable with 0%.
                self.rows = ['GPU 사용률 : 조회 불가', 'GPU 메모리 : 조회 불가']
            self.stop.wait(max(.05, self.interval-(time.monotonic()-started)))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--display', choices=('auto','live','plain'), default=os.environ.get('BSU_PROGRESS','auto'))
    parser.add_argument('--log-dir', type=Path, default=Path('.local/terminal_logs'))
    parser.add_argument('--stop-grace-seconds', type=float, default=30.,
                        help='time before forcing interrupted workers to exit; 0 waits for their checkpoint')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not math.isfinite(args.stop_grace_seconds) or args.stop_grace_seconds < 0:
        parser.error('--stop-grace-seconds must be finite and nonnegative')
    if args.display not in ('auto','live','plain'):
        parser.error('BSU_PROGRESS must be auto, live, or plain')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('worker command required after --')
    old = terminal.Progress
    with Monitor() as monitor:
        class Progress(old):
            def rows(self, elapsed, log):
                rows = super().rows(elapsed, log)
                return rows[:4] + list(monitor.rows) + rows[4:]
        terminal.Progress = Progress
        try:
            code = terminal.run(command, args.log_dir, args.display, stop_grace_seconds=args.stop_grace_seconds)
        finally:
            terminal.Progress = old
    raise SystemExit(code)


if __name__ == '__main__':
    main()
