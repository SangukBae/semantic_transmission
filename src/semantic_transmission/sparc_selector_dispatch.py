"""Finish immutable v2 measurements, then dispatch an explicitly pinned v3 run.

The frozen measurement modules and assignment plan remain byte-identical.
Only this process's post-measurement training callbacks are replaced.
"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import sparc_selector_distributed as d
from .sparc_selector_data import read, save, sha

MODULE = 'semantic_transmission.sparc_selector_dispatch'


def dispatch(root, p, rows):
    from . import sparc_selector_train as supervision
    route_path = root/'.distributed/training_route.json'
    while True:
        supervision.interrupted()
        route = read(route_path)
        if route.get('enabled'):
            break
        supervision.status(root, 'waiting_for_v3_release', resume='measurements retained; v2 optimizer disabled')
        time.sleep(5)
    if route.get('schema') != 'sparc_training_route_v3':
        raise ValueError('unrecognized automatic training route')
    for name, expected in route['code'].items():
        if sha(d.REPO/name) != expected:
            raise ValueError('released v3 source changed: '+name)
    if sha(route['config']) != route['config_sha256']:
        raise ValueError('released v3 settings changed')
    from .sparc_selector_train_v3 import pipeline
    return pipeline(Path(route['output']), root, p, rows, read(route['config']))


def run(root, plan):
    from . import sparc_selector_train_v2 as v2
    old_build, old_train = v2.build_inputs, v2.train
    v2.build_inputs = lambda *_: None
    v2.train = dispatch
    try:
        return d.worker(root, plan, 'remote', None)
    finally:
        v2.build_inputs, v2.train = old_build, old_train


def launch(root, plan):
    root = Path(root).resolve()
    d.checked_plan(plan)
    if not (root/'.distributed/training_route.json').exists():
        raise ValueError('automatic v3 training route is missing')
    with (root/'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status='ALREADY_RUNNING')
    env = dict(os.environ, PYTHONPATH=str(d.REPO/'src'), PYTHONNOUSERSITE='1',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (root/'.distributed/remote.log').open('ab') as log:
        proc = subprocess.Popen([sys.executable, '-m', MODULE, 'run', '--root', str(root),
                                 '--plan', str(Path(plan).resolve())], cwd=d.REPO, env=env,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    return dict(status='LAUNCHED', pid=proc.pid, training='v3_after_labels')


def stop(root):
    record = read(Path(root)/'.distributed/worker_remote.json')
    if not d.active(record):
        return dict(status='NOT_RUNNING')
    args = Path(f'/proc/{record["pid"]}/cmdline').read_bytes().split(b'\0')
    if (not any(m.encode() in args for m in (MODULE, d.MODULE)) or b'run' not in args
            or str(Path(root).resolve()).encode() not in args):
        raise ValueError('worker identity mismatch; no signal sent')
    os.kill(record['pid'], signal.SIGTERM)
    return dict(status='STOP_REQUESTED', pid=record['pid'])


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'launch', 'stop'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    result = run(args.root, args.plan) if args.command == 'run' else (
        launch(args.root, args.plan) if args.command == 'launch' else stop(args.root))
    if result is not None:
        print(json.dumps(result))


if __name__ == '__main__':
    main()
