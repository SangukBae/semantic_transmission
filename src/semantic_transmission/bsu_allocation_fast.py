"""Verified execution-only continuation of a frozen allocation-label run.

The original protocol, code, model weights and completed stage receipts remain
unchanged. A separate immutable execution receipt binds this wrapper and its
cache to real, byte-exact, faster benchmark replays before it may run.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from . import bsu_allocation_labels as labels
from .artifacts import sha256, write_json
from .webvid5 import fingerprint

MODULE = 'semantic_transmission.bsu_allocation_fast'
PROFILE = 'execution_cache.json'


def verify_execution(root):
    root = Path(root)
    value = labels.read(root/PROFILE)
    if value.get('signature') != fingerprint({k:v for k,v in value.items() if k != 'signature'}):
        raise ValueError('execution receipt signature differs')
    protocol = labels.verify(root)
    if value['parent_protocol_signature'] != protocol['signature']:
        raise ValueError('execution receipt belongs to another experiment')
    for path, digest in value['code'].items():
        if sha256(path) != digest:
            raise ValueError('optimized execution code changed: '+path)
    evidence = value['validation']
    if sha256(evidence['path']) != evidence['sha256']:
        raise ValueError('execution validation evidence changed')
    validation = labels.read(evidence['path'])
    if validation.get('status') != 'PASSED_EXACT_FASTER_REPLAY':
        raise ValueError('a successful exact speed validation is required')
    if validation.get('cache_config') != value['cache_config']:
        raise ValueError('cache configuration differs from the measured configuration')
    cases = validation.get('comparisons', [])
    if len(cases) < 2 or not all(c['frames_identical'] and c['mp4_identical'] for c in cases):
        raise ValueError('at least two exact received-input replay comparisons are required')
    if not validation.get('speedup', 0) > 1.05:
        raise ValueError('measured speed gain is insufficient')
    for receipt in validation['benchmark_receipts']:
        if sha256(receipt['path']) != receipt['sha256']:
            raise ValueError('benchmark receipt changed')
    return value


def status_lines(root):
    rows = labels.status_lines(root)
    profile = labels.maybe(Path(root)/PROFILE)
    fit = labels.maybe(Path(root)/'predictor/status.json')
    steps, total = fit.get('step', 0), fit.get('total', 1000)
    state = fit.get('status', 'NOT_STARTED; waiting for measured labels')
    rows.insert(2, f'Execution   {profile.get("name", "original frozen execution")}')
    rows.insert(12, f'Predictor   {state} | {steps}/{total} steps ({100*steps/total:.1f}%)')
    rows = [('Resume      bash scripts/run_bsu_allocation_fast.sh run; verified stages/checkpoints reused'
             if row.startswith('Resume ') else row) for row in rows]
    return rows


def watch():
    path = labels.REPO/'scripts/watch_bsu_allocation_labels.py'
    spec = importlib.util.spec_from_file_location('allocation_fixed_monitor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.status_lines = status_lines
    sys.argv = [str(path), *sys.argv[2:]]
    module.main()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('command')
    parser.add_argument('--output', type=Path, default=labels.DEFAULT)
    parser.add_argument('--stage')
    args, _ = parser.parse_known_args()
    root = args.output.resolve()
    if args.command == 'watch':
        watch()
        return
    if args.command == 'status':
        print('\n'.join(status_lines(root)), flush=True)
        return
    execution = verify_execution(root)
    labels.MODULE = MODULE
    cache = None
    if args.command == 'worker' and args.stage == 'generate':
        from .ltx_execution_cache import install
        cache = install(**execution['cache_config'])
    started = time.monotonic()
    outcome = 'COMPLETE'
    try:
        labels.main()
    except BaseException:
        outcome = 'INTERRUPTED_OR_FAILED'
        raise
    finally:
        if cache is not None:
            write_json(root/'execution_cache_runs'/f'{os.getpid()}.json', dict(
                status=outcome, execution_signature=execution['signature'],
                seconds=time.monotonic()-started, stats=cache.stats, arguments=sys.argv[1:]))
            cache.close()


if __name__ == '__main__':
    main()
