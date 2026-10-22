"""Start/resume/stop the two-host job, or display its fixed-row progress."""
import argparse
import json
import os
from pathlib import Path
import shutil
import shlex
import signal
import subprocess
import sys
import time

from semantic_transmission import sparc_selector_distributed as d


def remote_training(c, action):
    if not c.get('training_v3'):
        args = [action, '--root', c['remote_root'], '--role', 'remote']
        if action == 'launch':
            args += ['--plan', c['remote_plan']]
        return d.remote_call(c, args)
    args = ['env', 'PYTHONNOUSERSITE=1', 'PYTHONDONTWRITEBYTECODE=1',
            'PYTHONPATH='+c['remote_repo']+'/src', c['remote_python'], '-m',
            'semantic_transmission.sparc_selector_dispatch', action,
            '--root', c['remote_root'], '--plan', c['remote_plan']]
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o',
        'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10', c['host'], shlex.join(args)],
        text=True, timeout=120)


def start(config_path, c):
    evidence = d.read(c['preflight'])
    if evidence.get('status') != 'PASSED' or evidence.get('plan') != d.checked_plan(c['local_plan'])['signature']:
        raise ValueError('matching two-host preflight has not passed')
    remote = remote_training(c, 'launch')
    print('Remote:', remote.strip())
    command = [c['local_python'], '-m', d.MODULE, 'launch', '--root', c['local_root'],
               '--plan', c['local_plan'], '--role', 'local', '--runtimes', c['local_runtimes']]
    env = dict(os.environ, PYTHONPATH=str(d.REPO/'src'), PYTHONNOUSERSITE='1',
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    print('Local:', subprocess.check_output(command, env=env, text=True).strip())
    folder = Path(c['local_root'])/'.distributed'
    state = d.read(folder/'sync.json') if (folder/'sync.json').exists() else {}
    if not d.active(state):
        with (folder/'sync.log').open('ab') as log:
            process = subprocess.Popen([sys.executable, '-m', d.MODULE, 'sync', '--config', str(config_path)],
                cwd=d.REPO, env=env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        print('Verified-result synchronization PID:', process.pid)


def stop(c):
    print('Local:', d.stop_worker(c['local_root'], 'local'))
    print('Remote:', remote_training(c, 'stop').strip())
    path = Path(c['local_root'])/'.distributed/sync.json'
    if path.exists():
        state = d.read(path)
        if d.active(state):
            args = Path(f'/proc/{state["pid"]}/cmdline').read_bytes().split(b'\0')
            if d.MODULE.encode() not in args or b'sync' not in args:
                raise ValueError('sync process identity mismatch')
            os.kill(state['pid'], signal.SIGTERM)
    print('Safe stop requested; completed results and training checkpoints are retained.')


def duration(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'


def lines(c, fresh=False):
    root = Path(c['local_root'])
    plan = d.checked_plan(c['local_plan'])
    local = d.snapshot(root, c['local_plan'], 'local')
    cache = root/'.distributed/remote_snapshot.json'
    remote = d.read(cache) if cache.exists() else {}
    error = ''
    if fresh or time.time()-remote.get('observed_unix', 0) > 15:
        try:
            remote = json.loads(d.remote_call(c, ['snapshot', '--root', c['remote_root'],
                '--plan', c['remote_plan'], '--role', 'remote'], timeout=15))
        except (OSError, ValueError, subprocess.SubprocessError) as failure:
            error = str(failure)
    sync_path = root/'.distributed/sync.json'
    sync = d.read(sync_path) if sync_path.exists() else {}
    imported = remote.get('imported', 0)
    combined = min(len(plan['prepared']), remote.get('global_done', len(plan['completed']))+max(0, local['done']-imported))
    def worker(name, value):
        count, total = value.get('done', 0), value.get('total', len(plan['assignments'][name.lower()]))
        state = value.get('status', 'UNAVAILABLE')
        if state in ('RUNNING', 'VERIFYING', 'WAITING_FOR_LOCAL', 'TRAINING') and not value.get('alive'):
            state += ' (PROCESS NOT ALIVE)'
        return f'{name:9} {count}/{total} ({100*count/total if total else 100:.1f}%) | valid {value.get("valid", 0)}, rejected {value.get("rejected", 0)} | {state}'
    def gpu(value):
        fields = value.get('gpu', '').split(',')
        return f'{fields[0].strip()}% | VRAM {fields[1].strip()}/{fields[2].strip()} MiB' if len(fields) == 3 else 'unavailable'
    training = remote.get('training', {})
    stage_done, stage_total = training.get('done'), training.get('total')
    stage_progress = (f'{stage_done}/{stage_total} ({100*stage_done/stage_total:.1f}%)'
                      if stage_done is not None and stage_total else 'pending')
    batch_progress = (f' | batch {training["batch_done"]}/{training["batch_total"]}'
                      if training.get('batch_total') else '')
    return [
        'SPARC exact-protocol labels: remote Linux PC + local WSL',
        f'Total     {combined}/{len(plan["prepared"])} ({100*combined/len(plan["prepared"]):.1f}%) | preserved {len(plan["completed"])}',
        worker('Remote', remote), worker('Local', local),
        f'Published {imported}/{len(plan["assignments"]["local"])} local clips verified on remote | sync {sync.get("status", "NOT_STARTED")}',
        f'PC stored valid={remote.get("global_valid", 0)} rejected={remote.get("global_rejected", 0)} | channel failures are excluded from training',
        f'Remote    {remote.get("clip", "-")} | pair {remote.get("pair", "-")}/{remote.get("pairs_this_clip", "-")}',
        f'Local     {local.get("clip", "-")} | pair {local.get("pair", "-")}/{local.get("pairs_this_clip", "-")}',
        f'GPU PC    {gpu(remote)} (all processes)',
        f'GPU local {gpu(local)} (all processes)',
        f'Elapsed   {duration(time.time()-plan["created_unix"])} | ETA unknown: mixed clip lengths and concurrent GPU work',
        f'Training  {training.get("stage", "-")} | {stage_progress}{batch_progress} | optimizer step {training.get("step", "not started")}',
        f'Output    {training.get("v3_output", "v3 starts automatically after labels" if c.get("training_v3") else c["remote_root"])}',
        f'PC status age {duration(time.time()-remote.get("observed_unix", time.time()))} | communication {error or "OK"}',
        f'Failure   local={local.get("error", "-")} | remote={remote.get("error", "-")} | sync={sync.get("last_error", "-")}',
        f'Local log {root/".distributed/local.log"}',
        f'PC log    {c["remote_root"]}/.distributed/remote.log',
        'Resume    bash scripts/run_sparc_selector_distributed.sh resume',
        'Ctrl-C closes this monitor only; both workers and synchronization continue.',
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['start', 'resume', 'stop', 'watch', 'status'])
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    c = d.read(a.config)
    if a.action in ('start', 'resume'):
        start(a.config.resolve(), c)
        return
    if a.action == 'stop':
        stop(c)
        return
    if a.action == 'status' or not sys.stdout.isatty():
        print('\n'.join(lines(c, fresh=True)))
        return
    previous = ()
    try:
        sys.stdout.write('\x1b[?1049h\x1b[?25l')
        while True:
            size = shutil.get_terminal_size((110, 25))
            if size.columns < 1 or size.lines < 1:
                size = os.terminal_size((110, 25))
            current = tuple(x[:max(1, size.columns-1)] for x in lines(c)[:max(1, size.lines-1)])
            for i in range(max(len(previous), len(current))):
                value = current[i] if i < len(current) else ''
                if i >= len(previous) or previous[i] != value:
                    sys.stdout.write(f'\x1b[{i+1};1H{value}\x1b[K')
            sys.stdout.flush()
            previous = current
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write('\x1b[?25h\x1b[?1049l')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
