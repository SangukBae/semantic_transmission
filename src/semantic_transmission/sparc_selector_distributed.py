"""Two-host label acquisition around the unchanged, frozen selector protocol.

Only complete, verified clip receipts cross hosts. The original run remains the
training authority; local workers never change its source manifest or protocol.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time

from .sparc_selector_data import digest, read, save, sha

REPO = Path(__file__).resolve().parents[2]
MODULE = 'semantic_transmission.sparc_selector_distributed'
STOP = False
CHANNEL_FAILURE = 'channel integrity failure; no oracle replacement permitted'


def stop(*_):
    global STOP
    STOP = True


def identifier(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError('unsafe clip identifier')
    return value


def checked_plan(path):
    p = read(path)
    unsigned = {k: v for k, v in p.items() if k != 'signature'}
    if digest(unsigned) != p['signature']:
        raise ValueError('assignment plan checksum mismatch')
    groups = [p['completed'], p['assignments']['remote'], p['assignments']['local']]
    flat = [identifier(i) for group in groups for i in group]
    if len(flat) != len(set(flat)) or set(flat) != set(p['prepared']):
        raise ValueError('assignments overlap or fail to cover prepared clips')
    return p


def partition(items, forced_remote=()):
    """Balance remaining source frames, preserving partially measured clips."""
    forced = set(forced_remote)
    assigned = {'local': [], 'remote': []}
    weight = {'local': 0, 'remote': 0}
    for clip, frames in items:
        if clip in forced:
            assigned['remote'].append(clip)
            weight['remote'] += frames
    for clip, frames in sorted(items, key=lambda x: (-x[1], x[0])):
        if clip in forced:
            continue
        role = min(weight, key=lambda k: (weight[k], len(assigned[k]), k))
        assigned[role].append(clip)
        weight[role] += frames
    # Retain frozen source order within each worker for deterministic progress.
    order = {clip: i for i, (clip, _) in enumerate(items)}
    for values in assigned.values():
        values.sort(key=order.__getitem__)
    return assigned, weight


def create_plan(root, target):
    root, target = Path(root), Path(target)
    if target.exists():
        return checked_plan(target)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        p = read(root/'protocol.json')
        rows = read(root/'sources.json')
        complete, prepared, items, partial = [], [], [], []
        baseline = {}
        for row in rows:
            clip = identifier(row['id'])
            folder = root/'clips'/clip
            if not (folder/'prepared.json').exists():
                continue
            prepared.append(clip)
            if (folder/'labels.json').exists():
                receipt = verify_clip(folder, row, p['supervision_protocol'], full=False)
                complete.append(clip)
                baseline[clip] = sha(folder/'labels.json')
            else:
                ready = read(folder/'prepared.json')
                items.append((clip, ready['source']['video']['frames']))
                if any((folder/'observations').glob('*')):
                    partial.append(clip)
        assigned, weight = partition(items, partial)
        plan = dict(schema='sparc_two_host_labels_v1', created_unix=time.time(),
            orchestrator_sha256=sha(Path(__file__)),
            source_manifest_sha256=sha(root/'sources.json'),
            protocol_sha256=sha(root/'protocol.json'),
            supervision_signature=p['supervision_protocol']['signature'],
            original_root=str(root.resolve()), prepared=prepared, completed=complete,
            baseline_receipts=baseline, assignments=assigned, source_frame_work=weight,
            partial_clips_retained_remote=partial,
            balance_basis='remaining source frames; same candidate generation protocol',
            channel_failure_policy='retain reproducible failed packet; exclude entire clip from targets; report separately',
            maximum_rejected_fraction_for_automatic_training=0.01,
            no_label_definition_change=True)
        plan['signature'] = digest(plan)
        save(target, plan)
    return checked_plan(target)


def verify_clip(folder, row, protocol, *, full=True):
    """Verify receipt, arrays and all referenced wire/video evidence before import."""
    import numpy as np
    receipt = read(folder/'labels.json')
    if receipt.get('input') != row or receipt.get('protocol') != protocol['signature']:
        raise ValueError('label source/protocol mismatch: '+row['id'])
    if sha(folder/'labels.npz') != receipt['sha256']:
        raise ValueError('label checksum mismatch: '+row['id'])
    count = receipt['samples']
    if len(receipt['records']) != count:
        raise ValueError('label record count mismatch')
    with np.load(folder/'labels.npz', allow_pickle=False) as data:
        if any(len(data[k]) != count or not np.isfinite(data[k]).all() for k in ('x', 'y')):
            raise ValueError('nonfinite or incomplete labels')
        if count and data['y'].shape != (count, 5):
            raise ValueError('unexpected target shape')
    if not full:
        return receipt
    observations = set()
    for record in receipt['records']:
        observations.update(record[k] for k in ('before_observation', 'after_observation'))
    for name in observations:
        if not re.fullmatch(r'[0-9a-f]{24}', name):
            raise ValueError('invalid observation name')
        obs = folder/'observations'/name
        value = read(obs/'result.json')
        identity = value['identity']
        if digest(identity)[:24] != name or identity['protocol'] != protocol['signature']:
            raise ValueError('observation identity mismatch')
        if set(value['artifacts']) != {'sent.bin', 'received.bin', 'delivered.mp4'}:
            raise ValueError('incomplete observation evidence')
        for filename, expected in value['artifacts'].items():
            if sha(obs/filename) != expected:
                raise ValueError('observation artifact checksum mismatch: '+filename)
        if (value['channel']['bit_errors'] != 0 or
                value['sender']['complex_channel_uses'] != value['channel']['complex_channel_uses'] or
                value['receiver']['source_pixels_at_receiver'] is not False):
            raise ValueError('invalid channel/receiver evidence')
        if value['artifacts']['sent.bin'] != value['artifacts']['received.bin']:
            raise ValueError('packet roundtrip mismatch')
        if any(not math.isfinite(value['quality'][k]) for k in
               ('mse', 'lpips_vgg', 'temporal_mse', 'local_mae95', 'psnr_db')):
            raise ValueError('nonfinite measured quality')
    return receipt


def relocate(root):
    """Keep original metadata, changing only the local immutable-frame location."""
    root = Path(root).resolve()
    for folder in (root/'clips').iterdir():
        current = folder/'prepared.json'
        if not current.exists():
            continue
        origin = folder/'prepared.remote.json'
        if not origin.exists():
            origin.write_bytes(current.read_bytes())
        value = read(origin)
        value['source']['directory'] = str(folder/'frames')
        save(current, value)


def record_channel_rejection(root, clip, row, protocol, runtime, error):
    """Never repair payloads or change the channel seed to obtain a usable label."""
    folder = root/'clips'/clip
    pending = [f for f in (folder/'observations').iterdir()
               if (f/'sent.bin').exists() and not (f/'result.json').exists()]
    if len(pending) != 1:
        raise ValueError('cannot unambiguously identify failed channel observation')
    observation = pending[0]
    script = '''from pathlib import Path
import json,sys
from semantic_transmission.metadata_channel import transmit
packet=Path(sys.argv[1]).read_bytes()
received,report=transmit(packet,10.,42,bits_per_symbol=4)
Path(sys.argv[2]).write_bytes(received)
print(json.dumps(report))
'''
    env = dict(os.environ, PYTHONPATH=str(REPO/'src'), PYTHONNOUSERSITE='1',
               CUDA_VISIBLE_DEVICES='-1', TF_CPP_MIN_LOG_LEVEL='3',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    with (root/'channel_rejection_probe.log').open('ab') as log:
        report = json.loads(subprocess.check_output([runtime, '-c', script,
            str(observation/'sent.bin'), str(observation/'failed_received.bin')],
            env=env, stderr=log, text=True, timeout=180))
    if report['bit_errors'] <= 0:
        raise ValueError('channel failure did not reproduce; stop for investigation')
    receipt = dict(status='REJECTED_CHANNEL', input=row, protocol=protocol['signature'],
        observation=observation.name, error=str(error), report=report,
        artifacts={k:sha(observation/k) for k in ('sent.bin', 'failed_received.bin')},
        use_for_training=False, created_unix=time.time())
    save(folder/'label_failure.json', receipt)
    verify_rejection(folder, row, protocol)
    return receipt


def verify_rejection(folder, row, protocol):
    import numpy as np
    r = read(folder/'label_failure.json')
    if r['input'] != row or r['protocol'] != protocol['signature'] or r['use_for_training'] is not False:
        raise ValueError('channel rejection source/protocol mismatch')
    if r['status'] != 'REJECTED_CHANNEL' or not re.fullmatch(r'[0-9a-f]{24}', r['observation']):
        raise ValueError('invalid rejection observation')
    if (folder/'labels.json').exists():
        raise ValueError('clip cannot contain both targets and a channel rejection')
    if set(r['artifacts']) != {'sent.bin', 'failed_received.bin'}:
        raise ValueError('missing channel rejection evidence')
    obs = folder/'observations'/r['observation']
    for name, expected in r['artifacts'].items():
        if sha(obs/name) != expected:
            raise ValueError('channel rejection evidence changed')
    sent = np.frombuffer((obs/'sent.bin').read_bytes(), np.uint8)
    received = np.frombuffer((obs/'failed_received.bin').read_bytes(), np.uint8)
    if sent.shape != received.shape:
        raise ValueError('invalid rejected packet length')
    errors = int(np.unpackbits(np.bitwise_xor(sent, received)).sum())
    report = r['report']
    if (errors <= 0 or errors != report['bit_errors'] or report['snr_db'] != protocol['snr_db'] or
            report['seed'] != protocol['channel_seed'] or report['bits_per_symbol'] != 4 or
            report['ldpc_k'] != 6144 or report['ldpc_n'] != 9216):
        raise ValueError('unverified channel rejection')
    return r


def validate_inputs(root, plan, role):
    if sha(Path(__file__)) != plan['orchestrator_sha256']:
        raise ValueError('distributed execution code changed')
    p = read(root/'protocol.json')
    if sha(root/'protocol.json') != plan['protocol_sha256'] or sha(root/'sources.json') != plan['source_manifest_sha256']:
        raise ValueError('frozen source or protocol changed')
    supervision = p['supervision_protocol']
    for name, expected in supervision['code'].items():
        if sha(REPO/'src/semantic_transmission'/f'{name}.py') != expected:
            raise ValueError('frozen supervision code changed: '+name)
    for name, expected in p['code'].items():
        if sha(REPO/name) != expected:
            raise ValueError('frozen predictor code changed: '+name)
    for name, expected in supervision['assets'].items():
        relative = Path(name).relative_to(Path(plan['original_root']).parents[1])
        if sha(REPO/relative) != expected:
            raise ValueError('frozen model changed: '+name)
    from .sparc_uf_transport import UPSTREAM
    vendor = REPO/'.local/dcvc_uf_vendor'
    if subprocess.check_output(['git', '-C', str(vendor), 'rev-parse', 'HEAD'], text=True).strip() != UPSTREAM:
        raise ValueError('UF vendor revision changed')
    if subprocess.check_output(['git', '-C', str(vendor), 'diff', 'HEAD']):
        raise ValueError('UF vendor source changed')
    rows = read(root/'sources.json')
    by_id = {row['id']: row for row in rows}
    for clip in plan['assignments'][role]:
        ready = read(root/'clips'/clip/'prepared.json')
        if ready['input'] != by_id[clip] or ready['protocol'] != supervision['signature']:
            raise ValueError('prepared source identity changed: '+clip)
        directory = Path(ready['source']['directory'])
        if directory.resolve() != (root/'clips'/clip/'frames').resolve():
            raise ValueError('prepared frames point outside this worker')
        if role == 'local':
            original = read(root/'clips'/clip/'prepared.remote.json')
            original['source']['directory'] = str(directory)
            if original != ready:
                raise ValueError('prepared metadata changed beyond relocation')
        if any(sha(directory/name) != expected for name, expected in ready['source']['hashes'].items()):
            raise ValueError('prepared frame checksum mismatch: '+clip)
    return p, rows


def proc_identity(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def active(record):
    pid = record.get('pid')
    return bool(pid and record.get('start_ticks') and proc_identity(pid) == record['start_ticks'])


def lock_file(stack, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = stack.enter_context(path.open('a'))
    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


def worker(root, plan_path, role, runtimes):
    from . import sparc_selector_train as supervision
    root = Path(root).resolve()
    plan = checked_plan(plan_path)
    state_path = root/'.distributed'/f'worker_{role}.json'
    began = time.time()
    def report(state, **extra):
        save(state_path, dict(status=state, role=role, pid=os.getpid(),
            start_ticks=proc_identity(os.getpid()), started_unix=began,
            updated_unix=time.time(), plan=plan['signature'], **extra))
    with ExitStack() as stack:
        lock_file(stack, root/'run.lock')
        lock_file(stack, REPO/'.local/etri_60s_check.lock')
        if role == 'remote':
            lock_file(stack, Path(read(root/'protocol.json')['parent'])/'run.lock')
        try:
            report('VERIFYING', done=0, total=len(plan['assignments'][role]))
            p, rows = validate_inputs(root, plan, role)
            by_id = {r['id']: r for r in rows}
            selected = [by_id[k] for k in plan['assignments'][role]]
            if runtimes:
                runtime_map = read(runtimes)
                supervision.runtimes = lambda: runtime_map
            original_status = supervision.status
            def progress(target, stage, done=None, total=None, **extra):
                original_status(target, stage, done, total, **extra)
                report('RUNNING', stage=stage, done=done, total=total, **extra)
            supervision.status = progress
            def request_stop(*_):
                stop()
                supervision.stop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, request_stop)
            rejected = []
            for row in selected:
                if (root/'clips'/row['id']/'label_failure.json').exists():
                    verify_rejection(root/'clips'/row['id'], row, p['supervision_protocol'])
                    rejected.append(row['id'])
            while True:
                remaining_rows = [r for r in selected if r['id'] not in rejected]
                try:
                    supervision.label(root, remaining_rows, p['supervision_protocol'])
                    break
                except RuntimeError as error:
                    if CHANNEL_FAILURE not in str(error):
                        raise
                    clip = read(root/'status.json')['clip']
                    if clip not in {r['id'] for r in remaining_rows}:
                        raise ValueError('failed clip is outside this worker assignment') from error
                    record_channel_rejection(root, clip, by_id[clip], p['supervision_protocol'],
                                             supervision.runtimes()['channel'], error)
                    rejected.append(clip)
            report('LABELS_COMPLETE', done=len(selected), total=len(selected),
                   rejected=len(rejected), exit_code=0)
            if role == 'local':
                return
            while True:
                if STOP:
                    raise InterruptedError('stop requested while waiting for local imports')
                imported = read(root/'.distributed/imports.json') if (root/'.distributed/imports.json').exists() else {}
                remaining = [k for k in plan['assignments']['local']
                             if imported.get(k, {}).get('plan') != plan['signature']]
                if not remaining:
                    break
                report('WAITING_FOR_LOCAL', done=len(selected), total=len(selected),
                       local_imported=len(imported), local_total=len(plan['assignments']['local']))
                time.sleep(10)
            all_rejected = []
            for clip in plan['prepared']:
                if (root/'clips'/clip/'label_failure.json').exists():
                    verify_rejection(root/'clips'/clip, by_id[clip], p['supervision_protocol'])
                    all_rejected.append(clip)
                else:
                    verify_clip(root/'clips'/clip, by_id[clip], p['supervision_protocol'], full=False)
            for clip, expected in plan['baseline_receipts'].items():
                if sha(root/'clips'/clip/'labels.json') != expected:
                    raise ValueError('pre-existing completed receipt changed')
            all_rows = [by_id[k] for k in plan['prepared'] if k not in all_rejected]
            save(root/'label_summary.json', dict(clips=len(all_rows),
                samples=sum(read(root/'clips'/r['id']/'labels.json')['samples'] for r in all_rows),
                simulated_channel=True, actual_entropy_coding=True, distributed_plan=plan['signature'],
                channel_rejected_clips=all_rejected, labels_conditioned_on_successful_channel=True))
            if len(all_rejected)/len(plan['prepared']) > plan['maximum_rejected_fraction_for_automatic_training']:
                raise ValueError('channel rejection fraction exceeds 1 percent; all acquisition results retained, training requires review')
            from . import sparc_selector_train_v2 as v2
            report('TRAINING', done=len(selected), total=len(selected))
            v2.build_inputs(root, all_rows)
            supervision.interrupted()
            result = v2.train(root, p, all_rows)
            value = read(root/'status.json')
            value.update(status=result['status'], result=result, updated_unix=time.time())
            save(root/'status.json', value)
            report(result['status'], result=result, exit_code=0)
        except BaseException as error:
            state = 'INTERRUPTED' if STOP or isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'FAILED'
            report(state, error=repr(error), exit_code=130 if state == 'INTERRUPTED' else 1)
            value = read(root/'status.json') if (root/'status.json').exists() else {}
            value.update(status=state, error=repr(error), updated_unix=time.time())
            save(root/'status.json', value)
            raise


def import_clips(root, plan_path, clips):
    root = Path(root).resolve()
    plan = checked_plan(plan_path)
    if (sha(root/'sources.json') != plan['source_manifest_sha256'] or
            sha(root/'protocol.json') != plan['protocol_sha256'] or
            sha(Path(__file__)) != plan['orchestrator_sha256']):
        raise ValueError('import source/protocol/execution identity changed')
    rows = {r['id']: r for r in read(root/'sources.json')}
    protocol = read(root/'protocol.json')['supervision_protocol']
    with ExitStack() as stack:
        lock_file(stack, root/'.distributed/import.lock')
        path = root/'.distributed/imports.json'
        imported = read(path) if path.exists() else {}
        for clip in clips:
            identifier(clip)
            if clip not in plan['assignments']['local']:
                raise ValueError('refusing import outside local assignment')
            source = root/'.distributed/incoming/clips'/clip
            target = root/'clips'/clip
            if (source/'label_failure.json').exists():
                rejection = verify_rejection(source, rows[clip], protocol)
                target.mkdir(parents=True, exist_ok=True)
                if (target/'labels.json').exists():
                    raise ValueError('refusing to replace valid labels with a rejection')
                dest = target/'observations'/rejection['observation']
                dest.mkdir(parents=True, exist_ok=True)
                for name in rejection['artifacts']:
                    incoming = source/'observations'/rejection['observation']/name
                    if (dest/name).exists() and sha(dest/name) != rejection['artifacts'][name]:
                        raise ValueError('conflicting rejection artifact')
                    (dest/name).write_bytes(incoming.read_bytes())
                save(target/'label_failure.json', rejection)
                verify_rejection(target, rows[clip], protocol)
                imported[clip] = dict(plan=plan['signature'], status='REJECTED_CHANNEL',
                    receipt_sha256=sha(target/'label_failure.json'), verified_unix=time.time())
                save(path, imported)
                continue
            receipt = verify_clip(source, rows[clip], protocol)
            if (target/'labels.json').exists():
                old = verify_clip(target, rows[clip], protocol)
                if old != receipt:
                    raise ValueError('refusing to replace existing different labels')
            else:
                target.mkdir(parents=True, exist_ok=True)
                observations = source/'observations'
                if observations.exists():
                    (target/'observations').mkdir(exist_ok=True)
                    for obs in observations.iterdir():
                        if not re.fullmatch(r'[0-9a-f]{24}', obs.name):
                            raise ValueError('invalid incoming observation')
                        dest = target/'observations'/obs.name
                        if dest.exists():
                            # Resume a publication interrupted before the receipt commit.
                            if read(dest/'result.json') != read(obs/'result.json'):
                                raise ValueError('conflicting observation cache')
                            r = read(dest/'result.json')
                            if any(sha(dest/k) != h for k, h in r['artifacts'].items()):
                                raise ValueError('existing observation damaged')
                        else:
                            os.rename(obs, dest)
                temporary = target/'labels.npz.importing'
                temporary.write_bytes((source/'labels.npz').read_bytes())
                os.replace(temporary, target/'labels.npz')
                # Receipt is the commit marker and is always published last.
                save(target/'labels.json', receipt)
                verify_clip(target, rows[clip], protocol)
            imported[clip] = dict(plan=plan['signature'], labels_sha256=receipt['sha256'],
                                  receipt_sha256=sha(target/'labels.json'), verified_unix=time.time())
            save(path, imported)
    return dict(imported=len(imported), plan=plan['signature'])


def gpu():
    try:
        return subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'], text=True, timeout=4).strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        return 'unavailable'


def snapshot(root, plan_path, role):
    root = Path(root)
    plan = checked_plan(plan_path)
    path = root/'.distributed'/f'worker_{role}.json'
    state = read(path) if path.exists() else dict(status='NOT_STARTED')
    clips = plan['assignments'][role]
    receipts = [root/'clips'/k/'labels.json' for k in clips]
    completed = [p for p in receipts if p.exists()]
    rejected = sum((root/'clips'/k/'label_failure.json').exists() for k in clips)
    value = dict(state, alive=active(state), observed_unix=time.time(),
        done=len(completed)+rejected, valid=len(completed), rejected=rejected,
        total=len(clips), samples=sum(read(p)['samples'] for p in completed), gpu=gpu())
    if role == 'remote':
        value['global_valid'] = sum((root/'clips'/k/'labels.json').exists() for k in plan['prepared'])
        value['global_rejected'] = sum((root/'clips'/k/'label_failure.json').exists() for k in plan['prepared'])
        value['global_done'] = value['global_valid']+value['global_rejected']
        value['global_total'] = len(plan['prepared'])
        path = root/'.distributed/imports.json'
        value['imported'] = len(read(path)) if path.exists() else 0
        value['training'] = read(root/'status.json') if (root/'status.json').exists() else {}
    return value


def remote_call(config, arguments, *, timeout=120):
    command = shlex.join(['env', 'PYTHONNOUSERSITE=1', 'PYTHONDONTWRITEBYTECODE=1',
        'PYTHONPATH='+config['remote_repo']+'/src', config['remote_python'], '-m', MODULE, *arguments])
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
        '-o', 'ConnectTimeout=10', config['host'], command], text=True, timeout=timeout)


def launch(root, plan, role, runtimes=None):
    root = Path(root).resolve()
    checked_plan(plan)
    folder = root/'.distributed'
    folder.mkdir(parents=True, exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status='ALREADY_RUNNING', role=role)
    command = [sys.executable, '-m', MODULE, 'run', '--root', str(root),
               '--plan', str(Path(plan).resolve()), '--role', role]
    if runtimes:
        command += ['--runtimes', str(Path(runtimes).resolve())]
    env = dict(os.environ, PYTHONPATH=str(REPO/'src'), PYTHONNOUSERSITE='1',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (folder/f'{role}.log').open('ab') as log:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return dict(status='LAUNCHED', pid=process.pid, role=role)


def stop_worker(root, role):
    path = Path(root)/'.distributed'/f'worker_{role}.json'
    if not path.exists():
        return dict(status='NOT_STARTED')
    record = read(path)
    if not active(record):
        return dict(status='NOT_RUNNING')
    args = Path(f'/proc/{record["pid"]}/cmdline').read_bytes().split(b'\0')
    if MODULE.encode() not in args or str(Path(root).resolve()).encode() not in args or b'run' not in args:
        raise ValueError('worker process identity mismatch; no signal sent')
    os.kill(record['pid'], signal.SIGTERM)
    return dict(status='STOP_REQUESTED', pid=record['pid'])


def synchronize(config_path):
    c = read(config_path)
    root = Path(c['local_root'])
    plan = checked_plan(c['local_plan'])
    path = root/'.distributed/sync.json'
    with ExitStack() as stack:
        lock_file(stack, root/'.distributed/sync.lock')
        state = read(path) if path.exists() else {}
        imported = set(state.get('imported_ids', []))
        failures = 0
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, stop)
        while not STOP:
            state.update(status='RUNNING', pid=os.getpid(), start_ticks=proc_identity(os.getpid()),
                updated_unix=time.time(), plan=plan['signature'], imported_ids=sorted(imported),
                total=len(plan['assignments']['local']))
            save(path, state)
            try:
                pending = [k for k in plan['assignments']['local'] if k not in imported
                           and any((root/'clips'/k/name).exists() for name in ('labels.json', 'label_failure.json'))][:8]
                if pending:
                    files = []
                    for clip in pending:
                        folder = root/'clips'/clip
                        if (folder/'label_failure.json').exists():
                            files.append(folder/'label_failure.json')
                        else:
                            files.extend([folder/'labels.json', folder/'labels.npz'])
                        files.extend(f for f in (folder/'observations').rglob('*') if f.is_file())
                    with tempfile.NamedTemporaryFile(dir=root/'.distributed') as listing:
                        listing.write(b''.join(os.fsencode(str(f.relative_to(root)))+b'\0' for f in files))
                        listing.flush()
                        dest = c['host']+':'+c['remote_root']+'/.distributed/incoming/'
                        subprocess.run(['rsync', '-a', '--partial', '--from0', '--files-from='+listing.name,
                            '-e', 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10',
                            str(root)+'/', dest], check=True, timeout=600)
                    remote_call(c, ['import', '--root', c['remote_root'], '--plan', c['remote_plan'],
                        *sum((['--clip', k] for k in pending), [])], timeout=300)
                    imported.update(pending)
                    state.update(imported_ids=sorted(imported), updated_unix=time.time())
                    save(path, state)
                current = json.loads(remote_call(c, ['snapshot', '--root', c['remote_root'],
                    '--plan', c['remote_plan'], '--role', 'remote'], timeout=20))
                save(root/'.distributed/remote_snapshot.json', current)
                failures = 0
                state.pop('last_error', None)
                state['last_success_unix'] = time.time()
                if len(imported) == len(plan['assignments']['local']):
                    state.update(status='COMPLETE', imported_ids=sorted(imported), exit_code=0,
                                 updated_unix=time.time())
                    save(path, state)
                    return
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                failures += 1
                state.update(status='RETRYING', last_error=repr(error), consecutive_failures=failures,
                             updated_unix=time.time())
                save(path, state)
                print(json.dumps(state), flush=True)
            time.sleep(min(30, 5*max(1, failures)))
        state.update(status='INTERRUPTED', imported_ids=sorted(imported), updated_unix=time.time())
        save(path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'relocate', 'run', 'import', 'snapshot', 'sync', 'launch', 'stop'])
    parser.add_argument('--root', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--role', choices=['local', 'remote'])
    parser.add_argument('--runtimes', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--clip', action='append', default=[])
    args = parser.parse_args()
    if args.command == 'plan':
        print(json.dumps(create_plan(args.root, args.plan)))
    elif args.command == 'relocate':
        relocate(args.root)
    elif args.command == 'run':
        worker(args.root, args.plan, args.role, args.runtimes)
    elif args.command == 'import':
        print(json.dumps(import_clips(args.root, args.plan, args.clip)))
    elif args.command == 'snapshot':
        print(json.dumps(snapshot(args.root, args.plan, args.role)))
    elif args.command == 'sync':
        synchronize(args.config)
    elif args.command == 'launch':
        print(json.dumps(launch(args.root, args.plan, args.role, args.runtimes)))
    elif args.command == 'stop':
        print(json.dumps(stop_worker(args.root, args.role)))


if __name__ == '__main__':
    main()
