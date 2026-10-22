"""Local continuation from a sealed v3 checkpoint and measured long-WebVid data."""
import argparse
from contextlib import ExitStack
import copy
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from .sparc_selector_data import read, save, sha, digest
from .sparc_selector_workers import REPO
from . import sparc_selector_train as supervision
from . import sparc_selector_train_v3 as trainer
from . import sparc_selector_v3_data as data
from . import sparc_selector_webvid_search as long_search
from . import sparc_selector_distributed as distributed

MODULE = 'semantic_transmission.sparc_selector_finetune'
DEFAULT = REPO/'outputs/sparc_selector_webvid_finetune_20261022_v2'
LOCAL = REPO/'outputs/sparc_selector_distributed_20261021/local'
LONG = REPO/'outputs/sparc_selector_webvid_long_20261022'
NAMES = ('features.npz', 'prepared.json', 'inputs.json', 'base_observations.json')


def frozen(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('frozen continuation input changed: '+str(path))
    else:
        save(path, value)


def validate_selection(selection, original, dataset):
    lookup = {r['id']: r for r in original}
    all_rows = []
    for name, split in [('replay','train'), ('long','train'), ('val','val'), ('test','test')]:
        for row in selection[name]:
            if row != lookup.get(row['id']) or row['split'] != split:
                raise ValueError('source identity or split changed')
            all_rows.append(row)
    if len({r['id'] for r in all_rows}) != len(all_rows):
        raise ValueError('duplicate continuation clip')
    parents = {s:{r['parent'] for r in all_rows if r['split']==s} for s in ('train','val','test')}
    if any(parents[a]&parents[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('parent leakage between splits')
    if selection['long'] != [e['input'] for e in dataset['clips']] or dataset['split'] != 'train':
        raise ValueError('long training dataset inventory changed')
    return all_rows


def initialize(root):
    original = read(root/'parent/protocol.json')
    result = read(root/'parent/RESULT.json')
    checkpoint = root/'parent/training/best.pt'
    if result['status'] != 'TRAINED_RD_EVALUATED' or sha(checkpoint) != result['checkpoint_sha256']:
        raise ValueError('remote final selected checkpoint is not verified')
    dataset, completed = read(LONG/'dataset.json'), read(LONG/'RESULT.json')
    long_protocol = read(LONG/'protocol.json')
    if (completed['status'] != 'DATASET_COMPLETE' or sha(LONG/'dataset.json') != completed['dataset_sha256']
            or dataset['protocol'] != long_protocol['signature'] or dataset['features'] != original['features']
            or long_protocol['supervision_protocol'] != original['supervision_protocol']):
        raise ValueError('long corpus protocol or completion receipt changed')
    selection = read(root/'selection.json')
    rows = validate_selection(selection, read(root/'parent/sources.json'), dataset)
    for split in ('val','test'):
        expected = data.sample_parents(read(root/'parent/sources.json'), split,
            original['search']['validation_clips' if split=='val' else 'test_clips'], original['seed'])
        if selection[split] != expected:
            raise ValueError('original held-out evaluation cohort changed')
    for name, expected in original['supervision_protocol']['code'].items():
        if sha(REPO/'src/semantic_transmission'/f'{name}.py') != expected:
            raise ValueError('frozen measurement code changed: '+name)
    runtime_path = LOCAL/'.distributed/runtimes.json'
    runtime = read(runtime_path)
    if not all(Path(v).is_file() for v in runtime.values()):
        raise ValueError('local worker runtime missing')
    p = copy.deepcopy(original)
    p.pop('signature')
    p.update(parent=str(LOCAL), continuation='long_webvid_finetune_v1',
        initialization=dict(path=str(checkpoint), sha256=sha(checkpoint), protocol=original['signature'],
            optimizer='new AdamW for the added corpus; exact state retained on subsequent resume',
            normalization='retain parent checkpoint coordinates'),
        source_manifest_sha256=digest(rows), long_dataset_sha256=sha(LONG/'dataset.json'),
        local_runtime_sha256=sha(runtime_path), selection_sha256=sha(root/'selection.json'),
        imported_protocol_sha256=sha(root/'parent/protocol.json'),
        code={f:sha(REPO/f) for f in trainer.CODE + ['src/semantic_transmission/sparc_selector_finetune.py']})
    p['train'].update(max_epochs=20, patience=4, validation_every=2, lr=2e-5, warmup_epochs=2)
    p['signature'] = digest(p)
    frozen(root/'protocol.json', p)
    frozen(root/'sources.json', rows)
    return p, rows, selection, dataset, long_protocol, runtime


def link(source, dest):
    source = source.resolve()
    if dest.is_symlink() or dest.exists():
        if dest.resolve() != source:
            raise ValueError('cache link changed: '+str(dest))
    else:
        dest.symlink_to(source, target_is_directory=source.is_dir())


def prepare(root, p, selection, dataset, long_protocol, progress):
    long_search.validate_dependencies(long_protocol, read(LOCAL/'.distributed/plan.json'))
    # Cached training features are sufficient: held-out source pixels are accessed
    # only by validation/test, and never by an optimizer batch.
    entries = {e['input']['id']:e for e in dataset['clips']}
    training_rows = selection['long'] + selection['replay'] + selection['val']
    states = 0
    for i,row in enumerate(training_rows):
        supervision.interrupted()
        progress('verify_and_import', i, len(training_rows), clip=row['id'])
        extended = row['id'] in entries
        src = (LONG if extended else root/'parent')/'clips'/row['id']
        ready = read(src/'prepared.json')
        local_ready = read(LOCAL/'clips'/row['id']/'prepared.json')
        for key in ('input','protocol','keys','cuts'):
            if ready[key] != local_ready[key]:
                raise ValueError('local/remote prepared identity mismatch')
        for key in ('hashes','video'):
            if ready['source'][key] != local_ready['source'][key]:
                raise ValueError('local/remote source pixels differ')
        inputs = read(src/'inputs.json')
        if (sha(src/'features.npz') != inputs['features_sha256']
                or sha(src/'base_observations.json') != inputs['observations_sha256']):
            raise ValueError('imported feature/measurement checksum mismatch')
        observations = read(src/'base_observations.json')
        if extended:
            entry = entries[row['id']]
            for name, item in entry['files'].items():
                if sha(src/name) != item['sha256']:
                    raise ValueError('long dataset export changed')
            # Verify actual transport and delivered video evidence, not just labels.
            frames, _ = data.load_source(LOCAL/'clips'/row['id'])
            buf = io.BytesIO(); np.save(buf, frames)
            array_hash = hashlib.sha256(buf.getbuffer()).hexdigest()
            del frames, buf
            journal = read(src/'search_state.json')
            if journal['observations'] != observations:
                raise ValueError('export differs from measured search journal')
            long_search.verify_evidence(src, long_protocol, ready, array_hash, journal)
            if len(observations) != entry['measured_states']:
                raise ValueError('long state count changed')
        for obs in observations:
            ident = obs['identity']
            if (ident['protocol'] != p['supervision_protocol']['signature'] or ident['keys'] != ready['keys']
                    or ident['cuts'] != ready['cuts'] or obs['channel']['bit_errors']):
                raise ValueError('imported observation protocol mismatch')
        dst = root/'clips'/row['id']; dst.mkdir(parents=True, exist_ok=True)
        for name in ('features.npz','base_observations.json'):
            link(src/name, dst/name)
        frozen(dst/'prepared.json', local_ready)
        frozen(dst/'inputs.json', dict(identity=dict(origin=str(src), original_inputs_sha256=sha(src/'inputs.json'),
            protocol=p['signature']), features_sha256=inputs['features_sha256'],
            observations_sha256=inputs['observations_sha256']))
        # Only completed immutable validation observations can be reused; new
        # candidate directories are created under this continuation's own root.
        if row['split']=='val' and (src/'observations').exists():
            (dst/'observations').mkdir(exist_ok=True)
            for cached in (src/'observations').iterdir():
                if (cached/'result.json').exists():
                    link(cached, dst/'observations'/cached.name)
        if row['split']=='train':states += len(observations)
    frozen(root/'import_receipt.json', dict(protocol=p['signature'], train_clips=len(selection['replay'])+len(selection['long']),
        replay_clips=len(selection['replay']), long_clips=len(selection['long']), long_states=dataset['states'],
        train_states=states, validation_clips=len(selection['val']), test_clips=len(selection['test']),
        test_targets_imported=False, parent_checkpoint_sha256=p['initialization']['sha256']))


def write_progress(root, stage, done=None, total=None, **extra):
    old = read(root/'status.json') if (root/'status.json').exists() else {}
    now = time.time()
    same = old.get('stage')==stage and (stage!='train_v3' or old.get('epoch')==extra.get('epoch'))
    value = dict(status='RUNNING', stage=stage, done=done, total=total, pid=os.getpid(),
        start_ticks=distributed.proc_identity(os.getpid()), started_unix=old.get('started_unix',now),
        stage_started_unix=old.get('stage_started_unix',now) if same else now,
        epoch_start_batch=old.get('epoch_start_batch',0) if same else extra.get('batch_done',0),
        updated_unix=now, resume='weights/optimizer/scheduler/RNG/sampled order and budget')
    value.update(extra)
    save(root/'status.json', value)


def run(root, max_steps=None):
    with ExitStack() as stack:
        lock = stack.enter_context((root/'run.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX)
        signal.signal(signal.SIGTERM, supervision.stop)
        signal.signal(signal.SIGINT, supervision.stop)
        def progress(stage, done=None, total=None, **extra):
            write_progress(root,stage,done,total,**extra)
        try:
            progress('verify_protocol')
            p, rows, selection, dataset, long_protocol, runtime = initialize(root)
            gpu = stack.enter_context((REPO/'.local/etri_60s_check.lock').open('a'))
            while True:
                supervision.interrupted()
                try:
                    fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB); break
                except BlockingIOError:
                    progress('waiting_for_local_gpu');time.sleep(2)
            supervision.runtimes = lambda: runtime
            torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            save(root/'environment.json', dict(python=sys.version, torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0), runtimes=runtime))
            prepare(root,p,selection,dataset,long_protocol,progress)
            initial = Path(p['initialization']['path'])
            baseline = trainer.evaluate(root,LOCAL,rows,p,initial,'val','initial',progress)
            trained = trainer.train(root,LOCAL,rows,p,progress,max_steps=max_steps,
                initial_checkpoint=initial,initial_validation=baseline)
            if trained['status']=='INTERRUPTED':
                value=read(root/'status.json');value.update(status='INTERRUPTED',exit_code=130,result=trained)
                save(root/'status.json',value);return trained
            tested = trainer.evaluate(root,LOCAL,rows,p,Path(trained['checkpoint']),'test','final',progress)
            best = torch.load(trained['checkpoint'],map_location='cpu',weights_only=False)
            result = dict(trained,status='FINETUNED_RD_EVALUATED',heldout=tested,
                initial_validation=baseline, final_validation=best['validation'],
                retained_parent=best.get('retained_parent',False), target_met=tested['target_met'],
                validation_improved=best['validation']['selection_score'] < baseline['selection_score']-1e-5,
                automatic_default_replacement=False, heldout_scope='same fixed 8 parents as previous run; test once after model selection')
            save(root/'RESULT.json',result)
            progress('complete');value=read(root/'status.json')
            value.update(status=result['status'],exit_code=0,result={k:result[k] for k in ('steps','epochs','checkpoint','retained_parent','validation_improved','target_met')})
            save(root/'status.json',value);return value['result']
        except BaseException as error:
            value=read(root/'status.json') if (root/'status.json').exists() else {}
            interrupted=supervision.STOP or isinstance(error,InterruptedError)
            value.update(status='INTERRUPTED' if interrupted else 'FAILED',error=repr(error),
                         exit_code=130 if interrupted else 1,updated_unix=time.time())
            save(root/'status.json',value);raise


def launch(root):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return dict(status='ALREADY_RUNNING')
        if (root/'launcher.json').exists() and distributed.active(read(root/'launcher.json')):
            return dict(status='ALREADY_RUNNING')
        initialize(root)
        if (root/'RESULT.json').exists():return dict(status=read(root/'RESULT.json')['status'])
        env=dict(os.environ,PYTHONPATH=str(REPO/'src'),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
            PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with (root/'run.log').open('ab') as log:
            proc=subprocess.Popen([sys.executable,'-m',MODULE,'run','--root',str(root)],cwd=REPO,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'launcher.json',dict(pid=proc.pid,start_ticks=distributed.proc_identity(proc.pid)))
    return dict(status='LAUNCHED',pid=proc.pid,output=str(root))


def stop(root):
    record=read(root/'launcher.json')
    if not distributed.active(record):return dict(status='NOT_RUNNING')
    args=Path(f'/proc/{record["pid"]}/cmdline').read_bytes().split(b'\0')
    if MODULE.encode() not in args or b'run' not in args or str(root).encode() not in args:
        raise ValueError('process identity mismatch; no signal sent')
    os.kill(record['pid'],signal.SIGTERM)
    return dict(status='STOP_REQUESTED',pid=record['pid'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','launch','stop','check'])
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--max-steps',type=int)
    args=parser.parse_args();root=args.root.resolve()
    if args.command=='check':
        _,rows,selection,_,_,_=initialize(root)
        result=dict(status='READY',train_clips=len(selection['long'])+len(selection['replay']),rows=len(rows))
    else:result=run(root,args.max_steps) if args.command=='run' else launch(root) if args.command=='launch' else stop(root)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
