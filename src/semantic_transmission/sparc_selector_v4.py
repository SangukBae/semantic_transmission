"""Standalone v4 preparation/training/evaluation. Never dispatch through a v3 run."""
import argparse
from contextlib import ExitStack
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

from .sparc_selector_data import read, save, sha, digest
from .sparc_selector_workers import REPO
from . import sparc_selector_distributed as distributed
from . import sparc_selector_train as supervision
from . import sparc_selector_v3_data as old_data
from . import sparc_selector_v4_data as data
from . import sparc_selector_train_v4 as trainer
from .sparc_selector_risk import ARCHITECTURE

MODULE = 'semantic_transmission.sparc_selector_v4'
DEFAULT = REPO/'outputs/sparc_selector_risk_v4_20261022'
PARENT = REPO/'outputs/sparc_selector_budget_v3_20261022'
CONFIG = REPO/'configs/sparc_selector_v4.json'
CODE = ['src/semantic_transmission/'+name+'.py' for name in (
    'sparc_selector_risk', 'sparc_selector_search_v4', 'sparc_selector_v4_data',
    'sparc_selector_train_v4', 'sparc_selector_v4', 'sparc_selector_compare_v4', 'sparc_selector_budget',
    'sparc_selector_train_v3', 'sparc_selector_v3_data', 'sparc_selector_cnn')]


def validate_config(p):
    if (p['schema'] != 'sparc_risk_training_v4' or p['automatic_default_replacement']
            or p['budget_scope'] != 'aggregate_rgb_source_samples' or p['target_cbr'] <= 0
            or not p['training_budgets'] or min(p['training_budgets']) <= 0):
        raise ValueError('invalid v4 training configuration')
    search, cal, train = p['search'], p['calibration'], p['train']
    if (not 0 < cal['alpha'] < 1 or math.ceil((cal['clips']+1)*(1-cal['alpha'])) > cal['clips']
            or not 0 < search['initial_per_clip'] <= search['measurements_per_clip']
            or search['per_round'] < 1 or search['beam'] < 3 or search['risky_max_gap'] < 2
            or search['risk_ratio'] <= 1 or not search['budget_multipliers'] or min(search['budget_multipliers']) <= 0
            or min(search['validation_clips'],search['test_clips']) < 1):
        raise ValueError('invalid calibration or search budget')
    if (not 0 < p['loss']['local_tail_fraction'] <= 1 or min(p['loss'].values()) < 0
            or min(train[k] for k in ('max_epochs','patience','validation_every','states_per_clip','accumulate')) < 1
            or train['warmup_epochs'] < 0 or train['lr'] <= 0 or train['weight_decay'] < 0):
        raise ValueError('invalid training loss or schedule')


def initialize(root, parent, source_root=None, config=CONFIG, runtime_file=None, *, write=True):
    cfg, old = read(config), read(parent/'protocol.json')
    validate_config(cfg)
    result, checkpoint = read(parent/'RESULT.json'), parent/'training/best.pt'
    if sha(checkpoint) != result['checkpoint_sha256'] or result['status'] != 'TRAINED_RD_EVALUATED':
        raise ValueError('completed immutable v3 parent required')
    source_root = (source_root or Path(old['parent'])).resolve()
    if root.resolve() in (parent.resolve(), source_root) or root.resolve() == REPO:
        raise ValueError('v4 output must be separate from parent and source evidence')
    rows = read(parent/'sources.json')
    if sha(source_root/'sources.json') != old['source_manifest_sha256']:
        raise ValueError('parent source manifest changed')
    original_rows = {r['id']:r for r in read(source_root/'sources.json')}
    if any(original_rows.get(r['id']) != r for r in rows):
        raise ValueError('v3 prepared subset differs from original source manifest')
    for name, expected in old['supervision_protocol']['code'].items():
        if sha(REPO/'src/semantic_transmission'/f'{name}.py') != expected:
            raise ValueError('frozen codec/channel/receiver code changed: '+name)
    for path, expected in old['supervision_protocol']['assets'].items():
        local = REPO/'.local'/path.split('/.local/',1)[1]
        if sha(local) != expected:
            raise ValueError('frozen codec/receiver weights changed: '+str(local))
    from .sparc_uf_transport import UPSTREAM
    vendor = REPO/'.local/dcvc_uf_vendor'
    if (subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip() != UPSTREAM
            or subprocess.check_output(['git','-C',str(vendor),'diff','HEAD'])):
        raise ValueError('UF vendor identity changed')
    if sha(old_data.backbone_weights()) != old['features']['weights_sha256']:
        raise ValueError('feature backbone weights changed')
    runtime_file = runtime_file or source_root/'.distributed/runtimes.json'
    if runtime_file.exists():
        runtime = read(runtime_file)
    else:
        runtime = supervision.runtimes()
    required = ('encoder','decoder','receiver','quality','channel')
    if any(k not in runtime or not Path(runtime[k]).is_file() for k in required):
        raise ValueError('worker runtime missing; supply --runtimes JSON with encoder/decoder/receiver/quality/channel')
    p = copy.deepcopy(cfg)
    p.update(architecture=ARCHITECTURE, parent=str(parent.resolve()), source_root=str(source_root),
        source_manifest_sha256=sha(parent/'sources.json'), supervision_protocol=old['supervision_protocol'],
        features=old['features'], initialization=dict(path=str(checkpoint),sha256=sha(checkpoint)),
        runtimes=runtime, code={name:sha(REPO/name) for name in CODE}, **data.split_rows(rows,cfg))
    needed = [r for r in rows if r['split']=='train' or r['id'] in p['calibration_ids']]
    for row in needed:
        for name in ('prepared.json','features.npz','inputs.json','base_observations.json'):
            if not (parent/'clips'/row['id']/name).is_file():
                raise ValueError('missing parent evidence: '+row['id']+'/'+name)
        if not (source_root/'clips'/row['id']/'prepared.json').is_file():
            raise ValueError('missing local source evidence: '+row['id'])
    p['signature'] = digest(p)
    if (root/'protocol.json').exists() and read(root/'protocol.json') != p:
        raise ValueError('v4 protocol changed; use a new output root')
    if write:
        root.mkdir(parents=True,exist_ok=True)
        save(root/'protocol.json',p); save(root/'sources.json',rows)
    return p, rows


def progress(root, stage, done=None, total=None, **extra):
    old = read(root/'status.json') if (root/'status.json').exists() else {}
    now = time.time()
    same = old.get('stage') == stage
    value = dict(status='RUNNING',stage=stage,done=done,total=total,pid=os.getpid(),
        start_ticks=distributed.proc_identity(os.getpid()), started_unix=old.get('started_unix',now),
        stage_started_unix=old.get('stage_started_unix',now) if same else now, updated_unix=now,
        resume='immutable completed labels; full optimizer/scheduler/RNG/order checkpoint')
    value.update(extra)
    save(root/'status.json', value)


def run_options(args):
    return dict({k:str(getattr(args,k)) if getattr(args,k) is not None else None
                 for k in ('parent','source_root','config','runtimes')}, labels_only=args.labels_only)


def label_summary(root, rows, p):
    selected = [r for r in rows if r['split']=='train' or r['id'] in p['calibration_ids']]
    receipts = {r['id']:data.verify_cache(root/'clips'/r['id'],p) for r in selected}
    result = dict(status='LABELS_PREPARED', protocol=p['signature'], clips=len(selected),
        train_clips=sum(r['split']=='train' for r in selected), calibration_clips=len(p['calibration_ids']),
        candidates=sum(r['measurements'] for r in receipts.values()),
        data_hash=digest({key:sha(root/'clips'/key/'risk_inputs.json') for key in receipts}),
        training_started=False)
    save(root/'label_summary.json',result)
    return result


def run(args):
    root = args.root
    root.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM, supervision.stop); signal.signal(signal.SIGINT, supervision.stop)
    with ExitStack() as stack:
        lock = stack.enter_context((root/'run.lock').open('a'))
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            p, rows = initialize(root,args.parent,args.source_root,args.config,args.runtimes)
            save(root/'run_options.json', run_options(args))
            supervision.runtimes = lambda:p['runtimes']
            gpu = stack.enter_context((REPO/'.local/etri_60s_check.lock').open('a'))
            while True:
                supervision.interrupted()
                try:
                    fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB); break
                except BlockingIOError:
                    progress(root,'waiting_for_gpu'); time.sleep(2)
            callback = lambda stage,done=None,total=None,**kw:progress(root,stage,done,total,**kw)
            data.prepare(root,args.parent,Path(p['source_root']),rows,p,callback)
            labels = label_summary(root,rows,p)
            if args.labels_only:
                progress(root,'labels_complete',labels['clips'],labels['clips'],samples=labels['candidates'])
                value=read(root/'status.json')
                value.update(status=labels['status'],exit_code=0,ended_unix=time.time())
                save(root/'status.json',value)
                return labels
            trained = trainer.train(root,Path(p['source_root']),rows,p,callback,max_steps=args.max_steps,
                                    initial_checkpoint=Path(p['initialization']['path']))
            if trained['status']=='INTERRUPTED':
                value=read(root/'status.json');value.update(status='INTERRUPTED',exit_code=130)
                save(root/'status.json',value);return trained
            tested = trainer.evaluate(root,Path(p['source_root']),rows,p,Path(trained['checkpoint']),'test','final',callback)
            result = dict(trained,status='V4_TRAINED_RD_EVALUATED',heldout=tested,target_met=tested['target_met'],
                performance_improvement_proven=False, automatic_default_replacement=False,
                evidence_scope='held-out quality/budget feasibility; superiority requires matched v3 and rule-based controls')
            save(root/'RESULT.json',result)
            progress(root,'complete');value=read(root/'status.json');value.update(status=result['status'],exit_code=0)
            save(root/'status.json',value);return result
        except BaseException as error:
            value=read(root/'status.json') if (root/'status.json').exists() else {}
            interrupted=supervision.STOP or isinstance(error,InterruptedError)
            value.update(status='INTERRUPTED' if interrupted else 'FAILED',error=repr(error),exit_code=130 if interrupted else 1)
            save(root/'status.json',value);raise


def launch(args):
    args.root.mkdir(parents=True,exist_ok=True)
    with (args.root/'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status='ALREADY_RUNNING')
        if (args.root/'launcher.json').exists() and distributed.active(read(args.root/'launcher.json')):
            return dict(status='ALREADY_RUNNING')
        initialize(args.root,args.parent,args.source_root,args.config,args.runtimes)
        save(args.root/'run_options.json', run_options(args))
        if (args.root/'RESULT.json').exists():
            return dict(status=read(args.root/'RESULT.json')['status'])
        command=[sys.executable,'-m',MODULE,'run','--root',str(args.root),'--parent',str(args.parent),'--config',str(args.config)]
        command.append('--labels-only' if args.labels_only else '--no-labels-only')
        for flag in ('source_root','runtimes','max_steps'):
            if getattr(args,flag) is not None:
                command += ['--'+flag.replace('_','-'), str(getattr(args,flag))]
        env=dict(os.environ,PYTHONPATH=str(REPO/'src'),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
            PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with (args.root/'run.log').open('ab') as log:
            process=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
        save(args.root/'launcher.json',dict(pid=process.pid,start_ticks=distributed.proc_identity(process.pid)))
    return dict(status='LAUNCHED',pid=process.pid,root=str(args.root))


def stop(root):
    path=root/'launcher.json'
    if not path.exists() or not distributed.active(read(path)):
        return dict(status='NOT_RUNNING')
    record=read(path)
    command=Path(f'/proc/{record["pid"]}/cmdline').read_bytes().split(b'\0')
    if MODULE.encode() not in command or str(root).encode() not in command or b'run' not in command:
        raise ValueError('process identity mismatch; signal not sent')
    os.kill(record['pid'],signal.SIGTERM)
    return dict(status='STOP_REQUESTED',pid=record['pid'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','run','launch','stop'])
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--runtimes',type=Path)
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--labels-only',action=argparse.BooleanOptionalAction,default=None,
                        help='stop after verified label preparation; resume retains this choice')
    args=parser.parse_args()
    options_path=args.root/'run_options.json'
    options=read(options_path) if options_path.exists() else {}
    if args.labels_only is None:
        args.labels_only=options.get('labels_only',False)
    defaults=dict(parent=PARENT,config=CONFIG,source_root=None,runtimes=None)
    for key,default in defaults.items():
        if getattr(args,key) is None:
            value=options.get(key) or default
            setattr(args,key,Path(value) if value is not None else None)
    for name in ('root','parent','source_root','config','runtimes'):
        if getattr(args,name) is not None:
            setattr(args,name,getattr(args,name).resolve())
    if args.max_steps is not None and args.max_steps < 1:
        parser.error('--max-steps must be positive')
    if args.command=='check':
        p,rows=initialize(args.root,args.parent,args.source_root,args.config,args.runtimes,write=False)
        result=dict(status='PREFLIGHT_PASSED',rows=len(rows),protocol=p['signature'],training_started=False)
    elif args.command=='stop':result=stop(args.root)
    elif args.command=='launch':result=launch(args)
    else:result=run(args)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
