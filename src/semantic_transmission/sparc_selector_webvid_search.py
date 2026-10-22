"""Queued local WebVid teacher data: real UF/channel/RIFE, immutable evidence.

This job waits for the local distributed labels and the shared GPU lock. It
exports a separate train-only corpus; an active frozen training run is untouched.
"""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from . import sparc_selector_budget as budget
from . import sparc_selector_distributed as distributed
from . import sparc_selector_train as supervision
from . import sparc_selector_v3_data as data
from .sparc_selector_data import read, save, sha, digest
from .sparc_selector_workers import REPO

MODULE = 'semantic_transmission.sparc_selector_webvid_search'
DEFAULT = REPO/'outputs/sparc_selector_webvid_long_20261022'
DEFAULT_PARENT = REPO/'outputs/sparc_selector_distributed_20261021/local'
CONFIG = REPO/'configs/sparc_selector_webvid_search.json'


def select_rows(rows, cfg):
    parents = defaultdict(list)
    for row in rows:
        if (row['split'] == 'train' and row['dataset'] == 'WebVid-existing'
                and row['duration_seconds'] >= cfg['minimum_seconds']):
            parents[row['parent']].append(row)
    # Cover every parent's longest available view before its second window.
    groups = []
    for parent in sorted(parents):
        unique = {}
        for row in sorted(parents[parent], key=lambda r: (-r['duration_seconds'], r['id'])):
            unique.setdefault((row['start_seconds'], row['duration_seconds']), row)
        groups.append(list(unique.values())[:cfg['windows_per_parent']])
    return [group[i] for i in range(cfg['windows_per_parent']) for group in groups if i < len(group)]


def parent_completion(parent, plan):
    state_path = parent/'.distributed/worker_local.json'
    state = read(state_path) if state_path.exists() else {}
    assigned = plan['assignments']['local']
    done = sum(any((parent/'clips'/clip/name).exists() for name in ('labels.json', 'label_failure.json'))
               for clip in assigned)
    ready = (state.get('status') == 'LABELS_COMPLETE' and state.get('plan') == plan['signature']
             and state.get('exit_code') == 0 and done == len(assigned))
    return ready, dict(parent_status=state.get('status', 'UNKNOWN'), parent_done=done,
                      parent_total=len(assigned), parent_alive=distributed.active(state))


def feature_identity():
    return dict(model='torchvision_resnet50_imagenet1k_v2_layer3',
                weights_sha256=sha(data.backbone_weights()), input_hw=[192,336], grid=[3,5],
                motion='frozen_source_features_v1')


def protocol(parent, cfg):
    old = read(parent/'protocol.json')
    plan = distributed.checked_plan(parent/'.distributed/plan.json')
    if (sha(parent/'sources.json') != plan['source_manifest_sha256']
            or sha(parent/'protocol.json') != plan['protocol_sha256']):
        raise ValueError('parent manifest/protocol changed')
    for name, expected in old['supervision_protocol']['code'].items():
        if sha(REPO/'src/semantic_transmission'/f'{name}.py') != expected:
            raise ValueError('frozen measurement code changed: '+name)
    for name, expected in old['code'].items():
        if sha(REPO/name) != expected:
            raise ValueError('frozen v2 code changed: '+name)
    rows = select_rows(read(parent/'sources.json'), cfg)
    if not rows or cfg['schema'] != 'sparc_webvid_long_search_v1':
        raise ValueError('no eligible long training windows or invalid configuration')
    if min(cfg['budgets']) <= 0 or cfg['target_cbr'] not in cfg['budgets']:
        raise ValueError('invalid search budgets')
    prepared = {}
    for row in rows:
        path = parent/'clips'/row['id']/'prepared.json'
        ready = read(path)
        if ready['input'] != row or ready['protocol'] != old['supervision_protocol']['signature']:
            raise ValueError('prepared source/split changed')
        prepared[row['id']] = sha(path)
    names = ['sparc_selector_webvid_search', 'sparc_selector_budget', 'sparc_selector_v3_data',
             'sparc_selector_cnn', 'sparc_selector_distributed']
    value = dict(schema=cfg['schema'], config=cfg, parent=str(parent), rows=rows,
        prepared=prepared, parent_plan=plan['signature'], source_manifest_sha256=sha(parent/'sources.json'),
        runtime_sha256=sha(parent/'.distributed/runtimes.json'),
        supervision_protocol=old['supervision_protocol'], features=feature_identity(),
        code={n:sha(REPO/'src/semantic_transmission'/f'{n}.py') for n in names},
        train_only=True, automatic_training_import=False)
    value['signature'] = digest(value)
    return value


def initialize(root, parent, cfg):
    root.mkdir(parents=True, exist_ok=True)
    p = protocol(parent, cfg)
    path = root/'protocol.json'
    if path.exists() and read(path) != p:
        raise ValueError('frozen search protocol changed; use a new output directory')
    save(path, p)
    save(root/'sources.json', p['rows'])
    return p


def validate_dependencies(p, plan):
    # Hash the frozen models once; only selected long windows need frame audits.
    for name, expected in p['supervision_protocol']['assets'].items():
        relative=Path(name).relative_to(Path(plan['original_root']).parents[1])
        if sha(REPO/relative)!=expected:raise ValueError('frozen codec/receiver asset changed')
    from .sparc_uf_transport import UPSTREAM
    vendor=REPO/'.local/dcvc_uf_vendor'
    if (subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip()!=UPSTREAM
            or subprocess.check_output(['git','-C',str(vendor),'diff','HEAD'])):
        raise ValueError('UF vendor changed')


def proxy_cost(proxies, indices):
    # Only a proposal-ranking proxy; full-resolution decoded metrics decide labels.
    x = proxies[:,::4,::4].astype(np.float32)/255
    error = 0.
    for a, b in zip(indices, indices[1:]):
        if b-a > 1:
            t = np.arange(1,b-a,dtype=np.float32)[:,None,None,None]/(b-a)
            estimate = x[a][None]+(x[b]-x[a])[None]*t
            error += float(np.abs(x[a+1:b]-estimate).mean(axis=(1,2,3)).sum())
    return error/len(x)


def propose_batch(proxies, keys, cuts, observations, baseline, pixels, visited, cfg, round_index):
    """Measured rate beams across budgets; source proxy ranks within density bands.

    Cardinality only stratifies exploration. All acceptance and Pareto decisions
    use whole-packet measured channel uses, including padding and resets.
    """
    leaders = {}
    targets = cfg['budgets'][round_index % len(cfg['budgets']):]+cfg['budgets'][:round_index % len(cfg['budgets'])]
    for target in targets:
        for obs in sorted(observations, key=lambda o:budget.order_key(o,baseline,target*3*pixels))[:cfg['beam_per_budget']]:
            leaders[tuple(obs['identity']['indices'])] = obs
    leaders[tuple(baseline['identity']['indices'])] = baseline
    proposed = set()
    for indices, obs in leaders.items():
        proposed.update(budget.proposals(proxies, indices, keys, cuts,
            obs['quality'].get('per_frame',{}).get('mse')))
    options = sorted(proposed-set(visited), key=lambda s:(len(s),s))
    if not options:
        return []
    bands = np.array_split(np.arange(len(options)), min(cfg['per_round'],len(options)))
    return [list(min((options[int(i)] for i in band), key=lambda s:(proxy_cost(proxies,s),s))) for band in bands]


def measured_summary(observations, baseline, pixels, cfg):
    decisions = {}
    for target in cfg['budgets']:
        limit = math.floor(target*3*pixels)
        feasible = [o for o in observations if o['channel']['complex_channel_uses'] <= limit]
        preserving = [o for o in feasible
            if o['quality']['psnr_db'] >= baseline['quality']['psnr_db']-cfg['maximum_psnr_drop_db']
            and o['quality']['lpips_vgg'] <= baseline['quality']['lpips_vgg']+cfg['maximum_lpips_increase']]
        candidates = preserving or feasible
        best = min(candidates, key=lambda o:budget.quality_cost(o,baseline)) if candidates else None
        decisions[str(target)] = dict(feasible=bool(feasible), quality_goal_met=bool(preserving),
            budget_uses=limit, indices=best['identity']['indices'] if best else None,
            channel_uses=best['channel']['complex_channel_uses'] if best else None,
            quality=best['quality'] if best else None)
    frontier, lowest = [], float('inf')
    for obs in sorted(observations, key=lambda o:(o['channel']['complex_channel_uses'],budget.quality_cost(o,baseline))):
        q = budget.quality_cost(obs,baseline)
        if q < lowest:
            frontier.append(obs['identity']['indices']);lowest=q
    return dict(decisions=decisions, pareto_indices=frontier, measured_states=len(observations),
                baseline=baseline, per_video_budget=True, aggregate_budget_allocation_not_evaluated=True)


def search(path, identity, proxies, keys, cuts, pixels, measure, known, cfg, progress=lambda *_:None,
           stop=supervision.interrupted):
    initial = budget.seeds(proxies,keys,cuts)
    if path.exists():
        state = read(path)
        if state['identity'] != identity:
            raise ValueError('search journal identity changed')
    else:
        state = dict(identity=identity, observations=known, visited=[o['identity']['indices'] for o in known],
            pending=[list(s) for s in initial if list(s) not in [o['identity']['indices'] for o in known]],
            round=0, trials=0, phase='seeds', rejected=[])
        save(path,state)
    while state['phase'] != 'complete':
        stop()
        if state['pending'] and state['trials'] < cfg['maximum_new_candidates_per_clip']:
            indices = state['pending'][0]
            progress(state['trials'], state['round'], len(state['observations']))
            value = measure(indices)
            if value is not None:
                state['observations'].append(value)
            else:
                state['rejected'].append(indices)
            state['visited'].append(indices)
            state['pending'].pop(0);state['trials']+=1
            save(path,state)
            continue
        baseline = next((o for o in state['observations'] if tuple(o['identity']['indices'])==initial[0]),None)
        if baseline is None:
            state.update(phase='complete',status='BASELINE_CHANNEL_REJECTED')
        elif state['round'] >= cfg['rounds'] or state['trials'] >= cfg['maximum_new_candidates_per_clip']:
            state.update(phase='complete',status='MEASURED_SEARCH_COMPLETE')
        else:
            state['pending'] = propose_batch(proxies,keys,cuts,state['observations'],baseline,pixels,
                [tuple(v) for v in state['visited']],cfg,state['round'])
            state['round'] += 1;state['phase']='search'
            if not state['pending']:
                state.update(phase='complete',status='SEARCH_SPACE_EXHAUSTED')
        save(path,state)
    if state['status'] == 'BASELINE_CHANNEL_REJECTED':
        return dict(status=state['status'], use_for_training=False, trials=state['trials'])
    baseline = next(o for o in state['observations'] if tuple(o['identity']['indices'])==initial[0])
    return dict(status=state['status'], use_for_training=True, trials=state['trials'],
        **measured_summary(state['observations'],baseline,pixels,cfg))


def prepare_clip(root, parent, row, p):
    folder = root/'clips'/row['id'];folder.mkdir(parents=True,exist_ok=True)
    source = parent/'clips'/row['id']
    if sha(source/'prepared.json') != p['prepared'][row['id']]:
        raise ValueError('prepared source changed')
    frames, ready = data.load_source(source)
    buffer=io.BytesIO();np.save(buffer,frames);array_hash=hashlib.sha256(buffer.getbuffer()).hexdigest();del buffer
    receipt=folder/'prepared_receipt.json'
    if receipt.exists():
        record=read(receipt)
        for name, expected in record['hashes'].items():
            if sha(folder/name)!=expected:raise ValueError('prepared search evidence changed')
        if record['protocol']!=p['signature'] or record['source_sha256']!=array_hash:
            raise ValueError('prepared search identity changed')
        return ready, read(folder/'seed_observations.json'), array_hash
    observations=[];references=[]
    if (source/'labels.json').exists():
        labels=distributed.verify_clip(source,row,p['supervision_protocol'],full=True)
        names=sorted({record[k] for record in labels['records'] for k in ('before_observation','after_observation')})
        for name in names:
            path=source/'observations'/name/'result.json'
            observations.append(data.verify_observation(path,p['supervision_protocol'],ready,array_hash))
            references.append(dict(path=str(path),sha256=sha(path)))
    extractor=data.Extractor(p['config']['device'])
    try:
        if extractor.identity!=p['features']:raise ValueError('feature extractor changed')
        data.atomic_npz(folder/'features.npz',**extractor.extract(frames))
    finally:
        del extractor,frames
        import torch
        if torch.cuda.is_available():torch.cuda.empty_cache()
    save(folder/'prepared.json',ready)
    save(folder/'seed_observations.json',observations)
    save(folder/'base_observations.json',observations)
    save(receipt,dict(protocol=p['signature'],source_sha256=array_hash,reused=references,
        hashes={name:sha(folder/name) for name in ('prepared.json','features.npz','seed_observations.json')}))
    return ready,observations,array_hash


def verify_evidence(folder,p,ready,array_hash,state):
    seed=read(folder/'seed_observations.json')
    refs=read(folder/'prepared_receipt.json')['reused']
    if len(refs)!=len(seed):raise ValueError('reused evidence inventory changed')
    for ref,obs in zip(refs,seed):
        if sha(ref['path'])!=ref['sha256'] or data.verify_observation(Path(ref['path']),p['supervision_protocol'],ready,array_hash)!=obs:
            raise ValueError('reused measured evidence changed')
    original={tuple(o['identity']['indices']):o for o in seed}
    for obs in state['observations']:
        key=tuple(obs['identity']['indices'])
        if key in original:
            if original[key]!=obs:raise ValueError('journal observation changed')
        elif data.verify_observation(folder/'observations'/digest(obs['identity'])[:24]/'result.json',
                p['supervision_protocol'],ready,array_hash)!=obs:
            raise ValueError('journal differs from actual observation')
    for indices in state['rejected']:
        identity=dict(protocol=p['supervision_protocol']['signature'],source_sha256=array_hash,
                      indices=indices,keys=ready['keys'],cuts=ready['cuts'])
        out=folder/'observations'/digest(identity)[:24]
        rejection=read(out/'rejected.json')
        if (rejection['identity']!=identity or rejection['report']['bit_errors']<=0
                or any(sha(out/name)!=value for name,value in rejection['artifacts'].items())):
            raise ValueError('rejected channel evidence changed')


def export_dataset(root,p):
    entries=[];rejected=[]
    for row in p['rows']:
        if row['split']!='train':raise ValueError('held-out source in training dataset export')
        folder=root/'clips'/row['id'];result=read(folder/'search_result.json')
        if not result['use_for_training']:
            rejected.append(row['id']);continue
        names=['base_observations.json','features.npz','prepared.json','inputs.json','search_result.json']
        entries.append(dict(input=row,files={n:dict(path=str(folder/n),sha256=sha(folder/n)) for n in names},
            measured_states=result['measured_states'],new_candidates=result['trials'],
            target_met=result['decisions'][str(p['config']['target_cbr'])]['quality_goal_met']))
    out=dict(schema='sparc_absolute_selection_corpus_v1',protocol=p['signature'],split='train',clips=entries,
        parents=len({e['input']['parent'] for e in entries}),states=sum(e['measured_states'] for e in entries),
        target_met_clips=sum(e['target_met'] for e in entries),rejected_clips=rejected,
        original_reference_metrics=True,actual_channel_uses=True,features=p['features'],
        automatic_training_import=False,default_model_changed=False,
        source_frames_retained_in=str(p['parent']),artifact_dependencies='own observations plus hash-pinned reused parent observations')
    save(root/'dataset.json',out)
    return dict(status='DATASET_COMPLETE',clips=len(entries),parents=out['parents'],states=out['states'],
        target_met_clips=out['target_met_clips'],rejected_clips=rejected,dataset_sha256=sha(root/'dataset.json'),
        target_not_met_is_retained_as_training_evidence=True,automatic_training_import=False)


def run(root,parent,cfg):
    root.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        run_lock=stack.enter_context((root/'run.lock').open('a'))
        fcntl.flock(run_lock,fcntl.LOCK_EX)
        signal.signal(signal.SIGTERM,supervision.stop);signal.signal(signal.SIGINT,supervision.stop)
        began=time.time()
        def progress(stage,done=0,total=None,**extra):
            old=read(root/'status.json') if (root/'status.json').exists() else {}
            save(root/'status.json',dict(status='RUNNING',stage=stage,done=done,total=total,
                pid=os.getpid(),start_ticks=distributed.proc_identity(os.getpid()),started_unix=old.get('started_unix',began),
                updated_unix=time.time(),resume='verified source/artifact hashes and per-candidate journal',**extra))
        try:
            progress('verify_protocol')
            p=initialize(root,parent,cfg)
            plan=distributed.checked_plan(parent/'.distributed/plan.json')
            while True:
                supervision.interrupted()
                ready,info=parent_completion(parent,plan)
                if ready:break
                progress('waiting_for_local_labels',0,len(p['rows']),**info)
                time.sleep(5)
            gpu=stack.enter_context((REPO/'.local/etri_60s_check.lock').open('a'))
            while True:
                supervision.interrupted()
                try:fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    progress('waiting_for_gpu_lock',0,len(p['rows']));time.sleep(5)
            import torch
            torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            # Asset/vendor and original worker checks occur only after its successful completion.
            validate_dependencies(p,plan)
            runtime=read(parent/'.distributed/runtimes.json');supervision.runtimes=lambda:runtime
            for i,row in enumerate(p['rows']):
                supervision.interrupted()
                progress('prepare_long_window',i,len(p['rows']),clip=row['id'])
                ready,known,array_hash=prepare_clip(root,parent,row,p)
                folder=root/'clips'/row['id'];journal=folder/'search_state.json'
                if journal.exists():verify_evidence(folder,p,ready,array_hash,read(journal))
                with np.load(folder/'features.npz') as z:proxies=z['proxies'].copy()
                measure=data.Measurer(root,parent,row,dict(supervision_protocol=p['supervision_protocol']))
                # On resume the final export must not expand the frozen initial teacher state.
                measure.cached={tuple(o['identity']['indices']):o for o in known}
                def update(trials,round_index,states):
                    progress('search_long_window',i,len(p['rows']),clip=row['id'],candidate=trials,
                        candidate_cap=cfg['maximum_new_candidates_per_clip'],round=round_index,
                        rounds=cfg['rounds'],measured_states=states)
                try:
                    result=search(journal,dict(protocol=p['signature'],clip=row['id'],source_sha256=array_hash),
                        proxies,ready['keys'],ready['cuts'],measure.pixels,measure,known,cfg,update)
                    state=read(journal);verify_evidence(folder,p,ready,array_hash,state)
                    save(folder/'base_observations.json',state['observations'])
                    save(folder/'inputs.json',dict(identity=dict(protocol=p['signature'],source_sha256=array_hash),
                        features_sha256=sha(folder/'features.npz'),observations_sha256=sha(folder/'base_observations.json')))
                    save(folder/'search_result.json',result)
                finally:measure.close()
                progress('window_complete',i+1,len(p['rows']),clip=row['id'],new_candidates=result['trials'])
            result=export_dataset(root,p);save(root/'RESULT.json',result)
            progress('complete',len(p['rows']),len(p['rows']))
            state=read(root/'status.json');state.update(status=result['status'],result=result);save(root/'status.json',state)
            return result
        except BaseException as error:
            state=read(root/'status.json') if (root/'status.json').exists() else {}
            state.update(status='INTERRUPTED' if supervision.STOP or isinstance(error,InterruptedError) else 'FAILED',
                         error=repr(error),updated_unix=time.time())
            save(root/'status.json',state)
            raise


def launch(root,parent,cfg_path):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return dict(status='ALREADY_RUNNING')
        if (root/'launcher.json').exists() and distributed.active(read(root/'launcher.json')):
            return dict(status='ALREADY_RUNNING')
        initialize(root,parent,read(cfg_path))
        if (root/'RESULT.json').exists() and read(root/'RESULT.json')['status']=='DATASET_COMPLETE':
            return dict(status='DATASET_COMPLETE',output=str(root))
        env=dict(os.environ,PYTHONPATH=str(REPO/'src'),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
            PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with (root/'run.log').open('ab') as log:
            proc=subprocess.Popen([sys.executable,'-m',MODULE,'run','--root',str(root),'--parent',str(parent),
                '--config',str(cfg_path)],cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True)
        save(root/'launcher.json',dict(pid=proc.pid,start_ticks=distributed.proc_identity(proc.pid)))
    return dict(status='LAUNCHED',pid=proc.pid,output=str(root),waits_for_local_labels=True)


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
    parser.add_argument('command',choices=['run','launch','stop'])
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--parent',type=Path,default=DEFAULT_PARENT)
    parser.add_argument('--config',type=Path,default=CONFIG)
    args=parser.parse_args();root=args.root.resolve();parent=args.parent.resolve()
    result=(run(root,parent,read(args.config)) if args.command=='run' else
            launch(root,parent,args.config.resolve()) if args.command=='launch' else stop(root))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
