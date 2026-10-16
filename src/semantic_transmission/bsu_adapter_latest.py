"""Frozen 7h adapter on the adopted 14-video BSU routed development cohort.

Replay only the changed receiver branch. Reuse is bounded by packet, source,
output and action hashes; paired generator noise must match the original run.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from .artifacts import sha256, write_json
from .qwen35_pairs_video import freeze
from .webvid5 import fingerprint
from .bsu_dense_validation import REPO, read, progress

ROOT = REPO/'outputs/bsu_adapter_latest_20261016_v1'
TRAINING = REPO/'outputs/bsu_adapter_training_7h_v1'
ROUTED = REPO/'outputs/bsu_temporal_routed_qp32_s2_min5_20261016_v1'
DENSE = REPO/'outputs/bsu_dense_temporal_qp32_s2_20261016_v1'
MODULE = 'semantic_transmission.bsu_adapter_latest'


def local(path):
    return Path(str(path).replace('/home/sangukbae/ETRI/Semantic/semantic_transmission', str(REPO)))


def prepare(root):
    import torch
    from .trajectory_adapter_runtime import checkpoint_record, native_condition
    from .trajectory_adapter import TrajectoryAdapter
    from .bsu_dense_router import receive
    from .bsu import profile, CODE
    routed = read(ROUTED/'RESULT.json'); training = read(TRAINING/'protocol.json')
    final = read(TRAINING/'final.json'); checkpoint = checkpoint_record(TRAINING/'final.pt')
    if checkpoint['checkpoint_sha256'] != final['sha256'] or checkpoint['steps'] != final['step']:
        raise ValueError('final training export mismatch')
    cases = read(DENSE/'protocol.json')['cases']
    train_hashes = {c['source_sha256'] for c in training['cases'] if c['role']=='train'}
    if train_hashes & {c['source']['source_sha256'] for c in cases}:
        raise ValueError('training source leaked into evaluation')
    adapter = TrajectoryAdapter(); adapter.load_state_dict(torch.load(TRAINING/'final.pt',map_location='cpu',weights_only=True)['state_dict'])
    with torch.no_grad():
        zero = adapter(torch.from_numpy(native_condition(None,[None,None],33,40,72,True))[None])
    if torch.count_nonzero(zero):
        raise ValueError('absent trajectory is not an exact identity')
    changed=[]; inputs={}; audit=[]
    roles={c['id']:c['role'] for c in training['evaluation']}
    for row,case in zip(routed['rows'],cases):
        if row['id'] != case['id']: raise ValueError('cohort order differs')
        for path,digest in row['original_artifacts'].items():
            path=local(path)
            if sha256(path)!=digest: raise ValueError(f'protected original changed: {path}')
            inputs[str(path)]=digest
        movie=Path(row['quality']['movie']);source=Path(case['source']['normalized_video'])
        for path,digest in ((movie,row['quality']['movie_sha256']),(source,case['source']['normalized_sha256'])):
            if sha256(path)!=digest: raise ValueError(f'reused movie/source changed: {path}')
            inputs[str(path)]=digest
        if row['action']['input']['sha256']!=sha256(movie): raise ValueError('reused action input mismatch')
        packet=ROUTED/case['id']/'received.bin'
        mode,restored=receive(packet.read_bytes());inputs[str(packet)]=sha256(packet)
        if mode!=row['choice']['mode']: raise ValueError('route differs')
        old=Path(case['original_run']);trajectory=old.parents[3]/'received/trajectory.bin'
        if mode=='bsu':
            if restored['metadata']!=(old/'received/metadata.bin').read_bytes(): raise ValueError('routed metadata differs')
            if restored.get('trajectory')!=(trajectory.read_bytes() if trajectory.exists() else None): raise ValueError('routed trajectory differs')
        else:
            if restored['neural']!=(DENSE/case['id']/'received/packet.bin').read_bytes(): raise ValueError('neural receiver bytes differ')
        applied=mode=='bsu' and trajectory.exists()
        if applied: changed.append(case['id'])
        audit.append(dict(id=case['id'],mode=mode,adapter_changes_branch=applied,
                          role=roles.get(case['id'],'source_disjoint_no_training_trajectory'),
                          reuse_reason=None if applied else ('different_decoder' if mode=='neural' else 'zero_support_exact_identity')))
    modules=set(CODE)|{'bsu_adapter_latest','bsu_adapter_native','trajectory_adapter','trajectory_adapter_runtime',
        'bsu_sequence_refinement','generation_noise','tail_reference','short_schedule','condition_collision',
        'precision_text_cache','bsu_dense_validation','official_quality','stutter_metrics'}
    p=dict(profile=profile(),checkpoint=checkpoint,training_protocol_sha256=sha256(TRAINING/'protocol.json'),
        routed_result_sha256=sha256(ROUTED/'RESULT.json'),cases=cases,changed=changed,audit=audit,inputs=inputs,
        code={n:sha256(REPO/f'src/semantic_transmission/{n}.py') for n in sorted(modules)},
        zero_support_identity=True,source_disjoint=True,checkpoint_selection='fixed final step, no evaluation tuning',
        scope='14 existing development videos; not a new unseen cohort or WebVid validation')
    p['signature']=fingerprint(p);root.mkdir(parents=True,exist_ok=True);freeze(root/'protocol.json',p)
    return p


def initialize(root,case,p):
    from .bsu_dcvc_validation import portable_text_report
    old=Path(case['original_run']);run=root/case['id']/'run'
    receipt=run.parent/'prepared.json'
    if receipt.exists():
        for path,digest in read(receipt)['files'].items():
            if sha256(path)!=digest: raise ValueError('prepared receiver changed')
        return run
    files=[]
    for name in ('run_config.json','keyframes.json','receiver/decoder_inputs.json','receiver/metadata.csv',
                 'received/metadata.bin','receiver_policy.json'):
        dest=run/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old/name,dest);files.append(dest)
    keys=run/'receiver/frames/sample/key_frames_received'
    shutil.copytree(old/'receiver/frames/sample/key_frames_received',keys,dirs_exist_ok=True)
    files.extend(sorted(keys.glob('*.png')))
    shutil.copyfile(old.parents[3]/'received/trajectory.bin',run/'received/trajectory.bin');files.append(run/'received/trajectory.bin')
    visual=run/'received/visual.c64'
    if not visual.exists(): visual.symlink_to(old/'received/visual.c64')
    files.append(visual)
    portable_text_report(old,run);files.append(run/'receiver/text_embeddings.json')
    write_json(run/'adapter_runtime.json',p['checkpoint']);files.append(run/'adapter_runtime.json')
    write_json(receipt,dict(files={str(f):sha256(f) for f in files}))
    return run


def launch(root,stage,case_id=None,runtime=None):
    log=root/'logs'/f'{stage}_{case_id or "cohort"}.log';log.parent.mkdir(exist_ok=True)
    command=[runtime or sys.executable,'-m',MODULE,'--root',str(root),'--worker',stage]
    if case_id:command+=['--case',case_id]
    with log.open('w') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text(errors='replace')[-5000:],flush=True)
        raise RuntimeError(f'{stage} failed ({result.returncode}): {log}')


def worker(root,stage,case_id):
    p=read(root/'protocol.json')
    if stage=='action':
        from .bsu_key_reduction import action
        return action(read(root/'action_job.json'))
    case=next(c for c in p['cases'] if c['id']==case_id);out=root/case_id;run=out/'run'
    if stage=='generate':
        from .bsu_adapter_native import reconstruct
        from .bsu_key_reduction import refine
        reconstruct(run);refine(dict(run=str(run),output=str(out),keys=case['original_keys']))
        old=Path(case['original_run'])
        if read(run/'receiver/generation_noise.json')['records']!=read(old/'receiver/generation_noise.json')['records']:
            raise ValueError('paired noise tensors differ from baseline')
        from . import tail_reference,short_schedule,condition_collision
        tail_reference.validate_trace(read(run/'receiver/tail_reference_trace.json'),len(case['original_keys'])-2)
        short_schedule.validate_trace(read(run/short_schedule.TRACE),case['original_keys'],30)
        condition_collision.validate_trace(read(run/condition_collision.TRACE),
            condition_collision.preview(case['original_keys'],condition_collision.upstream_functions(REPO)))
        from PIL import Image
        import numpy as np
        for k in case['original_keys']:
            if not np.array_equal(np.asarray(Image.open(out/f'refined/frames/{k:05d}.png')),
                                  np.asarray(Image.open(run/f'receiver/frames/sample/key_frames_received/{k}.png'))):
                raise ValueError('received key RGB changed')
        paths=[out/'refined/reconstruction.mp4',run/'receiver/generation_noise.json',run/'receiver/adapter_trace.json']
        write_json(out/'generation.json',dict(noise_tensor_identity=True,key_rgb_exact=True,
            files={str(f):sha256(f) for f in paths}))
    elif stage=='evaluate':
        from .bsu_dense_validation import evaluate
        row=next(r for r in read(ROUTED/'RESULT.json')['rows'] if r['id']==case_id)
        for arm,movie in [('baseline',Path(row['quality']['movie'])),('adapter',out/'refined/reconstruction.mp4')]:
            dest=out/arm;dest.mkdir(exist_ok=True)
            evaluate(case,movie,dest,row['choice']['channel_uses'])
    else: raise ValueError(stage)


def summarize(root,p):
    rows=copy.deepcopy(read(ROUTED/'RESULT.json')['rows']);action=read(root/'action/RESULT.json')
    for row in rows:
        cid=row['id'];row['adapter_applied']=cid in p['changed']
        row['before']=row.pop('quality')
        if cid in p['changed']:
            row['before']=read(root/cid/'baseline/quality.json')
            row['after']=read(root/cid/'adapter/quality.json')
            row['action_before']=row.pop('action')
            row['action_after']=next(r for r in action['records'] if r['input']['id']==cid)
        else:
            row['after']=row['before'];row['action_before']=row.pop('action');row['action_after']=row['action_before']
    def means(group,arm):
        return dict(quality={k:statistics.mean(r[arm]['quality'][k] for r in group) for k in
                           ('psnr_db','ssim','lpips_vgg','clip','dists')},
            temporal={k:statistics.mean(r[arm]['temporal'][k] for r in group) for k in
                      ('flow_error_mean','cadence_error_mean','false_freeze_rate')},
            action={k:sum(r['action_'+arm][k] for r in group) for k in ('top1_correct','top5_correct')},videos=len(group))
    subsets={'all_14':rows,'adapter_active_4':[r for r in rows if r['adapter_applied']]}
    role_by_id={r['id']:r['role'] for r in p['audit']}
    subsets['heldout_test_active']=[r for r in rows if r['adapter_applied'] and role_by_id[r['id']]=='test']
    summary={name:{arm:means(group,arm) for arm in ('before','after')} for name,group in subsets.items()}
    from .key_reduction_selection import guard
    guards={r['id']:guard(r['before'],r['after']) for r in rows if r['adapter_applied']}
    result=dict(status='COMPLETE_LATEST_BSU_ADAPTER_VALIDATION',summary=summary,rows=rows,guards=guards,
        checkpoint=p['checkpoint'],protocol_sha256=sha256(root/'protocol.json'),source_disjoint=True,
        channel_uses_before=sum(r['choice']['channel_uses'] for r in rows),
        channel_uses_after=sum(r['choice']['channel_uses'] for r in rows),
        noise_tensor_identity=True,zero_support_identity=True,unchanged_outputs_reused=10,
        production_default_changed=False,scope=p['scope'],
        adopted_automatically=False)
    write_json(root/'RESULT.json',result)
    print(json.dumps(dict(status=result['status'],summary=summary),ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--check',action='store_true');parser.add_argument('--worker');parser.add_argument('--case')
    args=parser.parse_args();root=args.root.resolve()
    if args.worker:return worker(root,args.worker,args.case)
    from .etri_60s_check import lock
    root.mkdir(parents=True,exist_ok=True)
    with lock(root/'execution.lock'),lock(REPO/'.local/etri_60s_check.lock'):
        progress('checkpoint/input audit',0,1);p=prepare(root)
        if args.check:
            print(json.dumps(dict(status='READY',changed=p['changed'],checkpoint=p['checkpoint'])));return
        for i,cid in enumerate(p['changed']):
            case=next(c for c in p['cases'] if c['id']==cid);out=root/cid
            initialize(root,case,p)
            progress('adapter generation',i,len(p['changed']),cid)
            receipt=out/'generation.json'
            if receipt.exists():
                if any(sha256(f)!=h for f,h in read(receipt)['files'].items()):raise ValueError('completed generation changed')
            else:launch(root,'generate',cid)
            if not (out/'adapter/quality.json').exists():launch(root,'evaluate',cid)
            progress('paired quality',i+1,len(p['changed']),cid)
        if not (root/'action/RESULT.json').exists():
            original=read(ROUTED/'RESULT.json')['rows'];runtime=read(DENSE/'protocol.json')['action_runtime']
            action_rows=[]
            for cid in p['changed']:
                q=read(root/cid/'adapter/quality.json');row=next(r for r in original if r['id']==cid)
                action_rows.append(dict(id=cid,variant='adapter',label=row['action']['input']['label'],
                                        path=q['movie'],sha256=q['movie_sha256']))
            freeze(root/'action_job.json',dict(output=str(root/'action'),runtime=runtime,rows=action_rows))
            progress('TimeSformer action',0,len(action_rows));launch(root,'action',runtime=runtime['python'])
        summarize(root,p);progress('complete',14,14)


if __name__=='__main__':main()
