"""Reinvest a bounded part of key-codec savings into paid dense trajectories."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess

from . import bsu_dcvc_validation as full
from .artifacts import sha256, write_json
from .webvid5 import fingerprint
from .webvid_ablation import Stages
from .qwen35_pairs_video import freeze
from .key_reduction_selection import CRITERIA, guard
from .packets import accounting
from .wire import pack, unpack

REPO=full.REPO
MODULE='semantic_transmission.bsu_dcvc_motion_budget'
read=full.read
LEVELS=((32,32),(16,17))


def trajectory_packets(case,base_packet):
    import cv2
    import numpy as np
    from PIL import Image
    from . import sparse_trajectory as sparse, bsu_sequence_refinement as sequence
    cv2.setNumThreads(2);cv2.setRNGSeed(2025)
    h,_=unpack(base_packet)
    by_level=[[] for _ in LEVELS]
    for a,b in zip(case['original_keys'],case['original_keys'][1:]):
        frames=[]
        for index in range(a,b+1):
            with Image.open(Path(case['source']['source_frames'])/f'{index}.png') as im:
                frames.append(cv2.resize(np.asarray(im.convert('RGB')),(288,160),interpolation=cv2.INTER_AREA))
        frames=np.stack(frames)
        tracks=[sparse.track(frames,reverse=r) for r in (False,True)]
        for level,(points,knots) in enumerate(LEVELS):
            pieces=[]
            for reverse,(xy,valid) in enumerate(tracks):
                ids=sparse.choose_tracks(xy,valid,points//2,bool(reverse))
                pieces.append((xy[:,ids],valid[:,ids],np.full(len(ids),reverse)))
            xy=np.concatenate([x[0] for x in pieces],axis=1)
            valid=np.concatenate([x[1] for x in pieces],axis=1)
            origins=np.concatenate([x[2] for x in pieces])
            if xy.shape[1]<3:continue
            times=sparse.adaptive_knots(xy,valid,knots)
            by_level[level].append(dict(start=a,end=b,
                trajectory=sparse.encode(xy[times],valid[times],origins,times,len(frames))))
    return [pack(dict(policy=sequence.POLICY,video=h['video'],
        base_metadata_sha256=hashlib.sha256(base_packet).hexdigest(),windows=windows,
        transport_revision='bsu_reinvest_key_savings_v1',replaces_previous_trajectory=True,
        maximum_points=points,maximum_knots=knots))
        for windows,(points,knots) in zip(by_level,LEVELS)]


def worker(j,stage):
    if stage!='channel':full.worker(j,stage);return
    from .metadata_channel import transmit
    from .bsu_sequence_refinement import decode_packet
    run=Path(j['run']);packet=Path(j['packet']).read_bytes()
    reports=[]
    for seed in [43,1043,2043]:
        received,report=transmit(packet,10,seed,bits_per_symbol=4)
        if received!=packet or report['bit_errors']:raise ValueError('motion packet corrupted')
        decode_packet(received,(run/'received/metadata.bin').read_bytes())
        reports.append(report)
    (run/'received/trajectory.bin').write_bytes(received)
    uses=j['base_channel_uses']+report['complex_channel_uses']
    if uses>int(.75*j['case']['original_channel_uses']):raise ValueError('25 percent savings floor exceeded')
    write_json(run/'channel_accounting.json',dict(channel_uses=uses,base_channel_uses=j['base_channel_uses'],
        replacement_trajectory_reports=reports,conservative_original_trajectory_cost_still_counted=True))


def evaluate_variant(root,p,item,packet,level):
    case=item['case'];source=Path(item['run']);name=f'{item["arm"]}_points{LEVELS[level][0]}_knots{LEVELS[level][1]}'
    out=root/case['id']/name;run=out/'run';out.mkdir(parents=True,exist_ok=True)
    j=dict(protocol=p['signature'],case=case,keys=case['original_keys'],seed=2025,
           output=str(out),run=str(run),packet=str(packet),base_channel_uses=item['base_channel_uses'])
    freeze(out/'job.json',j);stages=Stages(out,fingerprint(j))
    def initialize(_):
        for path in ['received/metadata.bin','receiver/frames','receiver/reconstruction']:
            full.legacy.link(source/path,run/path)
        write_json(run/'generation_reuse.json',dict(original_run=str(source),
            original_noise_sha256=sha256(source/'receiver/generation_noise.json'),
            original_generated_movie_sha256=sha256(source/'receiver/reconstruction/sample_0000.mp4'),
            generation_rerun=False,receiver_reads_source=False))
    stages.step('initialize',['run'],['run/received/metadata.bin','run/receiver/frames',
                'run/receiver/reconstruction','run/generation_reuse.json'],initialize)
    for stage,owned,required in [
        ('channel',['run/received/trajectory.bin','run/channel_accounting.json'],['run/received/trajectory.bin','run/channel_accounting.json']),
        ('refine',['refined'],['refined/reconstruction.mp4','refined/audit.json']),
        ('evaluate',['quality.json','quality.csv'],['quality.json','quality.csv'])]:
        full.legacy.event(current=f'{case["id"]}/{name}',stage=stage)
        stages.step(stage,owned,required,lambda log,s=stage:full.execute(p,s,out/'job.json',log))
    q=read(out/'quality.json')
    return dict(id=case['id'],arm=item['arm'],level=LEVELS[level],quality=q,baseline=item['baseline'],
                verdict=guard(item['baseline'],q),saving_percent=100*(1-q['channel_uses']/case['original_channel_uses']))


def run(args):
    from .etri_60s_check import lock
    root=args.root.resolve();sources=[]
    original=read(REPO/'outputs/bsu_dcvc_validation_20261015_v2/protocol.json')
    for arm,directory in [('dcvc',REPO/'outputs/bsu_dcvc_validation_20261015_v2'),
                          ('hybrid',REPO/'outputs/bsu_dcvc_hybrid_20261015_v1')]:
        r=read(directory/'RESULT.json')
        for d in r['decisions']:
            if d['saving_percent']<25:continue
            case=next(c['case'] for c in original['candidates'] if c['case']['id']==d['id'])
            sources.append(dict(arm=arm,case=case,baseline=d['baseline'],
                run=str(directory/d['id']/'dcvc_2025/run'),base_channel_uses=d['candidate']['channel_uses']))
    p=dict(criteria=CRITERIA,levels=LEVELS,sources=sources,minimum_saving_percent=25,
        runtimes=original['runtimes'],worker_module=MODULE,action_runtime=original['action_runtime'],
        code={str(f.relative_to(REPO)):sha256(f) for f in (REPO/'src/semantic_transmission').glob('*.py')},
        hypothesis='use freed key-image bandwidth for denser source trajectories with the same combined_context receiver',
        final_gate='same full-video metrics; any pass still requires a new paired generation seed',
        original_digital_cost_still_counted=True,production_defaults_changed=False)
    p['signature']=fingerprint(p)
    with lock(REPO/'.local/etri_60s_check.lock'):
        freeze(root/'protocol.json',p)
        rows=[];cached={}
        for i,item in enumerate(sources):
            full.legacy.event(done=i,total=len(sources),current=item['case']['id'],stage='paid motion budget')
            key=item['case']['id']
            if key not in cached:
                base=(Path(item['run'])/'received/metadata.bin').read_bytes()
                cached[key]=trajectory_packets(item['case'],base)
            for level,packet in enumerate(cached[key]):
                uses=item['base_channel_uses']+accounting(len(packet))['complex_channel_uses']
                if uses>int(.75*item['case']['original_channel_uses']):continue
                target=root/'packets'/key/f'{level}.bin';target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists() and target.read_bytes()!=packet:raise ValueError('sender packet changed')
                target.write_bytes(packet)
                rows.append(evaluate_variant(root,p,item,target,level))
        labels={v['id']:v['label'] for v in read(REPO/'outputs/bsu_kinetics_context/benchmark/protocol.json')['videos']}
        action_rows=[]
        for d in rows:
            for variant,value in [('baseline',d['baseline']),(f'{d["arm"]}_{d["level"][0]}_{d["level"][1]}',d['quality'])]:
                action_rows.append(dict(id=d['id'],variant=variant,label=labels[d['id']],path=value['movie'],sha256=value['movie_sha256']))
        # Deduplicate shared original movies; evaluation never selects motion packets.
        action_rows=list({(r['id'],r['variant']):r for r in action_rows}.values())
        freeze(root/'action_job.json',dict(runtime=p['action_runtime'],rows=action_rows,output=str(root/'action')))
        Stages(root,p['signature']).step('action',['action'],['action/RESULT.json'],
                                       lambda log:full.execute(p,'action',root/'action_job.json',log))
        action=read(root/'action/RESULT.json')
        for d in rows:
            pair={r['input']['variant']:r for r in action['records'] if r['input']['id']==d['id']}
            a=pair['baseline'];b=pair[f'{d["arm"]}_{d["level"][0]}_{d["level"][1]}']
            d['action_noninferior']=all(not a[k] or b[k] for k in ['top1_correct','top5_correct'])
        write_json(root/'RESULT.json',dict(status='COMPLETE_PAID_MOTION_REFINEMENT_SCREEN',rows=rows,
            seed2025_passes=sum(r['verdict']['passed'] and r['action_noninferior'] for r in rows),
            confirmation_completed=False,action=action['summary'],production_defaults_changed=False))
        write_json(root/'completion_receipt.json',dict(status='PASSED',artifacts={str(f.relative_to(root)):sha256(f)
            for f in root.rglob('*') if f.is_file() and 'failed_attempts' not in f.parts and f.name!='completion_receipt.json'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=REPO/'outputs/bsu_dcvc_motion_budget_20261015_v1')
    parser.add_argument('--worker');parser.add_argument('--job',type=Path)
    args=parser.parse_args()
    if args.worker:worker(read(args.job),args.worker)
    else:run(args)
