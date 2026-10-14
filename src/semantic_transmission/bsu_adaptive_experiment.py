"""Frozen 2x2 receiver experiment: adaptive cadence and visible-key synthesis."""
import argparse
import copy
import csv
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from . import bsu_trajectory_experiment as parent
from .artifacts import sha256, write_json
from .webvid5 import read_json, fingerprint
from .wire import pack, unpack
from .packets import accounting
from .etri_60s_check import lock, pts_audit
from .webvid_ablation import Stages, snapshot

REPO = parent.REPO
PARENT = parent.ROOT
ROOT = REPO/'outputs/bsu_adaptive_visibility_20261012_v1'
ARMS = ('current_combined','adaptive_cadence','visible_keys','both')
TRANSFER = ('bus','lady-running','longboard','skate-park')
MODULE = 'semantic_transmission.bsu_adaptive_experiment'
METRICS = ('psnr_db','ssim','lpips_vgg','clip','dists')
TEMPORAL = ('flow_error_mean','cadence_error_mean','false_freeze_rate','stall_jump_rate',
            'motion_retention_ratio','max_luma_impulse')


def progress(label, n, total, started):
    print(f'{label} {n}/{total} ({100*n/total:.1f}%) elapsed={time.monotonic()-started:.1f}s; ETA unknown',flush=True)


def load(root):
    return parent.load(root)


def prepare(root):
    import cv2
    from .adaptive_receiver import CADENCE, VISIBILITY
    from .sparse_trajectory import fit_packet
    from .bsu_motion_experiment import source_window
    from .precision_text_cache import prepare as prepare_text
    if (root/'protocol.json').exists():
        load(root); print('REUSE frozen protocol and inputs',flush=True); return
    old=parent.load(PARENT); cases=copy.deepcopy(old['cases']);inputs=copy.deepcopy(old['inputs'])
    inputs[str(PARENT/'protocol.json')]=sha256(PARENT/'protocol.json')
    for c in cases:
        c.update(evaluation_group='previous_seven',reuse_baseline=str(PARENT/'generation'/c['name']/'baseline'),
                 received_reuse=str(PARENT/'received'/c['name']))
        for folder in (Path(c['reuse_baseline']),Path(c['received_reuse'])):
            for p in folder.rglob('*'):
                if p.is_file():inputs[str(p)]=sha256(p)
    cv2.setNumThreads(2);cv2.setRNGSeed(2025)
    work=root/'text_inputs';(work/'receiver').mkdir(parents=True,exist_ok=True)
    pending=[];rows=[]
    for name in TRANSFER:
        run=REPO/f'outputs/metric_validation_20260916_v1/03_natural/lgvsc/davis__{name}/steps_30/attempt_001/davis__{name}'
        h,_=unpack((run/'received/metadata.bin').read_bytes())
        channel_config=read_json(run/'run_config.json')
        if channel_config['snr_db']!=10:
            raise ValueError('transfer visual channel must be the declared 10 dB')
        keys=[k['index'] for k in h['keyframes']]
        if keys != [0,95] or h['video'] != dict(frames=96,fps=24,height=320,width=576):
            raise ValueError('transfer source geometry or keys changed')
        c=dict(name='transfer_'+name,run=str(run),keys=keys,frames=96,source_id='davis_'+name,
               source_video=str(run/'data/normalized.mp4'),source_sha256=sha256(run/'data/normalized.mp4'),
               visual_channel_provenance=dict(config=str(run/'run_config.json'),
                   config_sha256=sha256(run/'run_config.json'),snr_db=channel_config['snr_db'],
                   channel_seed=channel_config.get('channel_seed',channel_config['seed'])),
               evaluation_group='new_to_refinement',split='evaluation',previous_use='historical LGVSC metric validation; not in seven-source trajectory pilot',
               reuse_baseline=None,received_reuse=None)
        rows.append(dict(path='0,95',text=h['segments'][0]['text'],flow=h['segments'][0]['flow']))
        pending.append(c)
    with (work/'receiver/metadata.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=('path','text','flow'));writer.writeheader();writer.writerows(rows)
    print('PREPARE four received-caption embeddings; CPU T5 loading ETA unknown',flush=True)
    cfg=(parent.BASE/'receiver/decoder_config.py').read_text()
    prepare_text(work,REPO,cfg,'fp32')
    report_path=work/'receiver/text_embeddings.json';report=read_json(report_path)
    started=time.monotonic()
    for i,c in enumerate(pending):
        c.update(text_report=str(report_path),prompt=report['prompts'][i])
        run=Path(c['run']);header,payload=unpack((run/'received/metadata.bin').read_bytes())
        # The fixed cached decoder prompt is explicitly paid in this diagnostic.
        header['segments']=[dict(path='0,95',text=c['prompt'],flow=0)]
        header['schema']='bsu_two_endpoint_diagnostic_v1'
        baseline=pack(header,payload)
        visual=(run/'received/visual.c64').stat().st_size//8
        if visual != sum(k['complex_count'] for k in header['keyframes']):
            raise ValueError('visual payload accounting mismatch')
        cost=visual+accounting(len(baseline))['complex_channel_uses'];cap=int(cost*1.05)
        packet,audit=fit_packet(baseline,source_window(c['source_video'],*c['keys']),((cap-visual)//2304)*768)
        out=root/'sender'/c['name'];out.mkdir(parents=True,exist_ok=True)
        (out/'candidate.bin').write_bytes(packet);write_json(out/'extraction.json',audit)
        c['channel']=dict(baseline_uses=cost,candidate_uses=visual+accounting(len(packet))['complex_channel_uses'],
            visual_uses=visual,cap=cap,snr_db=10,metadata_seed=42,visual_channel_reused=True,
            cap_reference='same-window two-key BSU',candidate_packet_bytes=len(packet))
        if c['channel']['candidate_uses']>cap:raise ValueError('transfer packet exceeds budget')
        paths=[Path(c['source_video']),report_path,run/'run_config.json',run/'received/metadata.bin',run/'received/visual.c64',
               *[run/f'receiver/frames/sample/key_frames_received/{k}.png' for k in c['keys']],
               out/'candidate.bin',out/'extraction.json']
        entry=next(e for e in report['entries'] if e['prompt']==c['prompt']);paths.append(Path(entry['path']))
        inputs.update({str(p):sha256(p) for p in paths});cases.append(c)
        progress('PREPARE transfer sources',i+1,len(pending),started)
    if len({c['source_sha256'] for c in cases})!=len(cases):raise ValueError('source overlap')
    code=dict(old['code'])
    for name in ('adaptive_receiver.py','bsu_adaptive_experiment.py','bsu_webvid.py','bsu_sequence_refinement.py'):
        p=REPO/'src/semantic_transmission'/name;code[str(p)]=sha256(p)
    for relative in ('tests/test_adaptive_receiver.py','scripts/experiment_bsu_adaptive.sh'):
        p=REPO/relative;code[str(p)]=sha256(p)
    p=dict(policy='bsu_adaptive_visibility_factorial_v1',parent=str(PARENT),cases=cases,arms=ARMS,
        generator=old['generator'],parameters=old['parameters'],adaptive_cadence=CADENCE,visibility=VISIBILITY,
        inputs=inputs,code=code,training=False,receiver_reads_source=False,
        primary_comparison='each new arm minus current combined, matched received keys/tracks and raw generation',
        primary_boundary='lossless_frames',secondary_boundary='delivered_mp4',
        tuning='single shared parameter set frozen before any dataset output; no per-video oracle selection',
        adoption='overall quality and temporal tradeoff, per-video regressions and motion retention; no all-metric gate',
        scope='11 source-disjoint endpoint windows: previous seven plus four DAVIS sources new to this postprocessing experiment',
        limitations=['one generation seed/SNR; endpoint receiver test, not full current-Qwen selector/caption evaluation',
                     'transfer four had historical LGVSC use; not globally unseen or foundation-model holdout',
                     'short windows except 16s harbor; not full original-video or 60s evaluation',
                     'independent semantic and human smoothness evaluation pending'])
    p['signature']=fingerprint(p);write_json(root/'protocol.json',p)
    for path,digest in code.items():
        target=root/'frozen_code'/Path(path).relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        if sha256(target)!=digest:raise ValueError('code snapshot mismatch')


def channel(root):
    from .metadata_channel import transmit
    p=load(root);started=time.monotonic()
    for i,c in enumerate(p['cases'],1):
        out=root/'received'/c['name'];out.mkdir(parents=True,exist_ok=True)
        if (out/'receipt.json').exists():
            r=read_json(out/'receipt.json')
            if r['signature']!=p['signature']:raise ValueError('channel receipt mismatch')
            for rel,digest in r['artifacts'].items():
                if sha256(out/rel)!=digest:raise ValueError('channel artifacts changed')
        else:
            if c['received_reuse']:
                old=Path(c['received_reuse']);receipt=read_json(old/'receipt.json')
                for name in ('metadata.bin','key0.png','key1.png','visual.c64'):shutil.copyfile(old/name,out/name)
                channel_report=receipt['report']
            else:
                original=(root/'sender'/c['name']/'candidate.bin').read_bytes()
                restored,channel_report=transmit(original,10,42)
                if restored!=original:raise ValueError('channel packet corruption')
                (out/'metadata.bin').write_bytes(restored)
                run=Path(c['run'])
                shutil.copyfile(run/'received/visual.c64',out/'visual.c64')
                for j,k in enumerate(c['keys']):shutil.copyfile(run/f'receiver/frames/sample/key_frames_received/{k}.png',out/f'key{j}.png')
            write_json(out/'receipt.json',dict(signature=p['signature'],report=channel_report,exact_match=True,
                reused_from=c['received_reuse'],visual_and_keys_reused=True,
                artifacts={name:sha256(out/name) for name in ('metadata.bin','key0.png','key1.png','visual.c64')}))
        progress('CHANNEL cases',i,len(p['cases']),started)


def refine(root):
    import cv2
    import numpy as np
    from PIL import Image
    from . import adaptive_receiver as new, trajectory_receiver as old
    from .sparse_trajectory import FIELD, decode
    from .research_quality import read_frames
    from .receiver_joint_experiment import save_pixels
    from .receiver_alignment_experiment import complete_arm
    p=load(root);cv2.setNumThreads(2);cv2.setRNGSeed(2025);started=time.monotonic();done=0
    for c in p['cases']:
        baseline=root/'generation'/c['name']/'baseline'
        if not complete_arm(baseline,p['signature']):raise ValueError('raw generation incomplete')
        pixels=read_frames(baseline/'frames');received=root/'received'/c['name']
        h,_=unpack((received/'metadata.bin').read_bytes())
        tracks=decode(h[FIELD],c['frames']) if FIELD in h else None
        keys=[np.asarray(Image.open(received/f'key{j}.png').convert('RGB')) for j in (0,1)]
        for arm in ARMS:
            out=root/'generation'/c['name']/arm
            if not complete_arm(out,p['signature']):
                def execute(_):
                    start=time.monotonic()
                    def tick(n,total):
                        if n==1 or n%32==0 or n==total:progress(f'REFINE {c["name"]}/{arm} {phase}',n,total,start)
                    phase='reference'
                    op=new.refine_references if arm in ('visible_keys','both') else old.refine_references
                    reference,a=op(pixels,keys,tracks,tick)
                    phase='cadence';start=time.monotonic()
                    if arm in ('adaptive_cadence','both'):result,b=new.refine_temporal(reference,tracks,tick)
                    else:result,b=old.refine_temporal(reference,tick)
                    if result.shape!=pixels.shape or not np.array_equal(result[[0,-1]],pixels[[0,-1]]):
                        raise ValueError('timeline or endpoint changed')
                    if arm=='current_combined' and c['received_reuse']:
                        previous=read_frames(PARENT/'generation'/c['name']/'combined/frames')
                        if not np.array_equal(result,previous):raise ValueError('current combined differs from frozen parent pixels')
                    save_pixels(out,result,24)
                    write_json(out/'conditions.json',dict(arm=arm,reference=a,cadence=b,receiver_reads_source=False,
                        endpoint_pixels_preserved=True,parent_combined_pixel_match=arm=='current_combined' and bool(c['received_reuse'])))
                stages=Stages(out,p['signature']);stages.step('refine',['frames','reconstruction.mp4','conditions.json'],
                    ['frames','reconstruction.mp4','conditions.json'],execute)
                write_json(out/'result.json',dict(status='GENERATED',signature=p['signature'],frames=c['frames'],
                    artifacts=snapshot(out,['frames','reconstruction.mp4','conditions.json'])))
            done+=1;progress('REFINE completed conditions',done,len(p['cases'])*len(ARMS),started)


def aggregate(rows):
    import numpy as np
    result=[]
    for group in ('all','previous_seven','new_to_refinement'):
        for boundary in ('lossless_frames','delivered_mp4'):
            for arm in ARMS:
                selected=[r for r in rows if r['arm']==arm and r['boundary']==boundary and (group=='all' or r['group']==group)]
                if not selected:continue
                base=[next(b for b in rows if b['case']==r['case'] and b['boundary']==boundary and b['arm']==ARMS[0]) for r in selected]
                q={k:float(np.mean([r['quality'][k] for r in selected])) for k in METRICS}
                t={k:float(np.mean([r['temporal'][k] for r in selected])) for k in TEMPORAL}
                bq={k:float(np.mean([r['quality'][k] for r in base])) for k in METRICS}
                bt={k:float(np.mean([r['temporal'][k] for r in base])) for k in TEMPORAL}
                result.append(dict(group=group,boundary=boundary,arm=arm,videos=len(selected),quality=q,temporal=t,
                    quality_delta={k:q[k]-bq[k] for k in METRICS},temporal_delta={k:t[k]-bt[k] for k in TEMPORAL},
                    quality_relative_percent={k:100*(q[k]/bq[k]-1) if bq[k] else None for k in METRICS},
                    temporal_relative_percent={k:100*(t[k]/bt[k]-1) if bt[k] else None for k in TEMPORAL},
                    quality_improved_counts={k:sum((r['quality'][k]-b['quality'][k])*(1 if k in ('psnr_db','ssim','clip') else -1)>0 for r,b in zip(selected,base)) for k in METRICS},
                    temporal_improved_counts={k:sum(r['temporal'][k]<b['temporal'][k] for r,b in zip(selected,base)) for k in TEMPORAL if k!='motion_retention_ratio'}))
    return result


def evaluate(root):
    import cv2
    import torch
    import numpy as np
    from PIL import Image
    from .official_quality import OfficialMetrics
    from .research_quality import read_frames, read_video
    from .bsu_motion_experiment import source_window
    from .stutter_metrics import evaluate as temporal
    from .receiver_alignment_experiment import complete_arm
    from .receiver_alignment_evaluation import comparison_video
    p=load(root);cv2.setNumThreads(2);torch.set_num_threads(2);rows=[];started=time.monotonic();done=0
    for c in p['cases']:
        source=source_window(c['source_video'],*c['keys']);arrays={}
        for arm in ARMS:
            out=root/'generation'/c['name']/arm
            if not complete_arm(out,p['signature']):raise ValueError('refinement incomplete')
            arrays[arm]=read_frames(out/'frames')
            for boundary in ('lossless_frames','delivered_mp4'):
                cache=out/f'evaluation_{boundary}.json'
                if cache.exists():
                    row=read_json(cache)
                    if row['signature']!=p['signature'] or row['receipt_sha256']!=sha256(out/'result.json'):
                        raise ValueError('stale evaluation')
                else:
                    frames=arrays[arm] if boundary=='lossless_frames' else read_video(out/'reconstruction.mp4')
                    start=time.monotonic()
                    def tick(n,total):
                        if n==1 or n%128==0 or n==total:progress(f'QUALITY {c["name"]}/{arm}/{boundary}',n,total,start)
                    quality,per_frame=OfficialMetrics().evaluate(source,frames,tick)
                    row=dict(signature=p['signature'],case=c['name'],group=c['evaluation_group'],arm=arm,boundary=boundary,
                        quality=quality,temporal=temporal(source,frames),receipt_sha256=sha256(out/'result.json'),
                        channel_uses=c['channel']['candidate_uses'])
                    write_json(out/f'per_frame_{boundary}.json',per_frame);write_json(cache,row)
                rows.append(row);done+=1;progress('EVALUATE conditions/boundaries',done,len(p['cases'])*len(ARMS)*2,started)
                print(f'RESULT {c["name"]}/{arm}/{boundary}: PSNR={row["quality"]["psnr_db"]:.4f} LPIPS={row["quality"]["lpips_vgg"]:.5f} cadence={row["temporal"]["cadence_error_mean"]:.5f} freeze={row["temporal"]["false_freeze_rate"]:.5f}',flush=True)
        review=root/'review'/c['name'];review.mkdir(parents=True,exist_ok=True)
        if not (review/'comparison.mp4').exists():
            comparison_video(review/'comparison.mp4',[source,*[arrays[a] for a in ARMS]],['source',*ARMS],24)
        selected=sorted(set(np.linspace(0,c['frames']-1,5).round().astype(int).tolist()))
        for t in selected:
            Image.fromarray(np.concatenate([source[t],*[arrays[a][t] for a in ARMS]],axis=0)).save(review/f'frame_{t:04d}.png')
        ids=list(range(max(0,c['frames']//2-4),min(c['frames'],c['frames']//2+5)))
        strips=[np.concatenate([cv2.resize(a[t],(192,107)) for t in ids],axis=1) for a in [source,*[arrays[a] for a in ARMS]]]
        Image.fromarray(np.concatenate(strips,axis=0)).save(review/'motion_strip.png')
        write_json(review/'sample_plan.json',dict(fixed_frames=selected,motion_strip_frames=ids,rows=['source',*ARMS]))
    result=dict(status='COMPLETE_QUANTITATIVE_REVIEW_PENDING',signature=p['signature'],rows=rows,aggregates=aggregate(rows),
        cases=len(p['cases']),conditions=len(p['cases'])*len(ARMS),additional_channel_uses_vs_current=0,
        default_changed=False,independent_semantic_review='PENDING',limitations=p['limitations'])
    write_json(root/'RESULT.json',result)
    lines=['group,boundary,arm,videos,psnr_delta,lpips_percent,dists_percent,flow_percent,cadence_percent,freeze_percentage_points']
    for r in result['aggregates']:
        lines.append(','.join(map(str,[r['group'],r['boundary'],r['arm'],r['videos'],r['quality_delta']['psnr_db'],
            r['quality_relative_percent']['lpips_vgg'],r['quality_relative_percent']['dists'],
            r['temporal_relative_percent']['flow_error_mean'],r['temporal_relative_percent']['cadence_error_mean'],
            r['temporal_delta']['false_freeze_rate']*100])))
    (root/'summary.csv').write_text('\n'.join(lines)+'\n')


def verify(root):
    import numpy as np
    from PIL import Image
    from .receiver_alignment_experiment import complete_arm
    p=load(root);result=read_json(root/'RESULT.json');frames=0
    if result['signature']!=p['signature'] or len(result['rows'])!=len(p['cases'])*8:raise ValueError('incomplete scores')
    for c in p['cases']:
        received=root/'received'/c['name'];r=read_json(received/'receipt.json')
        for rel,digest in r['artifacts'].items():
            if sha256(received/rel)!=digest:raise ValueError('received artifact changed')
        for arm in ARMS:
            out=root/'generation'/c['name']/arm
            if not complete_arm(out,p['signature']):raise ValueError('output changed')
            if len(list((out/'frames').glob('*.png')))!=c['frames']:raise ValueError('frame count mismatch')
            pts_audit(out/'reconstruction.mp4',c['frames'],24)
            for k in (0,c['frames']-1):
                a=np.asarray(Image.open(out/f'frames/{k:05d}.png'))
                b=np.asarray(Image.open(root/f'generation/{c["name"]}/baseline/frames/{k:05d}.png'))
                if not np.array_equal(a,b):raise ValueError('endpoint changed')
            frames+=c['frames']
        if len({r['channel_uses'] for r in result['rows'] if r['case']==c['name']})!=1:raise ValueError('unequal transmission')
    write_json(root/'verification.json',dict(status='PASSED',signature=p['signature'],cases=len(p['cases']),
        conditions=len(p['cases'])*4,png_frames=frames,all_endpoints_preserved=True,output_timing_preserved=True,
        received_inputs_hashes_verified=True,all_arms_same_paid_packet=True,receiver_reads_source=False))
    print(f'VERIFIED {len(p["cases"])*4} conditions, {frames} PNGs, equal transmission',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--stage',choices=('all','prepare','channel','baseline','refine','evaluate','verify'),default='all')
    args=parser.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    if args.stage!='all':
        if args.stage=='baseline':parent.baseline(root)
        else:globals()[args.stage](root)
        return
    from .bsu_webvid import command
    from .cli import settings
    local=settings(REPO)
    with lock(root/'execution.lock'),lock(REPO/'.local/etri_60s_check.lock'):
        for i,stage in enumerate(('prepare','channel','baseline','refine','evaluate','verify')):
            write_json(root/'status.json',dict(status='RUNNING',stage=stage,completed_stages=i,total_stages=6))
            print(f'OVERALL stages {i}/6 ({100*i/6:.1f}%) | {stage}; ETA unknown',flush=True)
            py=local['channel_python' if stage=='channel' else 'python']
            try:
                command([py,'-m',MODULE,'--root',root,'--stage',stage],root/f'logs/{stage}_{time.time_ns()}.log',os.environ.copy())
            except BaseException as error:
                write_json(root/'status.json',dict(status='FAILED',stage=stage,error=str(error)));raise
        write_json(root/'status.json',dict(status='COMPLETE',completed_stages=6,total_stages=6))
        print('OVERALL 6/6 (100%) COMPLETE; quantitative results ready, review remains separate',flush=True)


if __name__=='__main__':main()
