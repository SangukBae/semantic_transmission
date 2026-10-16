"""Frozen-checkpoint full-video, native-rate extension of the adapter pilot."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from . import bsu_motion_validation as common
from .artifacts import sha256,write_json


def initialize(root):
    from . import generation_noise,tail_reference,precision_text_cache,condition_collision,short_schedule
    from .wire import unpack
    out=root/'adapter_native';out.mkdir(exist_ok=True)
    if (out/'plan.json').exists():return common.read(out/'plan.json')
    examples=[e for e in common.read(root/'adapter/plan.json')['examples'] if e['role']!='train']
    checkpoint=root/'adapter/adapter.pt';jobs=[];cases=[]
    for e in examples:
        original=Path(e['packet']).parent.parent;inputs=common.read(original/'receiver/decoder_inputs.json')
        keys=inputs['indices'];frames=inputs['video']['frames']
        cases.append(dict(id=e['id'],role=e['role'],keys=keys,frames=frames,source_sha256=e['source_sha256'],
            source_directory=e['source_directory'],source_movie=str(Path(e['source_directory']).parent/'normalized.mp4')))
        for arm in ('baseline','adapter'):
            run=out/e['id']/arm/'run';run.mkdir(parents=True,exist_ok=True)
            for name in ('run_config.json','keyframes.json','receiver/decoder_inputs.json','receiver/metadata.csv','received/metadata.bin'):
                dest=run/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original/name,dest)
            shutil.copytree(original/'receiver/frames/sample/key_frames_received',run/'receiver/frames/sample/key_frames_received')
            (run/'received/visual.c64').symlink_to(original/'received/visual.c64')
            shutil.copyfile(e['trajectory'],run/'received/trajectory.bin')
            cfg=common.read(run/'run_config.json')
            contract=dict(policy=generation_noise.POLICY,seed=cfg['seed'],segments=len(keys)-1,steps=cfg['steps'],
                comparison='same-received-data-noise-native-adapter-only',reference_policy=tail_reference.POLICY)
            write_json(run/'receiver_policy.json',dict(policy=tail_reference.POLICY,reference_frames=17,reference_latents=5,alignment=5,
                text_encoder_policy=precision_text_cache.POLICY,t5_precision='fp32',noise_contract=contract,noise_reference=None,
                condition_collision_policy=condition_collision.POLICY,short_schedule_policy=short_schedule.POLICY))
            write_json(run/'adapter_runtime.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),enabled=arm=='adapter'))
            jobs.append(dict(id=e['id'],arm=arm,run=str(run),role=e['role']))
    plan=dict(status='PREREGISTERED_NATIVE_EXTENSION',cases=cases,jobs=jobs,checkpoint_sha256=sha256(checkpoint),
        no_retraining=True,no_additional_transmission=True,selection='all eight source-disjoint architecture-pilot evaluation videos',
        caveat='same evaluation sources as the architecture pilot; native-scale confirmation, not a second unseen test set',
        prefix_policy='five zero trajectory condition latents for the fixed previous-17-frame context',
        criteria=common.read(root/'adapter/plan.json')['criteria'],
        inputs={e['packet']:sha256(e['packet']) for e in examples},
        code={str(p):sha256(p) for p in [Path(__file__),common.REPO/'src/semantic_transmission/trajectory_adapter_runtime.py',
            common.REPO/'src/semantic_transmission/trajectory_adapter.py',common.REPO/'scripts/etri_trajectory_adapter_decoder.py']})
    write_json(out/'plan.json',plan);return plan


def text(root):
    from . import precision_text_cache as cache,tail_reference
    from mmengine import Config
    plan=initialize(root);batch=root/'adapter_native/text';(batch/'receiver').mkdir(parents=True,exist_ok=True)
    rows=[]
    for j in plan['jobs']:
        with (Path(j['run'])/'receiver/metadata.csv').open() as f:rows.extend(csv.DictReader(f))
    with (batch/'receiver/metadata.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','text','flow']);w.writeheader();w.writerows(rows)
    cache.prepare(batch,common.REPO,tail_reference.decoder_config(Path(plan['jobs'][0]['run'])),'fp32')
    report=common.read(batch/cache.REPORT);entries={e['prompt']:e for e in report['entries']}
    for j in plan['jobs']:
        run=Path(j['run']);cfg=Config.fromstring(tail_reference.decoder_config(run),file_format='.py')
        prompts=cache.legacy.prepare_prompts(run/'receiver/metadata.csv',cfg,common.REPO)
        write_json(run/cache.REPORT,dict(report,prompts=prompts,entries=[entries[p] for p in dict.fromkeys(prompts)],
            prompt_count=len(prompts),metadata_sha256=sha256(run/'receiver/metadata.csv')))


def reconstruct(run):
    from .decoder_runner import run as decode
    from . import tail_reference,condition_collision,short_schedule,precision_text_cache
    config=run/'receiver/decoder_config.py';config.write_text(tail_reference.decoder_config(run))
    for p in ('receiver/reference_trace.json','receiver/tail_reference_trace.json',condition_collision.TRACE,short_schedule.TRACE):write_json(run/p,[])
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(common.REPO/'src'),str(common.REPO/'.local/vendor/Open-Sora')]),
        ETRI_REFERENCE_TRACE=str(run/'receiver/reference_trace.json'),ETRI_TAIL_REFERENCE_TRACE=str(run/'receiver/tail_reference_trace.json'),ETRI_PRECISION_RUN=str(run))
    dest=run/'receiver/reconstruction'
    if dest.exists():dest.rename(dest.with_name('reconstruction_failed_'+str(time.time_ns())))
    decode(run/'receiver/metadata.csv',dest,'key_frames_received',run/'receiver',
        decoder=common.REPO/'scripts/etri_trajectory_adapter_decoder.py',config=config,environment=env)
    precision_text_cache.validate_usage(run)


def evaluate(root):
    import numpy as np
    import statistics as st
    from PIL import Image
    from .bsu_receiver_improvement import PersistentMetrics
    from .stutter_metrics import evaluate as temporal
    from .bsu_receiver_compare import boundary_diagnostics
    from .generation_noise import TRACE
    plan=initialize(root);metrics=PersistentMetrics();rows=[]
    for ci,c in enumerate(plan['cases']):
        source=np.stack([np.asarray(Image.open(Path(c['source_directory'])/f'{i}.png').convert('RGB')) for i in range(c['frames'])])
        traces=[]
        for arm in ('baseline','adapter'):
            run=root/'adapter_native'/c['id']/arm/'run';traces.append(common.read(run/TRACE)['records'])
            keys=run/'receiver/frames/sample/key_frames_received'
            for kind,folder in [('raw',run/'receiver/reconstruction/sample_0000_frames'),('combined_context',run.parent/'refined/frames')]:
                receipt=run.parent/f'quality_{kind}.json'
                if receipt.exists():rows.append(common.read(receipt));continue
                output=np.stack([np.asarray(Image.open(folder/f'{i:05d}.png').convert('RGB')) for i in range(c['frames'])])
                if kind=='combined_context' and any(not np.array_equal(output[k],np.asarray(Image.open(keys/f'{k}.png').convert('RGB'))) for k in c['keys']):raise ValueError('received key changed')
                q,_=metrics.evaluate(source,output)
                row=dict(id=c['id'],role=c['role'],arm=arm,kind=kind,quality=q,temporal=temporal(source,output),boundaries=boundary_diagnostics(source,output,c['keys']))
                write_json(receipt,row);rows.append(row)
        if traces[0]!=traces[1]:raise ValueError('native paired noise differs')
        common.progress('NATIVE full-video evaluation',ci+1,len(plan['cases']))
    means={}
    for role in sorted({c['role'] for c in plan['cases']}):
        means[role]={}
        for kind in ('raw','combined_context'):
            means[role][kind]={}
            for arm in ('baseline','adapter'):
                selected=[r for r in rows if (r['role'],r['kind'],r['arm'])==(role,kind,arm)]
                means[role][kind][arm]=dict(**{k:st.mean(r['quality'][k] for r in selected) for k in ('psnr_db','lpips_vgg','ssim','dists','clip')},
                    **{k:st.mean(r['temporal'][k] for r in selected) for k in ('flow_error_mean','cadence_error_mean','false_freeze_rate')},
                    boundary_error=st.mean(st.mean(b['maximum_change_error'] for b in r['boundaries']) for r in selected))
    result=dict(status='COMPLETE_NATIVE_ADAPTER_EXTENSION',means=means,rows=rows,noise_tensor_identity=True,
        no_additional_transmission=True,frames=sum(c['frames'] for c in plan['cases']),sources=len(plan['cases']),
        production_default_changed=False,scope=plan['selection'],caveat=plan['caveat'])
    write_json(root/'adapter_native/RESULT.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


def worker(stage,root,log,run=None):
    log.parent.mkdir(parents=True,exist_ok=True);cmd=[sys.executable,'-m',__package__+'.bsu_adapter_native',stage,'--root',str(root)]
    if run:cmd+=['--run',str(run)]
    start=time.monotonic()
    with log.open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    write_json(log.with_suffix('.resource.json'),dict(returncode=p.returncode,seconds=time.monotonic()-start))
    if p.returncode:
        print(log.read_text(errors='replace')[-7000:],flush=True);raise RuntimeError(f'worker failed: {log}')


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['plan','run','text','reconstruct','refine','evaluate']);p.add_argument('--root',type=Path,default=common.ROOT);p.add_argument('--run',type=Path)
    a=p.parse_args()
    if a.stage=='plan':print(json.dumps(initialize(a.root),ensure_ascii=False));return
    if a.stage=='reconstruct':return reconstruct(a.run)
    if a.stage=='refine':return common.refine_run(a.run)
    if a.stage!='run':return globals()[a.stage](a.root)
    if common.verified(a.root,'adapter_native'):return
    plan=initialize(a.root);out=a.root/'adapter_native';worker('text',a.root,out/'logs/text.log');receipts=[]
    for i,j in enumerate(plan['jobs']):
        run=Path(j['run']);receipt=run.parent/'generation_complete.json'
        if receipt.exists():
            if any(sha256(p)!=h for p,h in common.read(receipt)['outputs'].items()):raise ValueError('native completed output changed')
        else:
            worker('reconstruct',a.root,run.parent/'logs/reconstruct.log',run)
            worker('refine',a.root,run.parent/'logs/refine.log',run)
            files=[run/'receiver/reconstruction/sample_0000.mp4',run.parent/'refined/reconstruction.mp4',run/'receiver/generation_noise.json',run/'receiver/adapter_trace.json']
            write_json(receipt,dict(outputs={str(p):sha256(p) for p in files}))
        receipts.append(receipt);common.progress('NATIVE paired reconstruction',i+1,len(plan['jobs']))
    worker('evaluate',a.root,out/'logs/evaluation.log');receipts.append(out/'RESULT.json')
    common.stage_complete(a.root,'adapter_native',[out/'plan.json',a.root/'adapter/adapter.pt'],receipts)
    print(json.dumps({k:v for k,v in common.read(out/'RESULT.json').items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
