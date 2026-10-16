"""Source-disjoint closed-loop native BSU quality checks for training checkpoints."""
import csv
import json
import os
from pathlib import Path
import shutil
import statistics as st
import subprocess
import sys
from . import bsu_motion_validation as common
from .artifacts import sha256,write_json


def quality_objective(result):
    """Equal dataset weighting; guards prevent rewarding incorrect extra motion."""
    scores=[]; guards={}; gains=[]
    for role,kinds in result['means'].items():
        a=kinds['combined_context']['baseline'];b=kinds['combined_context']['adapter']
        ratios={k:b[k]/max(a[k],1e-12) for k in ('lpips_vgg','flow_error_mean','cadence_error_mean')}
        scores.append((ratios['lpips_vgg']+.25*ratios['flow_error_mean']+.25*ratios['cadence_error_mean'])/1.5)
        guards[role]=dict(psnr=b['psnr_db']>=a['psnr_db']-.1,lpips=ratios['lpips_vgg']<=1.01,
            motion=ratios['flow_error_mean']<=1.02,cadence=ratios['cadence_error_mean']<=1.02,
            boundary=b['boundary_error']<=a['boundary_error']*1.05,
            dists=b['dists']<=a['dists']*1.01)
        gains.append(b['psnr_db']>=a['psnr_db']+.1 or ratios['lpips_vgg']<=.98 or ratios['flow_error_mean']<=.95)
    passed=all(all(v.values()) for v in guards.values()) and any(gains)
    return dict(score=st.mean(scores),guards=guards,passes_quality_guards=passed,improvement=any(gains),
        scope='engineering screen, not statistical significance or action-semantic correctness')


def setup(root,protocol,step,checkpoint,split):
    from . import generation_noise,tail_reference,precision_text_cache,condition_collision,short_schedule
    root=Path(root); output=root/('final_test' if split=='test' else f'validation/step_{step:07d}')
    native=output/'adapter_native';native.mkdir(parents=True,exist_ok=True)
    plan_path=native/'plan.json';checkpoint=Path(checkpoint)
    if plan_path.exists():
        p=common.read(plan_path)
        if p['checkpoint_sha256']!=sha256(checkpoint) or p['training_signature']!=protocol['signature']:raise ValueError('evaluation checkpoint/protocol changed')
        return output,p
    cases=[];jobs=[]
    selected=[c for c in protocol['evaluation'] if c['role']==split]
    if not selected:raise ValueError('empty source-disjoint evaluation split')
    for c in selected:
        original=Path(c['receiver']['base_packet']).parent.parent
        cases.append(dict(id=c['id'],role=f'{split}_{c["dataset"]}',keys=c['keys'],frames=c['frames'],
            source_sha256=c['source_sha256'],source_directory=str(common.source_directory(c)),source_movie=c['source_movie']))
        for arm in ('baseline','adapter'):
            arm_root=native/c['id']/arm
            # The disabled adapter baseline is independent of learned weights.
            # Its full output, noise trace and input hashes are checked on reuse.
            if arm=='baseline':
                shared=root/'quality_baseline'/c['id'];shared.mkdir(parents=True,exist_ok=True)
                arm_root.parent.mkdir(parents=True,exist_ok=True)
                if not arm_root.exists():arm_root.symlink_to(shared,target_is_directory=True)
            run=arm_root/'run';run.mkdir(parents=True,exist_ok=True)
            if not (run/'adapter_runtime.json').exists():
                for name in ('run_config.json','keyframes.json','receiver/decoder_inputs.json','receiver/metadata.csv','received/metadata.bin'):
                    dest=run/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original/name,dest)
                shutil.copytree(original/'receiver/frames/sample/key_frames_received',run/'receiver/frames/sample/key_frames_received',dirs_exist_ok=True)
                (run/'received/visual.c64').symlink_to(original/'received/visual.c64')
                shutil.copyfile(c['receiver']['trajectory'],run/'received/trajectory.bin')
                cfg=common.read(run/'run_config.json')
                contract=dict(policy=generation_noise.POLICY,seed=cfg['seed'],segments=len(c['keys'])-1,steps=cfg['steps'],
                    comparison='native-adapter-training-fixed-validation-noise-v1',reference_policy=tail_reference.POLICY)
                write_json(run/'receiver_policy.json',dict(policy=tail_reference.POLICY,reference_frames=17,reference_latents=5,alignment=5,
                    text_encoder_policy=precision_text_cache.POLICY,t5_precision='fp32',noise_contract=contract,noise_reference=None,
                    condition_collision_policy=condition_collision.POLICY,short_schedule_policy=short_schedule.POLICY))
                write_json(run/'adapter_runtime.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),enabled=arm=='adapter'))
            for path,source in [('received/metadata.bin',c['receiver']['base_packet']),('received/trajectory.bin',c['receiver']['trajectory'])]:
                if sha256(run/path)!=sha256(source):raise ValueError('evaluation receiver input differs')
            jobs.append(dict(id=c['id'],role=c['role'],arm=arm,run=str(run)))
    plan=dict(status='PREREGISTERED_TRAINING_QUALITY_CHECK',cases=cases,jobs=jobs,
        checkpoint_sha256=sha256(checkpoint),checkpoint=str(checkpoint),training_signature=protocol['signature'],
        selection=f'source-disjoint {split}, all full native frames',no_additional_transmission=True,
        caveat='validation is used for model selection; final test is evaluated only after training stops',
        split=split,step=step)
    write_json(plan_path,plan);return output,plan


def run(root,protocol,step,checkpoint,split,event):
    from .bsu_adapter_native import worker
    output,plan=setup(root,protocol,step,checkpoint,split);native=output/'adapter_native'
    final=output/'quality.json'
    if final.exists():
        d=common.read(final)
        if d['checkpoint_sha256']!=sha256(checkpoint) or d['result_sha256']!=sha256(native/'RESULT.json'):raise ValueError('quality receipt differs')
        return d
    event(f'{split} text preparation',None,None)
    worker('text',output,native/'logs/text.log')
    for i,j in enumerate(plan['jobs']):
        folder=Path(j['run']);receipt=folder.parent/'generation_complete.json'
        if receipt.exists():
            if any(sha256(p)!=h for p,h in common.read(receipt)['outputs'].items()):raise ValueError('completed quality generation changed')
        else:
            event(f'{split} full-video {j["id"]} {j["arm"]}',i,len(plan['jobs']))
            worker('reconstruct',output,folder.parent/'logs/reconstruct.log',folder)
            worker('refine',output,folder.parent/'logs/refine.log',folder)
            paths=[folder/'receiver/reconstruction/sample_0000.mp4',folder.parent/'refined/reconstruction.mp4',
                folder/'receiver/generation_noise.json',folder/'receiver/adapter_trace.json']
            write_json(receipt,dict(outputs={str(p):sha256(p) for p in paths}))
        event(f'{split} full-video reconstruction',i+1,len(plan['jobs']))
    worker('evaluate',output,native/'logs/evaluation.log')
    result=common.read(native/'RESULT.json')
    d=dict(quality_objective(result),step=step,split=split,checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
        result=str(native/'RESULT.json'),result_sha256=sha256(native/'RESULT.json'),sources=len(plan['cases']),
        noise_tensor_identity=result['noise_tensor_identity'])
    write_json(final,d);return d
