"""Frozen seven-source BSU trajectory/cadence pilot, with matched ablations."""
import argparse
import copy
import fcntl
import html
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .artifacts import sha256, write_json
from .webvid5 import read_json, fingerprint
from .bsu_anchor_experiment import REPO, BASE, progress
from .wire import pack, unpack
from .packets import accounting

ROOT=REPO/'outputs/bsu_trajectory_cadence_20261012_v2'
PARENT=REPO/'outputs/bsu_motion_pilot_20261010_v2'
ARMS=['baseline','cadence','trajectory','combined']
CODE=['sparse_trajectory.py','trajectory_receiver.py','stutter_metrics.py','bsu_trajectory_experiment.py',
      'official_quality.py','research_quality.py','receiver_joint_experiment.py','receiver_alignment_experiment.py',
      'receiver_alignment_evaluation.py','generation_noise.py','precision_text_cache.py','condition_collision.py',
      'wire.py','packets.py','metadata_channel.py','bsu_motion_experiment.py']


def load(root):
    p=read_json(root/'protocol.json');check=dict(p);signature=check.pop('signature')
    if fingerprint(check)!=signature:raise ValueError('protocol signature changed')
    for group in ['code','inputs']:
        for path,digest in p[group].items():
            if sha256(path)!=digest:raise ValueError(f'frozen {group} changed: {path}; use a new root')
    return p


def prepare(root):
    import cv2
    from .bsu_motion_experiment import source_window
    from .sparse_trajectory import fit_packet
    from .trajectory_receiver import PARAMETERS
    if (root/'protocol.json').exists():load(root);print('RESUME preparation verified',flush=True);return
    cv2.setNumThreads(2);cv2.setRNGSeed(2025)
    old=read_json(PARENT/'protocol.json')
    selected=['zea','person_walk','single_subject','candle_flowers','train_tv_00','train_miami-surf','train_rallye']
    cases=[];inputs={};start=time.monotonic()
    for i,name in enumerate(selected,1):
        c=copy.deepcopy(next(c for c in old['cases'] if c['name']==name))
        if sha256(c['source_video'])!=c['source_sha256']:raise ValueError('source video changed')
        c['evaluation_group']='known_development' if i<=4 else 'transfer_frozen_parameters'
        c['previous_use']=c['split'];c['split']='evaluation'
        run=Path(c['run']);header,payload=unpack((run/'received/metadata.bin').read_bytes())
        raw_visual=(run/'received/visual.c64').read_bytes()
        keyinfo=[];keypayload=bytearray();visual=bytearray()
        for j,k in enumerate(c['keys']):
            item=copy.deepcopy(next(x for x in header['keyframes'] if x['index']==k))
            rate=payload[item['rate_offset']:item['rate_offset']+item['rate_bytes']]
            symbols=raw_visual[item['complex_offset']*8:(item['complex_offset']+item['complex_count'])*8]
            item.update(index=j*(c['frames']-1),rate_offset=len(keypayload),complex_offset=len(visual)//8)
            keyinfo.append(item);keypayload.extend(rate);visual.extend(symbols)
        if name=='zea':
            baseline=(run/'received/metadata.bin').read_bytes()
        else:
            header=copy.deepcopy(header)
            header['video'].update(frames=c['frames'],fps=24,width=576,height=320)
            header['keyframes']=keyinfo
            # The exact cached decoder prompt is explicitly paid in this packet.
            header['segments']=[dict(path=f'0,{c["frames"]-1}',text=c['prompt'],flow='0')]
            header['schema']='bsu_two_endpoint_diagnostic_v1'
            baseline=pack(header,bytes(keypayload))
        baseline_uses=len(visual)//8+accounting(len(baseline))['complex_channel_uses']
        # Harbor has a verified same-input LGVSC budget. Other windows use the
        # stricter matched two-key cost; do not label that as measured LGVSC.
        cap=int((92475 if name=='zea' else baseline_uses)*1.05)
        max_blocks=(cap-len(visual)//8)//2304;capacity=max_blocks*768
        source=source_window(c['source_video'],*c['keys'])
        if source.shape[1:]!=(320,576,3):raise ValueError(f'unsupported source shape {source.shape}')
        candidate,audit=fit_packet(baseline,source,capacity)
        directory=root/'sender'/name;directory.mkdir(parents=True,exist_ok=True)
        (directory/'baseline.bin').write_bytes(baseline);(directory/'candidate.bin').write_bytes(candidate)
        (directory/'visual.c64').write_bytes(visual)
        write_json(directory/'extraction.json',audit)
        c['channel']=dict(baseline_packet_bytes=len(baseline),candidate_packet_bytes=len(candidate),
                         visual_uses=len(visual)//8,baseline_uses=baseline_uses,
                         candidate_uses=len(visual)//8+accounting(len(candidate))['complex_channel_uses'],
                         cap=cap,cap_reference='measured LGVSC 92475' if name=='zea' else 'matched two-key BSU',
                         max_increase_ratio=1.05,visual_channel_reused=True,snr_db=10,metadata_seed=42)
        if c['channel']['candidate_uses']>cap:raise ValueError('rate cap exceeded')
        reused=PARENT/'generation'/name/'baseline'
        c['reuse_baseline']=str(reused) if (reused/'result.json').exists() else None
        paths=[Path(c['source_video']),Path(c['text_report']),run/'received/metadata.bin',
               run/'received/visual.c64',*[run/f'receiver/frames/sample/key_frames_received/{k}.png' for k in c['keys']],
               *directory.iterdir()]
        entry=next(e for e in read_json(c['text_report'])['entries'] if e['prompt']==c['prompt'])
        paths.append(Path(entry['path']))
        if c['reuse_baseline']:
            paths.extend(reused/rel for rel in read_json(reused/'result.json')['artifacts'])
            paths.append(reused/'result.json')
        inputs.update({str(path):sha256(path) for path in paths})
        c.pop('inputs',None);c.pop('motion_packet',None)
        cases.append(c)
        progress(f'PREPARE {name} {c["channel"]["candidate_uses"]}/{cap} channel uses',i,len(selected),start)
    if len({c['source_sha256'] for c in cases})!=len(cases):raise ValueError('duplicate source video')
    code={str(REPO/'src/semantic_transmission'/f):sha256(REPO/'src/semantic_transmission'/f) for f in CODE}
    for rel in ['tests/test_sparse_trajectory.py','scripts/experiment_bsu_trajectory.sh']:
        code[str(REPO/rel)]=sha256(REPO/rel)
    protocol=dict(policy='sparse_trajectory_confidence_and_cadence_v1',cases=cases,arms=ARMS,
        generator=old['generator'],parameters=PARAMETERS,code=code,inputs=inputs,
        training=False,receiver_reads_source=False,selector_changed=False,generator_changed=False,
        receiver_integration='post-VAE RGB refinement from received endpoint keys and paid sparse trajectories',
        primary_comparison='combined vs baseline; cadence and trajectory are component ablations',
        selection='seven distinct sources chosen before new outputs; all parameters frozen; no per-video quality-based choice',
        adoption='weigh equal-video quality and source-paired temporal gains against loss magnitudes; do not require all metrics to improve or fixed22 targets; report worst scene and frozen-motion failures',
        scope='four previously studied scenes plus three transfer sources; three transfer scenes were previously used in a DIFFERENT learned-motion pilot, so not globally unseen data',
        limitations=['single generation seed and SNR; limited multi-source pilot, not universal generalization',
                     'short cases use endpoint windows, not a new full semantic selector evaluation',
                     'sparse tracks cannot describe new appearance or every object/occlusion',
                     'four-frame interpolation can miss genuine rapid motion or pauses',
                     'independent semantic and perceptual stutter labels unavailable'],
        literature=[dict(url='https://arxiv.org/abs/2507.07633',role='sparse trajectory representation idea; not a T-GVC reproduction'),
                    dict(url='https://openaccess.thecvf.com/content/CVPR2024/html/Jeong_OCAI_Improving_Optical_Flow_Estimation_by_Occlusion_and_Consistency_Aware_CVPR_2024_paper.html',role='forward-backward consistency idea; not an OCAI reproduction')])
    protocol['signature']=fingerprint(protocol)
    write_json(root/'protocol.json',protocol)
    for source,digest in code.items():
        target=root/'frozen_code'/Path(source).relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        assert sha256(target)==digest
    print('PREPARE COMPLETE: seven sources, fixed parameters, four arms, bounded +5% rate',flush=True)


def channel(root):
    from .metadata_channel import transmit
    from .sparse_trajectory import FIELD,decode
    p=load(root);start=time.monotonic();rows=[]
    for i,c in enumerate(p['cases'],1):
        destination=root/'received'/c['name'];destination.mkdir(parents=True,exist_ok=True)
        path=destination/'receipt.json'
        if path.exists():
            receipt=read_json(path)
            if receipt['signature']!=p['signature']:raise ValueError('stale channel')
            for name,digest in receipt['artifacts'].items():
                if sha256(destination/name)!=digest:raise ValueError('changed channel artifact')
        else:
            packet=(root/'sender'/c['name']/'candidate.bin').read_bytes()
            received,report=transmit(packet,10,42)
            if received!=packet:raise ValueError('packet failed integrity')
            header,_=unpack(received)
            if FIELD in header:decode(header[FIELD],c['frames'])
            (destination/'metadata.bin').write_bytes(received)
            for j,k in enumerate(c['keys']):
                shutil.copyfile(Path(c['run'])/f'receiver/frames/sample/key_frames_received/{k}.png',destination/f'key{j}.png')
            shutil.copyfile(root/'sender'/c['name']/'visual.c64',destination/'visual.c64')
            receipt=dict(signature=p['signature'],report=report,exact_match=True,
                         received_key_provenance='exact original channel-decoded endpoint PNGs; NTSCC symbols reused',
                         artifacts={q.name:sha256(q) for q in destination.iterdir() if q.is_file()})
            write_json(path,receipt)
        rows.append(dict(case=c['name'],**receipt));progress('CHANNEL verified packets',i,len(p['cases']),start,True)
    write_json(root/'channel.json',dict(signature=p['signature'],rows=rows))


def baseline(root):
    import importlib.metadata
    import numpy as np
    import torch
    from PIL import Image
    from .receiver_alignment_experiment import complete_arm
    from .receiver_joint_experiment import save_pixels
    from .bsu_motion_experiment import setup_torch
    from .generation_noise import NoiseControl
    from .condition_collision import apply_guard
    from .precision_text_cache import Encoder
    from mmengine import Config
    from opensora.registry import MODELS,SCHEDULERS,build_module
    from opensora.datasets.utils import get_transforms_image
    from opensora.utils import inference_utils as inference
    p=load(root);setup_torch(p);cfg=Config(p['generator']);vae=model=None;start=time.monotonic()
    for i,c in enumerate(p['cases'],1):
        directory=root/'generation'/c['name']/'baseline'
        if complete_arm(directory,p['signature']):
            progress('BASELINE verified cases',i,len(p['cases']),start);continue
        directory.mkdir(parents=True,exist_ok=True)
        if c['reuse_baseline']:
            previous=Path(c['reuse_baseline']);receipt=read_json(previous/'result.json')
            for rel,digest in receipt['artifacts'].items():
                origin=previous/rel;target=directory/rel;target.parent.mkdir(parents=True,exist_ok=True)
                if sha256(origin)!=digest:raise ValueError('baseline reuse hash mismatch')
                shutil.copyfile(origin,target)
            write_json(directory/'result.json',dict(status='GENERATED',signature=p['signature'],frames=c['frames'],
                       reused_from=str(previous),artifacts=receipt['artifacts']))
        else:
            if vae is None:
                print('BASELINE loading frozen Open-Sora VAE/STDiT; ETA unknown',flush=True)
                vae=build_module(cfg.vae,MODELS).to('cuda',torch.bfloat16).eval().requires_grad_(False)
                model=build_module(cfg.model,MODELS,input_size=vae.get_latent_size((49,*cfg.image_size)),
                    in_channels=vae.out_channels,caption_channels=4096,model_max_length=300,
                    enable_sequence_parallelism=False).to('cuda',torch.bfloat16).eval().requires_grad_(False)
            runtime=dict(packages={n:importlib.metadata.version(n) for n in ['torch','diffusers','transformers']},
                         cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
            contract=dict(seed=cfg.seed,segments=1,steps=cfg.scheduler.num_sampling_steps,inputs=fingerprint(c),code=p['signature'])
            control=NoiseControl(directory/'generation_noise.json',contract,runtime)
            transform=get_transforms_image(image_size=tuple(cfg.image_size),name='resize_crop')
            with torch.inference_mode():
                refs=[[]]
                for j in range(2):
                    with Image.open(root/'received'/c['name']/f'key{j}.png') as image:
                        x=transform(image.convert('RGB')).unsqueeze(0).unsqueeze(2).to('cuda',torch.bfloat16)
                    with control.scope('vae',j,x):refs[0].append(vae.encode(x).squeeze(0))
                size=vae.get_latent_size((c['frames'],*cfg.image_size));model.config.input_size=size
                z=torch.empty(1,vae.out_channels,*size,device='cuda',dtype=torch.bfloat16)
                with control.scope('initial',0,z):z.copy_(torch.randn_like(z))
                mask,conditions=apply_guard(z,refs,['0;0,1,0,-1,1'],0,5,2,inference)
                initial=z.clone()
                report=copy.deepcopy(read_json(c['text_report']))
                report['prompts']=[c['prompt']];report['prompt_count']=1
                report['entries']=[e for e in report['entries'] if e['prompt']==c['prompt']]
                encoder=Encoder(report,directory/'text_embedding_trace.json');encode=encoder.encode
                encoder.encode=lambda texts:{k:v.to('cuda',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in encode(texts).items()}
                encoder.y_embedder=model.y_embedder
                extra=inference.prepare_multi_resolution_info(cfg.multi_resolution,1,tuple(cfg.image_size),c['frames'],cfg.fps,'cuda',torch.bfloat16)
                scheduler=build_module(cfg.scheduler,SCHEDULERS);counter=[0];sample_start=time.monotonic()
                def tick(module,args):
                    counter[0]+=1
                    if counter[0]==1 or counter[0]%5==0:progress(f'BASELINE {c["name"]} model steps',counter[0],30,sample_start,True)
                hook=model.register_forward_pre_hook(tick)
                try:
                    with control.scope('sampling',0,z):sampled=scheduler.sample(model,encoder,z=z,prompts=[c['prompt']],device='cuda',additional_args=extra,progress=False,mask=mask)
                finally:hook.remove()
                control.finish(1,cfg.scheduler.num_sampling_steps)
                if not torch.equal(initial[:,:,mask[0]==0],sampled[:,:,mask[0]==0]):raise ValueError('hard keys changed')
                decoded=vae.decode(sampled.to(torch.bfloat16),num_frames=c['frames'])
                pixels=(decoded[0].clamp(-1,1).add(1).div(2).mul(255).add(.5).clamp(0,255).permute(1,2,3,0).to('cpu',torch.uint8).numpy())
                save_pixels(directory,pixels,cfg.fps)
                write_json(directory/'conditions.json',dict(conditions=conditions,prompt=c['prompt'],hard_keys_preserved=True,source_pixels_read=False))
                paths=[*sorted((directory/'frames').glob('*.png')),*[directory/n for n in ['reconstruction.mp4','conditions.json','generation_noise.json','text_embedding_trace.json']]]
                write_json(directory/'result.json',dict(status='GENERATED',signature=p['signature'],frames=c['frames'],artifacts={str(q.relative_to(directory)):sha256(q) for q in paths}))
                del refs,x,z,initial,sampled,decoded,pixels,encoder
        progress('BASELINE completed cases',i,len(p['cases']),start)


def refine(root):
    import cv2
    import numpy as np
    from PIL import Image
    from .research_quality import read_frames
    from .receiver_joint_experiment import save_pixels
    from .receiver_alignment_experiment import complete_arm
    from .trajectory_receiver import refine_references,refine_temporal
    from .sparse_trajectory import FIELD,decode
    p=load(root);cv2.setNumThreads(2);cv2.setRNGSeed(2025);start=time.monotonic();done=0
    for c in p['cases']:
        baseline_dir=root/'generation'/c['name']/'baseline'
        if not complete_arm(baseline_dir,p['signature']):raise ValueError('baseline incomplete')
        base=read_frames(baseline_dir/'frames')
        received=root/'received'/c['name'];header,_=unpack((received/'metadata.bin').read_bytes())
        trajectory=decode(header[FIELD],c['frames']) if FIELD in header else None
        keys=[np.asarray(Image.open(received/f'key{j}.png').convert('RGB')) for j in range(2)]
        for arm in ['trajectory','cadence','combined']:
            directory=root/'generation'/c['name']/arm
            if not complete_arm(directory,p['signature']):
                directory.mkdir(parents=True,exist_ok=True);arm_start=time.monotonic()
                def tick(count,total):
                    if count==1 or count%32==0 or count==total:progress(f'REFINE {c["name"]}/{arm} frames',count,total,arm_start,True)
                if arm=='trajectory':pixels,audit=refine_references(base,keys,trajectory,tick,p['parameters'])
                elif arm=='cadence':pixels,audit=refine_temporal(base,tick,p['parameters'])
                else:
                    prior=read_frames(root/'generation'/c['name']/'trajectory/frames')
                    pixels,audit=refine_temporal(prior,tick,p['parameters'])
                if pixels.shape!=base.shape or not np.array_equal(pixels[[0,-1]],base[[0,-1]]):raise ValueError('timeline or endpoint changed')
                save_pixels(directory,pixels,24)
                write_json(directory/'conditions.json',dict(signature=p['signature'],arm=arm,audit=audit,
                           receiver_inputs='baseline decoded RGB, received endpoint keys, received paid trajectory; no source video',
                           baseline_receipt_sha256=sha256(baseline_dir/'result.json'),received_packet_sha256=sha256(received/'metadata.bin'),
                           endpoint_pixels_preserved=True,seconds=time.monotonic()-arm_start))
                paths=[*sorted((directory/'frames').glob('*.png')),directory/'reconstruction.mp4',directory/'conditions.json']
                write_json(directory/'result.json',dict(status='GENERATED',signature=p['signature'],frames=c['frames'],artifacts={str(q.relative_to(directory)):sha256(q) for q in paths}))
            done+=1;progress('REFINE completed arms',done,len(p['cases'])*3,start)


def evaluate(root):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from .bsu_motion_experiment import source_window
    from .research_quality import read_frames
    from .official_quality import OfficialMetrics
    from .stutter_metrics import evaluate as temporal
    from .receiver_alignment_experiment import complete_arm
    from .receiver_alignment_evaluation import comparison_video
    p=load(root);cv2.setNumThreads(2);torch.set_num_threads(2);start=time.monotonic();rows=[];done=0
    for c in p['cases']:
        source=source_window(c['source_video'],*c['keys'])
        arrays={}
        for arm in ARMS:
            directory=root/'generation'/c['name']/arm
            if not complete_arm(directory,p['signature']):raise ValueError('generation incomplete')
            pixels=read_frames(directory/'frames');arrays[arm]=pixels
            cache=directory/'evaluation.json'
            if cache.exists():
                row=read_json(cache)
                if row['signature']!=p['signature'] or row['receipt_sha256']!=sha256(directory/'result.json'):raise ValueError('stale evaluation')
            else:
                metric_start=time.monotonic()
                def tick(count,total):
                    if count==1 or count%128==0 or count==total:progress(f'QUALITY {c["name"]}/{arm}',count,total,metric_start,True)
                quality,per_frame=OfficialMetrics().evaluate(source,pixels,progress=tick)
                diagnostic=temporal(source,pixels)
                row=dict(signature=p['signature'],case=c['name'],group=c['evaluation_group'],arm=arm,
                         quality=quality,temporal=diagnostic,receipt_sha256=sha256(directory/'result.json'),
                         channel_uses=c['channel']['baseline_uses'] if arm=='baseline' else c['channel']['candidate_uses'],
                         rate_note='all nonbaseline ablations charged the same trajectory packet; cadence alone could omit it')
                write_json(directory/'per_frame.json',per_frame);write_json(cache,row)
            rows.append(row);done+=1;progress('EVALUATE completed arms',done,len(p['cases'])*len(ARMS),start)
            print(f'RESULT {c["name"]}/{arm}: PSNR={row["quality"]["psnr_db"]:.3f} LPIPS={row["quality"]["lpips_vgg"]:.5f} flow={row["temporal"]["flow_error_mean"]:.5f} cadence={row["temporal"]["cadence_error_mean"]:.5f} freeze={row["temporal"]["false_freeze_rate"]:.4f}',flush=True)
        review=root/'review'/c['name'];review.mkdir(parents=True,exist_ok=True)
        comparison_video(review/'comparison.mp4',[source,arrays['baseline'],arrays['combined']],['Source','BSU baseline','Trajectory + cadence'],24)
        comparison_video(review/'ablations.mp4',[source,*[arrays[a] for a in ARMS]],['Source',*ARMS],24)
        selected=sorted(set(np.linspace(0,c['frames']-1,7).round().astype(int).tolist()))
        for t in selected:
            Image.fromarray(np.concatenate([source[t],arrays['baseline'][t],arrays['combined'][t]],axis=0)).save(review/f'frame_{t:04d}.png')
        # Predeclared motion strips expose stalls that isolated stills cannot.
        center=c['frames']//2;indices=list(range(max(0,center-4),min(c['frames'],center+5)))
        strips=[np.concatenate([cv2.resize(a[t],(288,160)) for t in indices],axis=1) for a in [source,arrays['baseline'],arrays['combined']]]
        Image.fromarray(np.concatenate(strips,axis=0)).save(review/'motion_strip.png')
        write_json(review/'sample_plan.json',dict(fixed_frames=selected,motion_strip_frames=indices,rows=['source','baseline','combined']))
    contrasts=[]
    for c in p['cases']:
        base=next(r for r in rows if r['case']==c['name'] and r['arm']=='baseline')
        for arm in ARMS[1:]:
            new=next(r for r in rows if r['case']==c['name'] and r['arm']==arm)
            contrasts.append(dict(case=c['name'],group=c['evaluation_group'],arm=arm,
                quality_delta={k:new['quality'][k]-base['quality'][k] for k in ['psnr_db','ssim','lpips_vgg','clip','dists']},
                temporal_delta={k:new['temporal'][k]-base['temporal'][k] for k in ['flow_error_mean','cadence_error_mean','false_freeze_rate','stall_jump_rate','motion_retention_ratio','max_luma_impulse']},
                rate_delta=new['channel_uses']-base['channel_uses']))
    aggregates=[]
    for group in ['all','known_development','transfer_frozen_parameters']:
        for arm in ARMS[1:]:
            subset=[c for c in contrasts if c['arm']==arm and (group=='all' or c['group']==group)]
            aggregates.append(dict(group=group,arm=arm,videos=len(subset),
                mean_quality_delta={k:float(np.mean([s['quality_delta'][k] for s in subset])) for k in subset[0]['quality_delta']},
                mean_temporal_delta={k:float(np.mean([s['temporal_delta'][k] for s in subset])) for k in subset[0]['temporal_delta']},
                improved_video_counts={k:sum(s['quality_delta'][k]*(1 if k in ['psnr_db','ssim','clip'] else -1)>0 for s in subset) for k in subset[0]['quality_delta']}))
    result=dict(status='COMPLETE_PILOT_REVIEW_PENDING',signature=p['signature'],rows=rows,contrasts=contrasts,
                aggregates=aggregates,adopted=False,adoption_review='pending tradeoff and visual review; no all-metric gate',
                rates=[dict(case=c['name'],**c['channel']) for c in p['cases']],limitations=p['limitations'])
    write_json(root/'RESULT.json',result)
    body=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>BSU trajectory and cadence</title><style>body{font:16px system-ui;max-width:1500px;margin:24px auto}video{width:100%}td,th{border:1px solid #bbb;padding:6px}table{border-collapse:collapse}</style><h1>BSU sparse trajectory + cadence pilot</h1><p>Same received keys and baseline generation per case. Paid trajectory transport. No original images at receiver. Review pending.</p>']
    for c in p['cases']:
        body.append(f'<h2>{html.escape(c["name"])}</h2><video controls loop src="review/{c["name"]}/comparison.mp4"></video><table><tr><th>Arm</th><th>PSNR</th><th>LPIPS</th><th>Flow error</th><th>Cadence error</th><th>False freeze</th><th>Channel uses</th></tr>')
        for r in rows:
            if r['case']==c['name']:body.append(f'<tr><td>{r["arm"]}</td><td>{r["quality"]["psnr_db"]:.3f}</td><td>{r["quality"]["lpips_vgg"]:.5f}</td><td>{r["temporal"]["flow_error_mean"]:.5f}</td><td>{r["temporal"]["cadence_error_mean"]:.5f}</td><td>{r["temporal"]["false_freeze_rate"]:.4f}</td><td>{r["channel_uses"]}</td></tr>')
        body.append('</table>')
    (root/'report.html').write_text('\n'.join(body)+ '</html>')


def verify(root):
    from .receiver_alignment_experiment import complete_arm
    p=load(root);checked=0;start=time.monotonic()
    for c in p['cases']:
        for arm in ARMS:
            directory=root/'generation'/c['name']/arm
            if not complete_arm(directory,p['signature']):raise ValueError(f'incomplete {c["name"]}/{arm}')
            files=sorted((directory/'frames').glob('*.png'))
            if len(files)!=c['frames']:raise ValueError('frame count changed')
            probe=read_json(directory/'result.json')
            checked+=len(files)
        receipt=read_json(root/'received'/c['name']/'receipt.json')
        for name,digest in receipt['artifacts'].items():
            if sha256(root/'received'/c['name']/name)!=digest:raise ValueError('received file changed')
        if c['channel']['candidate_uses']>c['channel']['cap']:raise ValueError('rate cap exceeded')
        progress('VERIFY cases',p['cases'].index(c)+1,len(p['cases']),start,True)
    write_json(root/'verification.json',dict(status='PASSED',signature=p['signature'],png_frames=checked,
               cases=len(p['cases']),arms_per_case=len(ARMS),rate_caps_passed=True,channel_exact=True,
               source_and_code_hashes_verified=True,baseline_generation_reused_identically_across_arms=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--stage',choices=['all','prepare','channel','baseline','refine','evaluate','verify'],default='all')
    args=parser.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(REPO/'.local/vendor/Open-Sora'))
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage=='all':
            stages=['prepare','channel','baseline','refine','evaluate','verify'];start=time.monotonic()
            for i,stage in enumerate(stages):
                progress('OVERALL stages',i,len(stages),start)
                write_json(root/'status.json',dict(status='RUNNING',stage=stage,completed=i,total=len(stages)))
                python='/home/sangukbae/anaconda3/envs/lgvsc-channel/bin/python' if stage=='channel' else sys.executable
                env=os.environ.copy()
                if stage=='channel':env['CUDA_VISIBLE_DEVICES']='-1'
                command=[python,'-m','semantic_transmission.bsu_trajectory_experiment','--root',str(root),'--stage',stage]
                # Parent owns lock; child receives a separate marker to skip it.
                env['BSU_TRAJECTORY_CHILD']='1'
                result=subprocess.run(command,env=env)
                if result.returncode:
                    write_json(root/'status.json',dict(status='FAILED',stage=stage,returncode=result.returncode))
                    raise SystemExit(result.returncode)
            progress('OVERALL stages',len(stages),len(stages),start)
            write_json(root/'status.json',dict(status='COMPLETE_PILOT',seconds=time.monotonic()-start))
        else:globals()[args.stage](root)


if __name__=='__main__':
    # Child stages execute under their parent's exclusive run lock.
    if os.environ.get('BSU_TRAJECTORY_CHILD')=='1':
        parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--stage',required=True)
        args=parser.parse_args();sys.path.insert(0,str(REPO/'.local/vendor/Open-Sora'));globals()[args.stage](args.root)
    else:main()
