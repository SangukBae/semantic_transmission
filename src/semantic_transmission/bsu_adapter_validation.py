"""Bounded source-disjoint BSU generation-adapter training and validation.

Uses 33 uniformly selected frames from actual received-key intervals at 160x288.
This is an architecture pilot, not a full-duration 24fps benchmark result.
"""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image
from . import bsu_motion_validation as common
from .artifacts import sha256,write_json

FRAMES=33
HEIGHT,WIDTH=160,288
STEPS=600


def model_config():
    from mmengine import Config
    c=common.pilot_cases()[0];run=Path(c['receiver']['base_packet']).parent.parent
    cfg=common.read(run/'run_config.json');recipe=Config.fromfile(common.REPO/'configs/official_opensora.py')
    for key,name in [('stdit','model'),('vae','vae'),('t5','text_encoder')]:recipe[name]['from_pretrained']=cfg['models'][key]
    recipe.vae['vae_2d_path']=cfg['models']['vae2d']
    recipe.seed=2025
    return recipe


def imports():
    sys.path.insert(0,str(common.REPO/'.local/vendor/Open-Sora'))
    import opensora.models
    from opensora.registry import MODELS,build_module
    return MODELS,build_module


def make_plan(root):
    from .wire import unpack
    from .bsu_sequence_refinement import decode_packet
    out=root/'adapter';out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists():return common.read(out/'plan.json')
    web=[c for c in common.cases() if c['snr']==10]
    available=[c for c in web if int(c['id'].split('_')[1]) not in common.PILOT_NUMBERS]
    train=available[:20];held=available[20:24]
    examples=[]
    def append_case(c,role,limit):
        packet=Path(c['receiver']['base_packet']).read_bytes();h,_=unpack(packet)
        tracks=decode_packet(Path(c['receiver']['trajectory']).read_bytes(),packet) if c['receiver']['trajectory'] else {}
        candidates=[(i,a,b) for i,(a,b) in enumerate(zip(c['keys'],c['keys'][1:])) if b-a+1>=FRAMES and (a,b) in tracks]
        # Fixed sender-side rule: longest existing semantic intervals first.
        for i,a,b in sorted(candidates,key=lambda row:(-(row[2]-row[1]),row[1]))[:limit]:
            indices=np.linspace(a,b,FRAMES).round().astype(int).tolist()
            examples.append(dict(name=f'{c["id"]}_{i:03d}',id=c['id'],role=role,dataset='kinetics' if c['id'].startswith('kinetics') else 'webvid',
                start=a,end=b,source_indices=indices,source_directory=str(common.source_directory(c)),
                key_paths=c['receiver']['key_paths'][i:i+2],packet=c['receiver']['base_packet'],trajectory=c['receiver']['trajectory'],
                text=h['segments'][i]['text'],flow=h['segments'][i]['flow'],
                source_sha256=c['source_sha256'],effective_fps=(FRAMES-1)*24/(b-a)))
    for c in train:append_case(c,'train',2)
    for c in held:append_case(c,'heldout_webvid',1)
    kinroot=common.REPO/'outputs/bsu_kinetics_context/benchmark';kinplan=common.read(kinroot/'protocol.json')
    for v in kinplan['videos'][:4]:
        base=kinroot/'snr_10'/v['id'];run=base/'base/reconstruction/event_pairs/run'
        h,_=unpack((run/'received/metadata.bin').read_bytes());keys=[k['index'] for k in h['keyframes']]
        prep=common.read(v['prepared']);trajectory=base/'received/trajectory.bin'
        c=dict(id=v['id'],snr=10,keys=keys,frames=v['frames'],source_movie=prep['normalized_video'],source_sha256=v['source_sha256'],
            receiver=dict(base_packet=str(run/'received/metadata.bin'),trajectory=str(trajectory) if trajectory.exists() else None,
                key_paths=[str(run/f'receiver/frames/sample/key_frames_received/{k}.png') for k in keys]))
        append_case(c,'heldout_kinetics',1)
    if len([e for e in examples if e['role']=='train'])<16 or len([e for e in examples if e['role']!='train'])<4:
        raise ValueError('insufficient disjoint received-trajectory examples')
    roles={r:{e['source_sha256'] for e in examples if e['role']==r} for r in {e['role'] for e in examples}}
    if any(roles['train']&s for r,s in roles.items() if r!='train'):raise ValueError('training/evaluation source leak')
    value=dict(status='PREREGISTERED_ARCHITECTURE_PILOT',examples=examples,steps=STEPS,seed=2025,
        frames=FRAMES,height=HEIGHT,width=WIDTH,lr=.0003,condition_dropout=.1,
        training='frozen pretrained OpenSora, small zero-initialized patch residual adapter; no backbone weight updates',
        temporal_scope='33 uniform frames of each complete received-key interval; effective fps preserves interval duration',
        limitation='not full native-fps videos; nominal condition pooling follows each VAE microblock; no full BSU performance claim',
        checkpoint_selection='last fixed training step; evaluation videos never select checkpoint or strength',
        criteria=common.read(root/'flow_reconstruction/plan.json')['acceptance'],
        inference_inputs='received keys, received quantized tracks, received text/flow, timestamps; no source pixels',
        model_paths=dict(model_config()['model']),code={str(Path(__file__).relative_to(common.REPO)):sha256(Path(__file__))})
    write_json(out/'plan.json',value);return value


def prepare_text(root):
    from . import precision_text_cache as cache
    plan=make_plan(root);out=root/'adapter/text';(out/'receiver').mkdir(parents=True,exist_ok=True)
    with (out/'receiver/metadata.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','text','flow']);w.writeheader()
        w.writerows(dict(path=f'clips/sample/{i:05d}.mp4',text=e['text'],flow=e['flow']) for i,e in enumerate(plan['examples']))
    recipe=model_config();cache.prepare(out,common.REPO,recipe.pretty_text,'fp32')
    print('ADAPTER text conditions prepared',flush=True)


def prepare_data(root):
    import torch
    import cv2
    from .trajectory_adapter import condition_maps
    from .bsu_sequence_refinement import decode_packet
    from .precision_text_cache import load
    MODELS,build=imports();recipe=model_config();plan=make_plan(root);out=root/'adapter'
    cv2.setNumThreads(2);torch.manual_seed(2025)
    vae=build(recipe.vae,MODELS).cuda().to(torch.bfloat16).eval()
    report=common.read(out/'text/receiver/text_embeddings.json');entries={e['prompt']:e for e in report['entries']}
    receipts=[]
    for i,e in enumerate(plan['examples']):
        dest=out/'data'/f'{e["name"]}.pt';dest.parent.mkdir(exist_ok=True)
        received_path=out/'received_only'/f'{e["name"]}.pt';received_path.parent.mkdir(exist_ok=True)
        if dest.exists() and received_path.exists():
            receipts.append(dict(name=e['name'],path=str(dest),sha256=sha256(dest),received_path=str(received_path),received_sha256=sha256(received_path)));continue
        def rgb(p):return cv2.resize(np.asarray(Image.open(p).convert('RGB')),(WIDTH,HEIGHT),interpolation=cv2.INTER_AREA)
        source=np.stack([rgb(Path(e['source_directory'])/f'{t}.png') for t in e['source_indices']])
        keys=np.stack([rgb(p) for p in e['key_paths']])
        tracks=decode_packet(Path(e['trajectory']).read_bytes(),Path(e['packet']).read_bytes())[e['start'],e['end']]
        indices=[t-e['start'] for t in e['source_indices']]
        condition=condition_maps(tracks,keys,indices,HEIGHT//8,WIDTH//8)
        tensor=lambda x:torch.from_numpy(x.copy()).permute(3,0,1,2)[None].cuda().to(torch.bfloat16)/127.5-1
        with torch.inference_mode():
            torch.manual_seed(2025+i);target=vae.encode(tensor(source))
            key_latents=torch.cat([vae.encode(tensor(k[None])) for k in keys],dim=2)
        text=load(entries[report['prompts'][i]],report['contract'],report['prompts'][i])
        received=dict(keys=torch.from_numpy(keys),key_latents=key_latents.cpu(),condition=torch.from_numpy(condition)[None],
            y=text['y'],mask=text['mask'],fps=e['effective_fps'])
        torch.save(dict(target=target.cpu(),source=torch.from_numpy(source),received=received,example=e),dest)
        torch.save(received,received_path)
        receipts.append(dict(name=e['name'],path=str(dest),sha256=sha256(dest),received_path=str(received_path),received_sha256=sha256(received_path)))
        common.progress('ADAPTER latent preparation',i+1,len(plan['examples']))
    write_json(out/'data.json',dict(receipts=receipts,receiver_fields=['keys','key_latents','condition','y','mask','fps'],
        provenance=[dict(path=e['packet'],sha256=sha256(e['packet'])) for e in plan['examples']]))


def load_model():
    import torch
    MODELS,build=imports();recipe=model_config()
    cfg=dict(recipe.model);cfg['enable_layernorm_kernel']=False
    model=build(cfg,MODELS,input_size=(9,HEIGHT//8,WIDTH//8),in_channels=4,caption_channels=4096,model_max_length=300,
        enable_sequence_parallelism=False).cuda().to(torch.bfloat16).eval()
    model.requires_grad_(False)
    return model


def arguments(received):
    import torch
    return dict(y=received['y'].cuda().to(torch.bfloat16),mask=received['mask'].cuda(),
        fps=torch.tensor([received['fps']],device='cuda',dtype=torch.bfloat16),
        height=torch.tensor([HEIGHT],device='cuda',dtype=torch.bfloat16),width=torch.tensor([WIDTH],device='cuda',dtype=torch.bfloat16))


def train(root):
    import torch
    from .trajectory_adapter import TrajectoryAdapter,POLICY
    plan=make_plan(root);out=root/'adapter'
    if (out/'training.json').exists():
        old=common.read(out/'training.json')
        if old['status']=='COMPLETE' and old['checkpoint_sha256']==sha256(out/'adapter.pt'):
            checkpoint=torch.load(out/'adapter.pt',map_location='cpu',weights_only=True)
            if checkpoint['plan_sha256']==sha256(out/'plan.json'):
                print('ADAPTER verified completed training reused',flush=True);return
        raise ValueError('training receipt mismatch')
    model=load_model()
    from opensora.acceleration.checkpoint import set_grad_checkpoint
    set_grad_checkpoint(model)
    adapter=TrajectoryAdapter(hidden=model.hidden_size).cuda();handle=adapter.attach(model)
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=plan['lr'],weight_decay=.01)
    examples=[e for e in plan['examples'] if e['role']=='train']
    dataset=[torch.load(out/'data'/f'{e["name"]}.pt',map_location='cpu',weights_only=True) for e in examples]
    rng=np.random.default_rng(plan['seed']);torch.manual_seed(plan['seed'])
    losses=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    # Persist a real backward pass before the bounded run; OOM is actionable.
    for step in range(plan['steps']):
        sample=dataset[int(rng.integers(len(dataset)))];received=sample['received'];target=sample['target'].cuda().float()
        noise=torch.randn_like(target);t=torch.rand(1,device='cuda')*900+50
        z=(1-t[:,None,None,None,None]/1000)*target+(t[:,None,None,None,None]/1000)*noise
        keys=received['key_latents'].cuda().float();z[:,:,0]=keys[:,:,0];z[:,:,-1]=keys[:,:,1]
        mask=torch.ones((1,z.shape[2]),device='cuda',dtype=torch.bool);mask[:,[0,-1]]=False
        condition=received['condition'].cuda().float()
        if rng.random()<plan['condition_dropout']:condition=torch.zeros_like(condition)
        adapter.condition=condition
        kwargs=arguments(received);kwargs['x_mask']=mask
        if step==0:
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                adapter.condition=None;baseline=model(z.to(torch.bfloat16),t,**kwargs)
                adapter.condition=condition;zero=model(z.to(torch.bfloat16),t,**kwargs)
            if not torch.equal(baseline,zero):raise ValueError('zero adapter changes frozen backbone output')
            del baseline,zero
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred=model(z.to(torch.bfloat16),t,**kwargs).chunk(2,dim=1)[0]
            loss=(pred.float()[:,:,1:-1]-(target-noise)[:,:,1:-1]).square().mean()
        if not torch.isfinite(loss):raise ValueError('adapter training loss nonfinite')
        loss.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.);optimizer.step()
        losses.append(float(loss))
        if step==0:
            write_json(out/'memory_preflight.json',dict(status='FORWARD_BACKWARD_PASSED',parameters=sum(p.numel() for p in adapter.parameters()),
                frozen_backbone=not any(p.requires_grad for p in model.parameters()),zero_adapter_identity=True,
                peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30))
        if step%25==0 or step+1==plan['steps']:common.progress('ADAPTER training',step+1,plan['steps'])
    handle.remove();torch.save(dict(policy=POLICY,state_dict=adapter.state_dict(),steps=plan['steps'],plan_sha256=sha256(out/'plan.json')),out/'adapter.pt')
    write_json(out/'training.json',dict(status='COMPLETE',steps=plan['steps'],losses=losses,seconds=time.monotonic()-start,
        peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30,checkpoint_sha256=sha256(out/'adapter.pt'),
        source_disjoint=True,training_source_count=len({e['source_sha256'] for e in examples}),
        no_validation_based_checkpoint_selection=True))


def generate(root):
    import torch
    import cv2
    from .trajectory_adapter import TrajectoryAdapter
    from .generation_noise import tensor_hash,NoiseControl
    from .trajectory_receiver import refine_references,refine_temporal
    from .receiver_key_context import preserve_context
    from .bsu_sequence_refinement import decode_packet
    from .bsu_receiver import encode_frames
    MODELS,build=imports();plan=make_plan(root);out=root/'adapter';model=load_model();recipe=model_config()
    vae=build(recipe.vae,MODELS).cuda().to(torch.bfloat16).eval()
    adapter=TrajectoryAdapter(hidden=model.hidden_size).cuda().eval()
    checkpoint=torch.load(out/'adapter.pt',map_location='cpu',weights_only=True);adapter.load_state_dict(checkpoint['state_dict']);handle=adapter.attach(model)
    from opensora.schedulers.rf import RFLOW
    from opensora.utils.inference_utils import prepare_multi_resolution_info
    from .short_schedule import transform
    from opensora.schedulers.rf.rectified_flow import timestep_transform
    class Text:
        def __init__(self,r):self.r=r
        def encode(self,prompts):return dict(y=self.r['y'].cuda().to(torch.bfloat16),mask=self.r['mask'].cuda())
        def null(self,n):return model.y_embedder.y_embedding[None].repeat(n,1,1)[:,None]
    evals=[e for e in plan['examples'] if e['role']!='train'];receipts=[]
    for ei,e in enumerate(evals):
        received_path=out/'received_only'/f'{e["name"]}.pt'
        r=torch.load(received_path,map_location='cpu',weights_only=True)
        if set(r)!={'keys','key_latents','condition','y','mask','fps'}:raise ValueError('unexpected inference field')
        keys=r['key_latents'].cuda().to(torch.bfloat16)
        tracks=decode_packet(Path(e['trajectory']).read_bytes(),Path(e['packet']).read_bytes())[e['start'],e['end']]
        indices=np.asarray(e['source_indices'])-e['start'];small_tracks=dict(tracks,xy=tracks['xy'][indices],valid=tracks['valid'][indices],frames=FRAMES)
        for arm in ('baseline','adapter'):
            dest=out/'evaluation'/e['name']/arm;dest.mkdir(parents=True,exist_ok=True)
            receipt=dest/'generation.json'
            if receipt.exists():
                old=common.read(receipt)
                if old['checkpoint_sha256']!=sha256(out/'adapter.pt') or old['received_sha256']!=sha256(received_path):raise ValueError('generation receipt mismatch')
                for label in ('raw','post'):
                    movie='raw.mp4' if label=='raw' else 'combined_context.mp4'
                    if old[label+'_sha256']!=sha256(dest/movie):raise ValueError('generation output changed')
                receipts.append(old);continue
            scheduler=RFLOW(num_sampling_steps=30,cfg_scale=7.,use_timestep_transform=True)
            torch.manual_seed(2025+ei);z=torch.randn(1,4,len(r['condition'][0,0]),HEIGHT//8,WIDTH//8,device='cuda',dtype=torch.bfloat16)
            initial=tensor_hash(z);z[:,:,0]=keys[:,:,0];z[:,:,-1]=keys[:,:,1]
            mask=torch.ones((1,z.shape[2]),device='cuda',dtype=torch.bfloat16);mask[:,[0,-1]]=0
            extra=prepare_multi_resolution_info('STDiT2',1,(HEIGHT,WIDTH),FRAMES,r['fps'],'cuda',torch.bfloat16)
            adapter.condition=r['condition'].cuda().float() if arm=='adapter' else None
            noise=NoiseControl(dest/'sampling_noise.json',dict(seed=2025+ei),dict(scope='sampling_only_shared_cached_key_latents'))
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),noise.scope('sampling',0,z):
                result=scheduler.sample(model,Text(r),z,['received'],device='cuda',additional_args=extra,mask=mask,progress=False)
            if len(noise.records[0]['draws'])!=30:raise ValueError('unexpected sampler noise count')
            noise.save('PASSED_SAMPLING_ONLY')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                decoded=vae.decode(result.to(torch.bfloat16),num_frames=FRAMES)[0].float().cpu()
            rgb=((decoded.permute(1,2,3,0).numpy()+1)*127.5).round().clip(0,255).astype(np.uint8)
            cv2.setRNGSeed(2025)
            keyrgb=r['keys'].numpy();post,_=refine_references(rgb,keyrgb,small_tracks);post,_=refine_temporal(post);post,_=preserve_context(post,keyrgb)
            if not np.array_equal(post[[0,-1]],keyrgb):raise ValueError('received endpoint changed')
            for label,frames in [('raw',rgb),('combined_context',post)]:
                folder=dest/label;folder.mkdir(exist_ok=True)
                for t,f in enumerate(frames):Image.fromarray(f).save(folder/f'{t:05d}.png')
                encode_frames(folder,dest/f'{label}.mp4',FRAMES,24)
            record=dict(example=e['name'],arm=arm,initial_noise_sha256=initial,receiver_reads_source=False,
                raw_sha256=sha256(dest/'raw.mp4'),post_sha256=sha256(dest/'combined_context.mp4'),
                received_sha256=sha256(received_path),checkpoint_sha256=sha256(out/'adapter.pt'),sampling_draws=noise.records,
                clip_display_fps=24,semantic_effective_fps=r['fps'],samples=FRAMES)
            write_json(receipt,record);receipts.append(record)
        pair=[r for r in receipts if r['example']==e['name']]
        if len(pair)!=2 or any(pair[0][k]!=pair[1][k] for k in ('initial_noise_sha256','sampling_draws','received_sha256')):raise ValueError('paired inference inputs/noise differ')
        common.progress('ADAPTER paired generation',ei+1,len(evals))
    handle.remove();write_json(out/'generation.json',dict(status='COMPLETE',rows=receipts))


def evaluate(root):
    import torch
    import statistics as st
    from .bsu_receiver_improvement import PersistentMetrics
    from .stutter_metrics import evaluate as temporal
    from .research_quality import read_video
    plan=make_plan(root);out=root/'adapter';metrics=PersistentMetrics();rows=[]
    for e in [e for e in plan['examples'] if e['role']!='train']:
        data=torch.load(out/'data'/f'{e["name"]}.pt',map_location='cpu',weights_only=True);source=data['source'].numpy()
        for arm in ('baseline','adapter'):
            for kind in ('raw','combined_context'):
                folder=out/'evaluation'/e['name']/arm
                # Both source and reconstruction use lossless arrays here; the
                # MP4s are review artifacts only, at an explicitly marked fps.
                generated=np.stack([np.asarray(Image.open(folder/kind/f'{i:05d}.png').convert('RGB')) for i in range(FRAMES)])
                q,_=metrics.evaluate(source,generated)
                rows.append(dict(name=e['name'],role=e['role'],arm=arm,kind=kind,quality=q,temporal=temporal(source,generated)))
    means={}
    for role in sorted({r['role'] for r in rows}):
        means[role]={}
        for kind in ('raw','combined_context'):
            means[role][kind]={arm:dict(**{k:st.mean(r['quality'][k] for r in rows if r['role']==role and r['kind']==kind and r['arm']==arm) for k in ('psnr_db','lpips_vgg','ssim','dists','clip')},
                **{k:st.mean(r['temporal'][k] for r in rows if r['role']==role and r['kind']==kind and r['arm']==arm) for k in ('flow_error_mean','cadence_error_mean','false_freeze_rate')}) for arm in ('baseline','adapter')}
    result=dict(status='COMPLETE_SOURCE_DISJOINT_ADAPTER_ARCHITECTURE_PILOT',means=means,rows=rows,
        training=common.read(out/'training.json'),no_additional_transmission=True,
        scope=plan['temporal_scope'],limitation=plan['limitation'],production_default_changed=False)
    write_json(out/'RESULT.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('rows','training')},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['plan','run','prepare_text','prepare_data','train','generate','evaluate']);p.add_argument('--root',type=Path,default=common.ROOT)
    a=p.parse_args()
    if a.stage=='plan':
        plan=make_plan(a.root);print({r:len([e for e in plan['examples'] if e['role']==r]) for r in {e['role'] for e in plan['examples']}});return
    if a.stage!='run':return globals()[a.stage](a.root)
    for stage in ('prepare_text','prepare_data','train','generate','evaluate'):
        subprocess.run([sys.executable,'-m',__package__+'.bsu_adapter_validation',stage,'--root',str(a.root)],check=True)


if __name__=='__main__':main()
