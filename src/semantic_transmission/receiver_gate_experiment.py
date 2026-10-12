"""Train and test a small history gate with source-disjoint counterfactual labels."""
import argparse
import copy
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

from . import receiver_reference_gate as gate, receiver_alignment_experiment as base
from . import auto_extraction as ae, precision_text_cache as text_cache
from .artifacts import sha256, write_json
from .webvid5 import fingerprint, read_json
from .validation_progress import emit
from .sender_repair_experiment import launch
from .qwen35_pairs_video import freeze
from .webvid_ablation import Stages, environment
from .etri_60s_check import lock

REPO=Path(__file__).resolve().parents[2]
MODULE='semantic_transmission.receiver_gate_experiment'
DEFAULT_OUTPUT=REPO/'outputs/receiver_gate_20261009'
OWN_CODE=('receiver_reference_gate.py','receiver_gate_experiment.py','receiver_gate_evaluation.py',
          'receiver_alignment_evaluation.py','receiver_joint_experiment.py','sender_repair_experiment.py',
          'official_quality.py','research_quality.py','webvid_ablation.py','validation_progress.py')


def inventory(root):
    import numpy as np
    test=base.cases(REPO)
    test[0]['run']=str(REPO/'outputs/lgvsc_qwen_zea_16s_20261008/qwen/reconstruction/event_pairs/run')
    selected=[dict(base.prepare_case(c),split='test',source_id=c['name']) for c in test]
    cfg,protocol=base.make_protocol(REPO,selected)
    tv=REPO/'outputs/qwen35_pairs_caption_comparison_20261007/reconstruction/event_pairs/run'
    keys=read_json(tv/'keyframes.json')['indices']
    eligible=[];stop=-1
    for j in range(len(keys)-2):
        triple=keys[j:j+3]
        if triple[0]>=stop and all(16<=b-a<=24 for a,b in zip(triple,triple[1:])):
            eligible.append(triple);stop=triple[-1]
    if len(eligible)<10:raise ValueError('insufficient disjoint training windows')
    for n,j in enumerate(np.linspace(0,len(eligible)-1,10).round().astype(int)):
        c=dict(name=f'train_tv_{n:02d}',run=str(tv),keys=eligible[j],scene='TVSum source-disjoint training window')
        selected.append(dict(base.prepare_case(c),split='train',source_id='tvsum_tv_low_08'))
    for name in ('miami-surf','rallye'):
        run=REPO/f'outputs/metric_validation_20260916_v1/03_natural/lgvsc/davis__{name}/steps_30/attempt_001/davis__{name}'
        keys=read_json(run/'keyframes.json')['indices']
        if len(keys)!=3:raise ValueError('DAVIS training case must have exactly two received intervals')
        inputs={str(run/f'receiver/frames/sample/key_frames_received/{k}.png'):sha256(run/f'receiver/frames/sample/key_frames_received/{k}.png') for k in keys}
        inputs[str(run/'receiver/metadata.csv')]=sha256(run/'receiver/metadata.csv')
        selected.append(dict(name='train_'+name,run=str(run),keys=keys,split='train',source_id='davis_'+name,
            scene='DAVIS source-disjoint training video',inputs=inputs,prepare_text=True,
            lengths=base.segment_lengths([k-keys[0] for k in keys])))
    for c in selected:
        source=Path(c['run'])/'data/normalized.mp4'
        c.update(source_video=str(source),source_sha256=sha256(source))
        c['inputs'][str(source)]=c['source_sha256']
        if not c.get('prepare_text'):
            c['text_report']=str(Path(c['run'])/'receiver/text_embeddings.json')
    gate.assert_source_split(selected)
    protocol.update(policy='receiver_learned_history_gate_pilot_v1',cases=selected,
        arms=['baseline','constant_edit','learned_gate'],edit_ratio=gate.EDIT_RATIO,
        gate_features=list(gate.FEATURES),training_planned=True,
        training=dict(type='counterfactual_utility_MLP',steps=400,seed=31415,threshold=.05,
            loss='source-balanced SmoothL1',utility='relative affected-motion gain minus positive LPIPS/.01 and PSNR/.3 regressions, clipped [-2,2]',
            splits='original source identity, not adjacent frames or differently normalized files',
            leave_one_source_out_diagnostic=True,heldout_used_for_training_or_tuning=False),
        modified_mask='older four history latents: edit 0 or .35; newest history latent and end key remain hard fixed',
        source_pixels_used_for_generation=False,source_pixels_used_for_offline_labels=True,
        output_pixel_interpolation=False,additional_transmitted_information=False,
        scope='12 training windows from 3 originals; 4 known development scenes excluded from training; single seed/SNR; two segments with restarted context',
        adoption_gate=dict(flow_ratio_max=.95,psnr_drop_max=.3,lpips_increase_max=.01,mad_max_ratio_max=1.05))
    for name in OWN_CODE:
        path=REPO/'src/semantic_transmission'/name;protocol['code'][str(path)]=sha256(path)
    path=REPO/'scripts/experiment_receiver_gate.sh';protocol['code'][str(path)]=sha256(path)
    protocol['signature']=fingerprint(protocol)
    return protocol


def load(root, prepared=True):
    p=read_json(root/'plan.json');check=dict(p);signature=check.pop('signature')
    if fingerprint(check)!=signature:raise ValueError('plan signature changed')
    for name,digest in p['code'].items():
        if sha256(name)!=digest:raise ValueError(f'frozen code changed: {name}')
    for c in p['cases']:
        for name,digest in c['inputs'].items():
            if sha256(name)!=digest:raise ValueError(f'frozen input changed: {name}')
    if prepared:
        q=read_json(root/'prepared.json')
        if q['signature']!=signature:raise ValueError('prepared identity differs')
        p['cases']=q['cases']
    return p


def prepare(root):
    p=load(root,False)
    pending=[c for c in p['cases'] if c.get('prepare_text')]
    work=root/'text_inputs';(work/'receiver').mkdir(parents=True,exist_ok=True)
    rows=[]
    for c in pending:
        with (Path(c['run'])/'receiver/metadata.csv').open() as stream:
            c['text_offset']=len(rows);rows.extend(list(csv.DictReader(stream)))
    with (work/'receiver/metadata.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['path','text','flow']);writer.writeheader();writer.writerows(rows)
    cfg_text=(Path(next(c for c in p['cases'] if c['split']=='test')['run'])/'receiver/decoder_config.py').read_text()
    print(f'TEXT preparing {len(rows)} frozen DAVIS captions; no caption model or new transmitted text',flush=True)
    old=text_cache.legacy.save_entry;counter=[0];started=time.monotonic()
    def save(*args,**kwargs):
        value=old(*args,**kwargs);counter[0]+=1
        emit(counter[0],len(rows))
        print(f'TEXT {counter[0]}/{len(rows)} ({100*counter[0]/len(rows):.1f}%) elapsed={time.monotonic()-started:.0f}s',flush=True)
        return value
    text_cache.legacy.save_entry=save
    try:text_cache.prepare(work,REPO,cfg_text,'fp32')
    finally:text_cache.legacy.save_entry=old
    report=read_json(work/'receiver/text_embeddings.json')
    for c in pending:
        c['prompts']=report['prompts'][c['text_offset']:c['text_offset']+2]
        c['text_report']=str(work/'receiver/text_embeddings.json')
    write_json(root/'prepared.json',dict(signature=p['signature'],cases=p['cases']))


def save_video(directory, pixels, fps):
    from .receiver_joint_experiment import save_pixels
    save_pixels(directory,pixels,fps)


def arm_complete(directory,signature):
    return base.complete_arm(directory,signature)


def generate(root,split):
    import torch
    import numpy as np
    from PIL import Image
    from mmengine import Config
    from opensora.registry import MODELS,SCHEDULERS,build_module
    from opensora.datasets.utils import get_transforms_image
    from opensora.utils import inference_utils as inference
    from .generation_noise import NoiseControl,tensor_hash
    from .temporal import conditioning_indices,concatenate_segments
    p=load(root);signature=p['signature'];cfg=Config(p['generator'])
    cases=[c for c in p['cases'] if c['split']==split]
    arms=['baseline','constant_edit'] if split=='train' else p['arms']
    checkpoint=torch.load(root/'gate.pt',map_location='cpu',weights_only=True) if split=='test' else None
    if split=='test' and (checkpoint['protocol_signature']!=signature
            or sha256(root/'gate.pt')!=read_json(root/'training.json')['checkpoint_sha256']):
        raise ValueError('gate checkpoint identity or bytes changed')
    if split=='test' and set(checkpoint['training_source_ids']) & {c['source_id'] for c in cases}:
        raise ValueError('held-out source leaked into gate training')
    if all(arm_complete(root/split/c['name']/arm,signature) for c in cases for arm in arms):
        print('RESUME all generation receipts verified; no models loaded',flush=True);return
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.manual_seed(cfg.seed);torch.cuda.manual_seed_all(cfg.seed)
    device,dtype='cuda',torch.bfloat16
    runtime=dict(packages={n:importlib.metadata.version(n) for n in ('torch','diffusers','transformers')},
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
    print(f'LOAD frozen VAE/STDiT for {split}; initialization ETA unknown',flush=True)
    vae=build_module(cfg.vae,MODELS).to(device,dtype).eval().requires_grad_(False)
    model=build_module(cfg.model,MODELS,input_size=vae.get_latent_size((25,*cfg.image_size)),in_channels=vae.out_channels,
        caption_channels=4096,model_max_length=300,enable_sequence_parallelism=False).to(device,dtype).eval().requires_grad_(False)
    transform=get_transforms_image(image_size=tuple(cfg.image_size),name='resize_crop')
    steps=cfg.scheduler.num_sampling_steps;total=len(cases)*(1+len(arms))*steps;done=0;started=time.monotonic()
    def to_pixels(x):
        return x[0].clamp(-1,1).add(1).div(2).mul(255).add(.5).clamp(0,255).permute(1,2,3,0).to(torch.uint8).numpy()
    def text_encoder(c,path):
        report=copy.deepcopy(read_json(c['text_report']));report['prompts']=c['prompts'];report['prompt_count']=2
        report['entries']=[e for e in report['entries'] if e['prompt'] in c['prompts']]
        encoder=text_cache.Encoder(report,path);original=encoder.encode
        encoder.encode=lambda texts:{k:v.to(device=device,dtype=dtype if v.is_floating_point() else v.dtype) for k,v in original(texts).items()}
        encoder.y_embedder=model.y_embedder
        return encoder
    def sample(c,loop,refs,strategy,control,encoder,ratio,arm):
        nonlocal done
        frames=c['lengths'][loop];latent_size=vae.get_latent_size((frames,*cfg.image_size));model.config.input_size=latent_size
        z=torch.empty(1,vae.out_channels,*latent_size,device=device,dtype=dtype)
        with control.scope('initial',loop,z):z.copy_(torch.randn_like(z))
        mask,record=base.mask_for_arm(z,refs,strategy,loop,'legacy_aligned',inference,None)
        mask,slots=gate.relax_history(mask,record,ratio)
        initial=z.clone();scheduler=build_module(cfg.scheduler,SCHEDULERS)
        extra=inference.prepare_multi_resolution_info(cfg.multi_resolution,1,tuple(cfg.image_size),frames,cfg.fps,device,dtype)
        step=[0]
        def progress(module,inputs):
            step[0]+=1
            if step[0]==1 or step[0]%5==0:
                current=done+step[0]-1;emit(current,total)
                print(f'GENERATE {split} {current}/{total} ({100*current/total:.1f}%) {c["name"]}/{arm} '
                      f'segment {loop+1}/2 step {step[0]}/{steps} elapsed={time.monotonic()-started:.0f}s',flush=True)
        handle=model.register_forward_pre_hook(progress)
        try:
            with control.scope('sampling',loop,z):
                latent=scheduler.sample(model,encoder,z=z,prompts=[c['prompts'][loop]],device=device,
                    additional_args=extra,progress=False,mask=mask)
        finally:handle.remove()
        if step[0]!=steps or not torch.isfinite(latent).all():raise ValueError('invalid diffusion output')
        if not torch.equal(latent[:,:,mask[0]==0],initial[:,:,mask[0]==0]):raise ValueError('hard condition changed')
        done+=steps;emit(done,total)
        record.update(edit_ratio=ratio,editable_history_slots=slots,mask=mask.cpu().float().tolist(),
            fixed_anchors_verified=True,initial_condition_hash=tensor_hash(initial))
        return vae.decode(latent.to(dtype),num_frames=frames).cpu(),record
    with torch.inference_mode():
        for c in cases:
            directory=root/split/c['name'];directory.mkdir(parents=True,exist_ok=True)
            if all(arm_complete(directory/a,signature) for a in arms):
                done+=(1+len(arms))*steps;emit(done,total);continue
            contract=dict(seed=cfg.seed,segments=2,steps=steps,inputs=fingerprint(c),code=signature,
                reference_policy='last_17_frames_before_vae')
            common=directory/'common.pt';common_receipt=directory/'common.json'
            if common_receipt.exists():
                receipt=read_json(common_receipt)
                if receipt['signature']!=signature or receipt['sha256']!=sha256(common):raise ValueError('common context changed')
                shared=torch.load(common,map_location='cpu',weights_only=True);done+=steps;emit(done,total)
            else:
                encoder=text_encoder(c,directory/'common_text_trace.json')
                control=NoiseControl(directory/'common_noise.json',contract,runtime)
                refs=[[]]
                for i,key in enumerate(c['keys']):
                    with Image.open(Path(c['run'])/f'receiver/frames/sample/key_frames_received/{key}.png') as im:
                        x=transform(im.convert('RGB')).unsqueeze(0).unsqueeze(2).to(device,dtype)
                    with control.scope('vae',i,x):refs[0].append(vae.encode(x).squeeze(0))
                first,record=sample(c,0,refs,'0;0,1,0,-1,1;1,2,0,-1,1',control,encoder,0.,'shared_first')
                history=first[:,:,conditioning_indices(first.shape[2])]
                received=[]
                for key in c['keys'][1:]:
                    with Image.open(Path(c['run'])/f'receiver/frames/sample/key_frames_received/{key}.png') as im:
                        received.append(np.asarray(im.convert('RGB')))
                vector=gate.features(to_pixels(history),*received,[b-a for a,b in zip(c['keys'],c['keys'][1:])],cfg.fps)
                shared=dict(first=first,refs=[x.cpu() for x in refs[0]],features=vector,records=control.records,condition=record)
                temp=common.with_suffix('.partial.pt');torch.save(shared,temp);temp.replace(common)
                write_json(common_receipt,dict(signature=signature,sha256=sha256(common),features=vector,
                    feature_inputs='previous generated tail17 and two received key images, gaps/fps only',source_pixels_used=False))
            for arm in arms:
                dest=directory/arm
                if arm_complete(dest,signature):done+=steps;emit(done,total);continue
                dest.mkdir(parents=True,exist_ok=True)
                ratio=0. if arm=='baseline' else gate.EDIT_RATIO
                prediction=None
                if arm=='learned_gate':prediction,ratio=gate.predict(checkpoint,shared['features'])
                expected=read_json(directory/'baseline/generation_noise.json') if arm!='baseline' else None
                control=NoiseControl(dest/'generation_noise.json',contract,runtime,expected)
                control.records=copy.deepcopy(shared['records']);control.save('RUNNING')
                encoder=text_encoder(c,dest/'text_trace.json')
                refs=[[x.to(device,dtype) for x in shared['refs']]]
                history=shared['first'][:,:,conditioning_indices(shared['first'].shape[2])].to(device,dtype)
                with control.scope('vae',3,history):refs[0].append(vae.encode(history).squeeze(0))
                second,record=sample(c,1,refs,'0;0,1,0,-1,1;1,2,0,-1,1;1,3,-5,0,5,0',control,encoder,ratio,arm)
                control.finish(2,steps)
                pixels=to_pixels(concatenate_segments([shared['first'][0],second[0]]).unsqueeze(0))
                if len(pixels)!=c['keys'][-1]-c['keys'][0]+1:raise ValueError('output timeline changed')
                save_video(dest,pixels,cfg.fps)
                write_json(dest/'conditions.json',[shared['condition'],record])
                write_json(dest/'gate_decision.json',dict(features=shared['features'],prediction=prediction,ratio=ratio,
                    checkpoint_sha256=sha256(root/'gate.pt') if arm=='learned_gate' else None,
                    source_pixels_used=False,common_context_sha256=sha256(common)))
                paths=[*sorted((dest/'frames').glob('*.png')),dest/'reconstruction.mp4',dest/'generation_noise.json',
                    dest/'conditions.json',dest/'gate_decision.json',dest/'text_trace.json']
                write_json(dest/'result.json',dict(status='GENERATED',signature=signature,case=c['name'],arm=arm,
                    frames=len(pixels),fps=cfg.fps,artifacts={str(path.relative_to(dest)):sha256(path) for path in paths},
                    reused_common_first_segment=True))
                print(f'DONE {split}/{c["name"]}/{arm}: edit={ratio} predicted_utility={prediction}',flush=True)
    emit(total,total)


def train(root):
    import torch
    p=load(root);metrics=read_json(root/'train_metrics.json');records=[]
    for c in p['cases']:
        if c['split']!='train':continue
        before=metrics['cases'][c['name']]['baseline']['affected']
        after=metrics['cases'][c['name']]['constant_edit']['affected']
        records.append(dict(case=c['name'],source_id=c['source_id'],
            features=read_json(root/'train'/c['name']/'common.json')['features'],utility=gate.utility(before,after)))
    folds=[]
    groups=sorted({r['source_id'] for r in records})
    for j,held in enumerate(groups):
        fold,loss=gate.fit([r for r in records if r['source_id']!=held],p['training']['steps'],p['training']['seed'])
        predictions=[dict(case=r['case'],target=r['utility'],prediction=gate.predict(fold,r['features'])[0]) for r in records if r['source_id']==held]
        folds.append(dict(excluded_source=held,predictions=predictions,train_loss=loss))
        print(f'GROUP VALIDATION {j+1}/{len(groups)} ({100*(j+1)/len(groups):.1f}%): {held}',flush=True)
    def progress(n,total,loss):
        emit(n,total);print(f'TRAIN gate {n}/{total} ({100*n/total:.1f}%) loss={loss:.6f}',flush=True)
    checkpoint,loss=gate.fit(records,p['training']['steps'],p['training']['seed'],progress)
    checkpoint.update(protocol_signature=p['signature'],training_metrics_sha256=sha256(root/'train_metrics.json'))
    torch.save(checkpoint,root/'gate.pt')
    write_json(root/'training.json',dict(status='TRAINED',training_executed=True,parameters=checkpoint['parameter_count'],
        steps=checkpoint['steps'],loss=loss,examples=records,source_groups=groups,
        leave_one_source_out=folds,hyperparameters_selected_on_test=False,
        checkpoint_sha256=sha256(root/'gate.pt'),training_metrics_sha256=sha256(root/'train_metrics.json'),
        architecture='14 -> 16 SiLU -> 8 SiLU -> 1; source-balanced counterfactual utility regression',
        end_to_end_diffusion_training=False,generator_weights_frozen=True))


def worker(root,stage):
    if stage=='prepare':return prepare(root)
    if stage in ('generate_train','generate_test'):return generate(root,stage.split('_')[1])
    if stage=='train':return train(root)
    from .receiver_gate_evaluation import evaluate
    return evaluate(root,stage.split('_')[1])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--worker',choices=['prepare','generate_train','evaluate_train','train','generate_test','evaluate_test'])
    args=parser.parse_args();root=args.output.resolve()
    sys.path.insert(0,str(REPO/'.local/vendor/Open-Sora'))
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    if args.worker:return worker(root,args.worker)
    print('CHECK source identity split, received inputs and checkpoint hashes; ETA unknown',flush=True)
    if root==REPO or root in REPO.parents or any(x in root.parts for x in ('.local','data','assets')):
        raise ValueError('choose a separate experiment output')
    p=inventory(root)
    if args.check:
        print(json.dumps(dict(status='READY',training_windows=12,training_originals=3,test_windows=4,
            text_embeddings_to_prepare=4,model_training_started=False,signature=p['signature'])));return
    root.mkdir(parents=True,exist_ok=True)
    with lock(REPO/'.local/etri_60s_check.lock'),lock(root/'.execution.lock'):
        freeze(root/'plan.json',p)
        for name in p['code']:
            relative=Path(name).relative_to(REPO);dest=root/'code'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copyfile(name,dest)
        stages=Stages(root,p['signature']);env=dict(environment(2025),PYTHONPATH=str(REPO/'src'),PYTHONUNBUFFERED='1')
        jobs=[('prepare',['prepared.json','text_inputs']),('generate_train',['train']),
            ('evaluate_train',['train_metrics.json']),('train',['gate.pt','training.json']),
            ('generate_test',['test']),('evaluate_test',['test_metrics.json','results.json','report.html','comparisons'])]
        for j,(stage,products) in enumerate(jobs):
            print(f'전체 {j}/{len(jobs)} ({100*j/len(jobs):.1f}%, 단계 수 기준) | {stage}',flush=True)
            required=[*products,f'resources/{stage}.json']
            # Generators keep and validate completed individual arms on restart.
            owned=[f'resources/{stage}.json'] if stage.startswith('generate_') else required
            stages.step(stage,owned,required,
                lambda log,s=stage:launch([sys.executable,'-m',MODULE,'--output',str(root),'--worker',s],
                    log,env,root/f'resources/{s}.json'))
        print('전체 6/6 완료 (100%, 단계 수 기준)',flush=True)
        print(json.dumps(read_json(root/'results.json'),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
