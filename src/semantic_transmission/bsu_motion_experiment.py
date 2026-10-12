"""Reproducible caption ablation and bounded learned-motion receiver pilot."""
import argparse
import copy
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .artifacts import sha256, write_json
from .webvid5 import read_json, fingerprint
from . import compact_motion as motion

REPO = Path(__file__).resolve().parents[2]
BASE = REPO/'outputs/bsu_zea_16s_20261009/reconstruction/event_pairs/run'
OLD = REPO/'outputs/lgvsc_qwen_zea_16s_20261008'
DEFAULT = REPO/'outputs/bsu_motion_pilot_20261010_v2'
ARMS = ['baseline', 'motion_zero', 'motion_learned']


def progress(stage, done, total, start):
    elapsed = time.monotonic()-start
    comparable = stage.startswith('CHANNEL') or stage == 'TRAIN adapter updates'
    eta = elapsed/done*(total-done) if done and comparable else None
    print(f'{stage} {done}/{total} ({100*done/total:.1f}%) elapsed={elapsed:.0f}s'
          + (f' remaining~{eta:.0f}s' if eta is not None else ' ETA unknown'), flush=True)


def source_window(path, first, last):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames = []
    for _ in range(last-first+1):
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f'incomplete source window: {path}')
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def load(root):
    p = read_json(root/'protocol.json')
    q = dict(p); signature = q.pop('signature')
    if fingerprint(q) != signature:
        raise ValueError('protocol signature changed')
    for path, digest in p['code'].items():
        if sha256(path) != digest:
            raise ValueError(f'experiment implementation changed; use a new output root: {path}')
    for c in p['cases']:
        for path, digest in c['inputs'].items():
            if sha256(path) != digest:
                raise ValueError(f'frozen experiment input changed: {path}')
    return p


def prepare(root):
    from .wire import pack, unpack
    from .packets import accounting
    from mmengine import Config
    if (root/'protocol.json').exists():
        load(root); print('RESUME protocol and inputs verified', flush=True); return
    root.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(str(BASE/'receiver/decoder_config.py'))
    old = read_json(REPO/'outputs/receiver_gate_20261009/prepared.json')['cases']
    cases = []
    for item in old:
        if item['name'] == 'zea':
            continue
        c = dict(name=item['name'], run=item['run'], source_video=item['source_video'],
            source_id=item['source_id'], source_sha256=item['source_sha256'],
            keys=[item['keys'][0], item['keys'][-1]], split=item['split'],
            text_report=item['text_report'], prompt=item['prompts'][0])
        if c['name'] == 'train_rallye':
            c['split'] = 'validation'
        cases.append(c)
    text = read_json(BASE/'receiver/text_embeddings.json')
    cases.insert(0, dict(name='zea', run=str(BASE), keys=[0,383], split='test', source_id='zea',
        source_video=str(BASE/'data/normalized.mp4'), source_sha256=sha256(BASE/'data/normalized.mp4'),
        text_report=str(BASE/'receiver/text_embeddings.json'), prompt=text['prompts'][0]))
    source_sets = {s:{c['source_sha256'] for c in cases if c['split']==s} for s in ('train','validation','test')}
    if any(source_sets[a] & source_sets[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('source leakage')
    start = time.monotonic()
    for number, c in enumerate(cases, 1):
        run = Path(c['run'])
        c['frames'] = c['keys'][1]-c['keys'][0]+1
        paths = [Path(c['source_video']), Path(c['text_report']),
                 *[run/f'receiver/frames/sample/key_frames_received/{k}.png' for k in c['keys']]]
        report = read_json(c['text_report'])
        entry = next(e for e in report['entries'] if e['prompt']==c['prompt'])
        paths.append(Path(entry['path']))
        c['inputs'] = {str(path):sha256(path) for path in paths}
        encoded, audit = motion.extract(source_window(c['source_video'], *c['keys']))
        directory = root/'sender'/c['name']; directory.mkdir(parents=True, exist_ok=True)
        packet = pack({'video':dict(frames=c['frames']),motion.FIELD:encoded})
        (directory/'motion.bin').write_bytes(packet)
        write_json(directory/'extraction.json', audit)
        c['motion_packet'] = str(directory/'motion.bin')
        c['inputs'][c['motion_packet']] = sha256(directory/'motion.bin')
        progress('PREPARE motion intervals', number, len(cases), start)
    # The decisive full-video comparison uses the actual existing NTSCC packet.
    original = (BASE/'transmitter/metadata.bin').read_bytes()
    header, payload = unpack(original)
    zea_motion = unpack(Path(cases[0]['motion_packet']).read_bytes())[0][motion.FIELD]
    capacity = accounting(len(original))['blocks'] * 6144 // 8
    full = motion.add_to_packet(original, zea_motion, capacity)
    directory = root/'sender/zea'
    (directory/'full_motion.bin').write_bytes(full)
    with (OLD/'lgvsc/run/receiver/metadata.csv').open() as stream:
        pllava = next(csv.DictReader(stream))['text']
    alternate = copy.deepcopy(header); alternate['segments'][0]['text'] = pllava
    caption_packet = pack(alternate, payload)
    (directory/'caption_only.bin').write_bytes(caption_packet)
    current_cfg = read_json(BASE/'run_config.json')
    p = dict(policy='bsu_compact_motion_learned_prior_pilot_v1', cases=cases,
        generator={k:cfg[k] for k in ('model','vae','image_size','fps','seed','scheduler','multi_resolution')},
        base_config=str(BASE/'receiver/decoder_config.py'), arms=ARMS, zea_extra_arm='pllava_caption',
        training=dict(seed=31415, steps=500, learning_rate=.001, source_balanced=True,
            train_sources=sorted({c['source_id'] for c in cases if c['split']=='train'}),
            validation_sources=sorted({c['source_id'] for c in cases if c['split']=='validation'}),
            target='frozen VAE original-video latents', objective='L1 + .2 temporal-difference L1',
            checkpoint_selection='lowest validation latent L1 at every 50 updates',
            receiver='two received endpoint latents plus decoded motion only',
            end_to_end_diffusion_training=False),
        guidance_strength=.15, caption_text=pllava,
        channel=dict(snr_db=current_cfg['snr_db'], seed=current_cfg.get('channel_seed',current_cfg['seed']),
            baseline_packet_bytes=len(original), motion_packet_bytes=len(full),
            packet_growth_bytes=len(full)-len(original), capacity_bytes=capacity,
            visual_complex_channel_uses=read_json(BASE/'channel_accounting.json')['visual_complex_channel_uses']),
        adoption_gate=dict(psnr_gain_min=.3, lpips_increase_max=.01, flow_ratio_max=1.05, mad_p95_ratio_max=1.10),
        scope='one full 16s harbor development video plus three source-disjoint 49f diagnostics; one seed/SNR',
        selection_policy_changed=False, output_pixel_warping=False,
        code={str(REPO/'src/semantic_transmission'/n):sha256(REPO/'src/semantic_transmission'/n) for n in
            ('compact_motion.py','motion_condition_adapter.py','bsu_motion_experiment.py','bsu_motion_evaluation.py')})
    for name in ('full_motion.bin','caption_only.bin'):
        cases[0]['inputs'][str(directory/name)] = sha256(directory/name)
    p['signature'] = fingerprint(p)
    write_json(root/'protocol.json', p)
    print(f'PACKET {len(original)} -> {len(full)} bytes / {capacity} byte capacity', flush=True)


def channel(root):
    from .metadata_channel import transmit
    from .wire import unpack
    p = load(root)
    jobs = [(c['name'],Path(c['motion_packet'])) for c in p['cases']]
    jobs.extend(('zea',root/'sender/zea'/name) for name in ('full_motion.bin','caption_only.bin'))
    start = time.monotonic(); reports = []
    for i,(name,path) in enumerate(jobs,1):
        out = root/'received'/name/path.name; out.parent.mkdir(parents=True,exist_ok=True)
        packet = path.read_bytes()
        received, report = transmit(packet,p['channel']['snr_db'],p['channel']['seed'])
        unpack(received)
        if packet != received:
            raise ValueError('digital channel changed packet')
        out.write_bytes(received)
        report.update(case=name, filename=path.name, exact_match=True, sha256=sha256(out))
        reports.append(report); progress('CHANNEL verified packets',i,len(jobs),start)
    write_json(root/'channel.json', dict(status='PASSED',signature=p['signature'],reports=reports))


def text_prepare(root):
    from .wire import unpack
    from .precision_text_cache import prepare as prepare_text
    p=load(root)
    work=root/'caption_input';(work/'receiver').mkdir(parents=True,exist_ok=True)
    header,_=unpack((root/'received/zea/caption_only.bin').read_bytes())
    with (work/'receiver/metadata.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['path','text','flow']);writer.writeheader();writer.writerows(header['segments'])
    print('TEXT preparing one received PLLaVA caption, FP32 T5; loading ETA unknown',flush=True)
    prepare_text(work,REPO,Path(p['base_config']).read_text(),'fp32')
    print('TEXT 1/1 (100%)',flush=True)


def setup_torch(p):
    import torch
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.manual_seed(p['generator']['seed']);torch.cuda.manual_seed_all(p['generator']['seed'])


def received_motion(root,c,zero=False):
    from .wire import unpack
    # Full Zea uses the accounted NTSCC+metadata packet, not an evaluation shortcut.
    name='full_motion.bin' if c['name']=='zea' else 'motion.bin'
    header,_=unpack((root/'received'/c['name']/name).read_bytes())
    return motion.decode(header[motion.FIELD],c['frames'])


def endpoint(vae,c,i):
    import torch
    from PIL import Image
    from opensora.datasets.utils import get_transforms_image
    path=Path(c['run'])/f'receiver/frames/sample/key_frames_received/{c["keys"][i]}.png'
    transform=get_transforms_image(image_size=(320,576),name='resize_crop')
    with Image.open(path) as im:
        return transform(im.convert('RGB')).unsqueeze(0).unsqueeze(2).to('cuda',torch.bfloat16)


def train(root):
    import torch
    import numpy as np
    from opensora.registry import MODELS,build_module
    from .motion_condition_adapter import MotionPrior,features
    from .receiver_joint_experiment import seed_for
    p=load(root);setup_torch(p)
    if (root/'training.json').exists():
        record=read_json(root/'training.json')
        if record['signature']!=p['signature'] or record['checkpoint_sha256']!=sha256(root/'adapter.pt'):
            raise ValueError('training receipt changed')
        print('RESUME trained adapter verified',flush=True);return
    selected=[c for c in p['cases'] if c['split']!='test']
    print('TRAIN loading frozen VAE; STDiT is not loaded for latent-prior supervision',flush=True)
    vae=build_module(p['generator']['vae'],MODELS).to('cuda',torch.bfloat16).eval().requires_grad_(False)
    dataset=[];start=time.monotonic()
    for i,c in enumerate(selected,1):
        cache=root/'training_data'/f'{c["name"]}.pt';cache.parent.mkdir(parents=True,exist_ok=True)
        if cache.exists():
            sample=torch.load(cache,map_location='cpu',weights_only=True)
            if sample['signature']!=p['signature']:raise ValueError('stale training cache')
        else:
            with torch.inference_mode():
                refs=[]
                for j in range(2):
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                        torch.manual_seed(seed_for('vae',j,p['generator']['seed']))
                        refs.append(vae.encode(endpoint(vae,c,j))[:,:,0].float())
                value=features(*refs,received_motion(root,c),c['frames'])
                source=source_window(c['source_video'],*c['keys'])
                tensor=torch.from_numpy(source.copy()).permute(3,0,1,2).unsqueeze(0).to('cuda',torch.bfloat16)/127.5-1
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(p['training']['seed'])
                    target=vae.encode(tensor).float()
                if value.shape[2:]!=target.shape[2:]:raise ValueError('latent timeline mismatch')
                sample=dict(signature=p['signature'],features=value.cpu(),target=target.cpu(),name=c['name'],
                    split=c['split'],source_id=c['source_id'])
                torch.save(sample,cache)
        dataset.append(sample);progress('TRAIN encode source-disjoint windows',i,len(selected),start)
    del vae;torch.cuda.empty_cache()
    torch.manual_seed(p['training']['seed'])
    adapter=MotionPrior().cuda();optimizer=torch.optim.AdamW(adapter.parameters(),lr=p['training']['learning_rate'])
    groups={name:[s for s in dataset if s['split']=='train' and s['source_id']==name] for name in p['training']['train_sources']}
    validation=[s for s in dataset if s['split']=='validation']
    rng=np.random.default_rng(p['training']['seed']);history=[];best=float('inf');start=time.monotonic()
    with torch.no_grad():
        untrained=float(np.mean([torch.nn.functional.l1_loss(adapter(s['features'].cuda()),s['target'].cuda()).item() for s in validation]))
    for step in range(1,p['training']['steps']+1):
        group=groups[p['training']['train_sources'][(step-1)%len(groups)]]
        s=group[int(rng.integers(len(group)))];x=s['features'].cuda();target=s['target'].cuda()
        prediction=adapter(x)
        loss=torch.nn.functional.l1_loss(prediction,target)+.2*torch.nn.functional.l1_loss(prediction[:,:,1:]-prediction[:,:,:-1],target[:,:,1:]-target[:,:,:-1])
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.0);optimizer.step()
        if step%50==0:
            with torch.no_grad():
                val=float(np.mean([torch.nn.functional.l1_loss(adapter(s['features'].cuda()),s['target'].cuda()).item() for s in validation]))
            history.append(dict(step=step,train_loss=float(loss),validation_l1=val))
            if val<best:
                best=val
                torch.save(dict(state_dict={k:v.detach().cpu() for k,v in adapter.state_dict().items()},signature=p['signature'],
                    selected_step=step,training_source_ids=p['training']['train_sources']),root/'adapter.pt')
            progress('TRAIN adapter updates',step,p['training']['steps'],start)
    write_json(root/'training.json',dict(status='TRAINED',signature=p['signature'],checkpoint_sha256=sha256(root/'adapter.pt'),
        parameters=sum(v.numel() for v in adapter.parameters()),history=history,untrained_validation_l1=untrained,
        selected_validation_l1=best,training_data=[dict(name=s['name'],split=s['split'],source_id=s['source_id']) for s in dataset],
        test_sources_used_for_training_or_selection=False,end_to_end_diffusion_training=False))


def generate(root,caption_only=False):
    import torch
    import numpy as np
    from mmengine import Config
    from opensora.registry import MODELS,SCHEDULERS,build_module
    from opensora.utils import inference_utils as inference
    from .generation_noise import NoiseControl,tensor_hash
    from .precision_text_cache import Encoder
    from .receiver_alignment_experiment import complete_arm
    from .condition_collision import apply_guard
    from .receiver_joint_experiment import save_pixels
    from .motion_condition_adapter import MotionPrior,features,guidance_hook
    p=load(root);setup_torch(p);cfg=Config(p['generator']);signature=p['signature']
    cases=[c for c in p['cases'] if c['split']=='test' and (not caption_only or c['name']=='zea')]
    jobs=[(c,a) for c in cases for a in (['baseline','pllava_caption'] if caption_only else ARMS)]
    if all(complete_arm(root/'generation'/c['name']/a,signature) for c,a in jobs):
        print('RESUME generation receipts verified',flush=True);return
    print('GENERATE loading frozen VAE and STDiT; model loading ETA unknown',flush=True)
    vae=build_module(cfg.vae,MODELS).to('cuda',torch.bfloat16).eval().requires_grad_(False)
    model=build_module(cfg.model,MODELS,input_size=vae.get_latent_size((384,*cfg.image_size)),in_channels=vae.out_channels,
        caption_channels=4096,model_max_length=300,enable_sequence_parallelism=False).to('cuda',torch.bfloat16).eval().requires_grad_(False)
    runtime=dict(packages={n:importlib.metadata.version(n) for n in ('torch','diffusers','transformers')},cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
    adapter=None
    if not caption_only:
        receipt=read_json(root/'training.json');checkpoint=torch.load(root/'adapter.pt',map_location='cpu',weights_only=True)
        if checkpoint['signature']!=signature or receipt['checkpoint_sha256']!=sha256(root/'adapter.pt'):raise ValueError('adapter changed')
        adapter=MotionPrior().cuda().eval();adapter.load_state_dict(checkpoint['state_dict'])
    finished=sum(complete_arm(root/'generation'/c['name']/a,signature) for c,a in jobs);start=time.monotonic()
    with torch.inference_mode():
        for c,arm in jobs:
            directory=root/'generation'/c['name']/arm
            if complete_arm(directory,signature):continue
            directory.mkdir(parents=True,exist_ok=True)
            # Zea replays every actual VAE/initial/RFLOW draw from the completed BSU.
            expected=read_json(BASE/'receiver/generation_noise.json') if c['name']=='zea' else None
            contract=expected['contract'] if expected else dict(seed=cfg.seed,segments=1,steps=cfg.scheduler.num_sampling_steps,inputs=fingerprint(c),code=signature)
            baseline=root/'generation'/c['name']/'baseline/generation_noise.json'
            if expected is None and arm!='baseline':expected=read_json(baseline)
            control=NoiseControl(directory/'generation_noise.json',contract,runtime,expected)
            refs=[[]]
            for j in range(2):
                x=endpoint(vae,c,j)
                with control.scope('vae',j,x):refs[0].append(vae.encode(x).squeeze(0))
            latent_size=vae.get_latent_size((c['frames'],*cfg.image_size));model.config.input_size=latent_size
            z=torch.empty(1,vae.out_channels,*latent_size,device='cuda',dtype=torch.bfloat16)
            with control.scope('initial',0,z):z.copy_(torch.randn_like(z))
            mask,conditions=apply_guard(z,refs,['0;0,1,0,-1,1'],0,5,2,inference)
            initial=z.clone()
            report=copy.deepcopy(read_json(root/'caption_input/receiver/text_embeddings.json' if arm=='pllava_caption' else c['text_report']))
            prompt=report['prompts'][0] if arm=='pllava_caption' else c['prompt']
            report['prompts']=[prompt];report['prompt_count']=1;report['entries']=[e for e in report['entries'] if e['prompt']==prompt]
            encoder=Encoder(report,directory/'text_embedding_trace.json');encode=encoder.encode
            encoder.encode=lambda texts:{k:v.to('cuda',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in encode(texts).items()}
            encoder.y_embedder=model.y_embedder
            hooks=[];prior=None
            if arm.startswith('motion_'):
                inputs=features(refs[0][0][:,0].unsqueeze(0),refs[0][1][:,0].unsqueeze(0),received_motion(root,c),c['frames'],zero_motion=arm=='motion_zero')
                prior=adapter(inputs)
                if list(prior.shape)!=list(z.shape):raise ValueError('prior timeline differs')
                hooks.append(model.register_forward_hook(guidance_hook(prior,p['guidance_strength'])))
            step=[0]
            def tick(module,args):
                step[0]+=1
                if step[0]==1 or step[0]%5==0:
                    progress(f'GENERATE {c["name"]}/{arm} | model calls',finished*30+step[0],len(jobs)*30,start)
            hooks.append(model.register_forward_pre_hook(tick))
            extra=inference.prepare_multi_resolution_info(cfg.multi_resolution,1,tuple(cfg.image_size),c['frames'],cfg.fps,'cuda',torch.bfloat16)
            scheduler=build_module(cfg.scheduler,SCHEDULERS)
            try:
                with control.scope('sampling',0,z):
                    sampled=scheduler.sample(model,encoder,z=z,prompts=[prompt],device='cuda',additional_args=extra,progress=False,mask=mask)
            finally:
                for hook in hooks:hook.remove()
            control.finish(1,cfg.scheduler.num_sampling_steps)
            if not torch.equal(initial[:,:,mask[0]==0],sampled[:,:,mask[0]==0]):raise ValueError('hard key conditions changed')
            if not torch.isfinite(sampled).all():raise ValueError('nonfinite reconstruction')
            decoded=vae.decode(sampled.to(torch.bfloat16),num_frames=c['frames'])
            pixels=(decoded[0].clamp(-1,1).add(1).div(2).mul(255).add(.5).clamp(0,255).permute(1,2,3,0).to('cpu',torch.uint8).numpy())
            save_pixels(directory,pixels,cfg.fps)
            write_json(directory/'conditions.json',dict(conditions=conditions,prompt=prompt,hard_keys_preserved=True,
                prior_sha256=tensor_hash(prior) if prior is not None else None,source_pixels_read=False,
                receiver_inputs='existing channel-decoded key PNGs, cached received caption, received motion packet',
                noise_contract_reference='historical BSU noise identity; actual caption/motion differences recorded separately'))
            paths=[*sorted((directory/'frames').glob('*.png')),directory/'reconstruction.mp4',directory/'generation_noise.json',directory/'conditions.json']
            write_json(directory/'result.json',dict(status='GENERATED',signature=signature,case=c['name'],arm=arm,frames=c['frames'],fps=cfg.fps,
                artifacts={str(path.relative_to(directory)):sha256(path) for path in paths}))
            finished+=1;progress('GENERATE completed arms',finished,len(jobs),start)
            del refs,z,initial,sampled,decoded,pixels,encoder,prior


def evaluate(root):
    from .bsu_motion_evaluation import evaluate as evaluate_impl
    evaluate_impl(root,load(root))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--stage',choices=['all','prepare','channel','text','caption','train','generate','evaluate'],default='all')
    args=parser.parse_args();root=args.root.resolve()
    if args.stage=='all':
        stages=['prepare','channel','text','caption','train','generate','evaluate'];start=time.monotonic()
        root.mkdir(parents=True,exist_ok=True)
        import fcntl
        with (root/'execution.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for i,stage in enumerate(stages):
                progress('OVERALL stages',i,len(stages),start)
                python='/home/sangukbae/anaconda3/envs/lgvsc-channel/bin/python' if stage=='channel' else sys.executable
                env=os.environ.copy()
                if stage=='channel':env['CUDA_VISIBLE_DEVICES']='-1'
                write_json(root/'status.json',dict(status='RUNNING',stage=stage,completed=i,total=len(stages)))
                result=subprocess.run([python,'-m','semantic_transmission.bsu_motion_experiment','--root',str(root),'--stage',stage],env=env)
                if result.returncode:
                    write_json(root/'status.json',dict(status='FAILED',stage=stage,returncode=result.returncode))
                    raise SystemExit(result.returncode)
            progress('OVERALL stages',len(stages),len(stages),start)
            write_json(root/'status.json',dict(status='COMPLETE_PILOT',seconds=time.monotonic()-start))
    else:
        sys.path.insert(0,str(REPO/'.local/vendor/Open-Sora'))
        if args.stage=='caption':generate(root,caption_only=True)
        elif args.stage=='text':text_prepare(root)
        else:globals()[args.stage](root)


if __name__=='__main__':
    main()
