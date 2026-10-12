"""Run the receiver overlap experiment without training or pixel interpolation."""
import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .artifacts import sha256, write_json
from .webvid5 import fingerprint, read_json
from .receiver_alignment_experiment import cases, prepare_case, make_protocol, complete_arm
from .receiver_temporal_alignment import capture_single_image
from .receiver_joint_windows import POLICY, grid_plan, key_condition, sample_windows

ARMS = ['legacy_aligned', 'shared_sequential', 'shared_joint']
OWN_CODE = ['receiver_joint_windows.py', 'receiver_joint_experiment.py', 'receiver_joint_evaluation.py',
            'receiver_alignment_evaluation.py', 'official_quality.py', 'research_quality.py']


def prepare(repo):
    selected = [prepare_case(c) for c in cases(repo)]
    cfg, protocol = make_protocol(repo, selected)
    old = repo/'outputs/receiver_temporal_alignment_20261007/paired_v1'
    old_protocol = read_json(old/'protocol.json')
    old_signature = fingerprint(old_protocol)
    baselines = {}
    plans = {}
    for case, previous in zip(selected, old_protocol['cases']):
        if fingerprint(case) != fingerprint(previous):
            raise ValueError('legacy pilot inputs differ from this comparison')
        directory = old/case['name']/'legacy_aligned'
        if not complete_arm(directory, old_signature):
            raise ValueError('missing completed legacy baseline')
        baselines[case['name']] = dict(directory=str(directory), receipt_sha256=sha256(directory/'result.json'))
        relative = [k-case['keys'][0] for k in case['keys']]
        plans[case['name']] = grid_plan(relative[-1]+1, relative)
        if len(plans[case['name']]['windows']) != len(case['prompts']):
            raise ValueError('pilot requires one frozen prompt per window')
    protocol.update(policy=POLICY, arms=ARMS, baselines=baselines, plans=plans,
        shared_grid_paired_arms=['shared_sequential', 'shared_joint'],
        legacy_comparison='descriptive; legacy uses a different latent grid and cannot have identical noise tensors',
        schedule='one global-frame RFLOW schedule in both shared-grid arms',
        prompt_assignment='original first/second captions to overlapping first/second windows; identical across paired arms',
        scope='four 49-frame windows, one seed, equal 60 model evaluations per new arm',
        full_freenoise_reproduction=False, sampling_noise='binary masks; native RFLOW extra noise has zero contribution')
    protocol['code'].update({str(repo/'src/semantic_transmission'/name):sha256(repo/'src/semantic_transmission'/name) for name in OWN_CODE})
    return cfg, protocol


def seed_for(kind, index, seed):
    return int(hashlib.sha256(f'{seed}:{kind}:{index}'.encode()).hexdigest()[:15], 16)


def save_pixels(directory, pixels, fps):
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=True)
    frames = directory/'frames'
    frames.mkdir(exist_ok=True)
    if any(int(p.stem) >= len(pixels) for p in frames.glob('*.png')):
        raise ValueError('stale frames in incomplete output')
    for i, frame in enumerate(pixels):
        Image.fromarray(frame).save(frames/f'{i:05d}.png')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', str(fps), '-i', str(frames/'%05d.png'),
        '-c:v', 'libx264', '-threads', '2', '-crf', '18', '-pix_fmt', 'yuv420p', str(directory/'reconstruction.mp4')], check=True)


def receipt(directory, signature, case, arm, extra=None):
    paths = [*sorted((directory/'frames').glob('*.png')), directory/'reconstruction.mp4']
    paths += [p for p in (directory/'sampling_trace.json', directory/'legacy_provenance.json') if p.exists()]
    write_json(directory/'result.json', dict(status='GENERATED', signature=signature, case=case, arm=arm,
        frames=49, fps=24, artifacts={str(p.relative_to(directory)):sha256(p) for p in paths}, **(extra or {})))


def generate(repo, output, cfg, protocol, signature):
    import torch
    from PIL import Image
    from opensora.registry import MODELS, build_module
    from opensora.datasets.utils import get_transforms_image
    from opensora.utils.inference_utils import prepare_multi_resolution_info
    from opensora.schedulers.rf import timestep_transform
    from .generation_noise import tensor_hash
    from .precision_text_cache import Encoder

    for case in protocol['cases']:
        directory = output/case['name']/'legacy_aligned'
        if complete_arm(directory, signature):
            continue
        original = Path(protocol['baselines'][case['name']]['directory'])
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copytree(original/'frames', directory/'frames', dirs_exist_ok=True)
        shutil.copyfile(original/'reconstruction.mp4', directory/'reconstruction.mp4')
        write_json(directory/'legacy_provenance.json', protocol['baselines'][case['name']])
        receipt(directory, signature, case['name'], 'legacy_aligned', dict(reused_frozen_result=True))
    total = 2*len(protocol['cases'])
    finished = sum(complete_arm(output/c['name']/a, signature) for c in protocol['cases'] for a in ARMS[1:])
    if finished == total:
        print('RESUME generation 8/8 (100%): all receipts verified', flush=True)
        return
    print(f'LOAD frozen VAE and STDiT; {finished}/{total} completed; loading ETA unknown', flush=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    device, dtype = 'cuda', torch.bfloat16
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    model = build_module(cfg.model, MODELS, input_size=(10,40,72), in_channels=vae.out_channels,
        caption_channels=4096, model_max_length=300, enable_sequence_parallelism=False).to(device,dtype).eval()
    transform = get_transforms_image(image_size=tuple(cfg.image_size), name='resize_crop')
    runtime = dict(packages={n:importlib.metadata.version(n) for n in ('torch','diffusers','transformers')},
                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
    start, newly_done = time.monotonic(), 0
    with torch.inference_mode():
        for case in protocol['cases']:
            case_root = output/case['name']
            if all(complete_arm(case_root/a, signature) for a in ARMS[1:]):
                continue
            plan = protocol['plans'][case['name']]
            windows = [(w['start'],w['stop']) for w in plan['windows']]
            key_conditions, key_hashes = [], []
            for i, (key, anchor) in enumerate(zip(case['keys'], plan['anchors'])):
                path = Path(case['run'])/f'receiver/frames/sample/key_frames_received/{key}.png'
                with Image.open(path) as im:
                    value = transform(im.convert('RGB')).unsqueeze(0).unsqueeze(2).to(device,dtype)
                seed = seed_for('vae', i, cfg.seed)
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.random.default_generator.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    with capture_single_image(vae) as capture:
                        vae.encode(value)
                condition = key_condition(vae, capture, anchor)
                key_conditions.append(condition)
                key_hashes.append(dict(seed=seed, condition=tensor_hash(condition), spatial=tensor_hash(capture['spatial']),
                                       epsilon=tensor_hash(capture['epsilon'])))
            generator = torch.Generator(device=device).manual_seed(seed_for('shared_initial', 0, cfg.seed))
            initial = torch.randn(1, vae.out_channels, plan['latent_frames'], 40, 72,
                                  generator=generator, device=device, dtype=dtype)
            noise_hash = tensor_hash(initial)
            fixed = torch.zeros((1,plan['latent_frames']), device=device, dtype=torch.bool)
            for anchor, value in zip(plan['anchors'],key_conditions):
                index = anchor['latent'];initial[:,:,index:index+1] = value;fixed[:,index] = True
            global_args = prepare_multi_resolution_info(cfg.multi_resolution,1,tuple(cfg.image_size),plan['frames'],cfg.fps,device,dtype)
            steps = cfg.scheduler.num_sampling_steps
            times = [timestep_transform(torch.tensor([(1-i/steps)*1000],device=device),global_args,num_timesteps=1000)
                     for i in range(steps)]
            report = copy.deepcopy(read_json(Path(case['run'])/'receiver/text_embeddings.json'))
            report['prompts'],report['prompt_count'] = case['prompts'],len(case['prompts'])
            report['entries'] = [e for e in report['entries'] if e['prompt'] in case['prompts']]
            encoder = Encoder(report,case_root/'text_embedding_trace.json');encoder.y_embedder = model.y_embedder
            conditions = []
            for prompt, window in zip(case['prompts'],plan['windows']):
                condition = {k:v.to(device=device,dtype=dtype if v.is_floating_point() else v.dtype)
                             for k,v in encoder.encode([prompt]).items()}
                condition['y'] = torch.cat([condition['y'],encoder.null(1)],0)
                condition.update(prepare_multi_resolution_info(cfg.multi_resolution,1,tuple(cfg.image_size),window['frames'],cfg.fps,device,dtype))
                conditions.append(condition)
            shared = dict(signature=signature, initial_noise_sha256=noise_hash, conditioned_initial_sha256=tensor_hash(initial),
                key_samples=key_hashes, fixed_indices=[a['latent'] for a in plan['anchors']],
                timesteps=[float(t[0]) for t in times], plan=plan, runtime=runtime,
                text_y_sha256=[tensor_hash(c['y']) for c in conditions])
            existing = case_root/'paired_inputs.json'
            if existing.exists() and fingerprint(read_json(existing)) != fingerprint(shared):
                raise ValueError('shared conditioning/runtime changed on resume')
            write_json(existing,shared)
            for arm, mode in [('shared_sequential','sequential'),('shared_joint','joint')]:
                directory = case_root/arm
                if complete_arm(directory,signature):
                    continue
                directory.mkdir(parents=True,exist_ok=True)
                write_json(directory/'status.json',dict(status='GENERATING',signature=signature))
                trace = []
                def predict(index,z,t,active):
                    model.config.input_size = tuple(z.shape[2:])
                    extra = dict(conditions[index],x_mask=active.repeat(2,1))
                    prediction = model(torch.cat([z,z],0),t.repeat(2),**extra).chunk(2,dim=1)[0]
                    conditional,unconditional = prediction.chunk(2,dim=0)
                    trace.append(dict(window=index,t=float(t[0]),active=active[0].cpu().tolist()))
                    return unconditional+cfg.scheduler.cfg_scale*(conditional-unconditional)
                def progress(done,count):
                    if done == 1 or done%5 == 0 or done == count:
                        overall = finished*count+done
                        print(f'GENERATE {overall}/{total*count} ({100*overall/(total*count):.1f}%) | {case["name"]}/{arm} '
                              f'{done}/{count} model calls | elapsed={time.monotonic()-start:.0f}s',flush=True)
                rng_cpu,rng_cuda = torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
                sampled = sample_windows(initial,fixed,windows,times,predict,mode,progress)
                unchanged = torch.equal(rng_cpu,torch.get_rng_state()) and torch.equal(rng_cuda,torch.cuda.get_rng_state())
                if not unchanged:
                    raise ValueError('unexpected random draws during supposedly deterministic sampling')
                decoded = vae.decode(sampled.to(dtype),num_frames=plan['frames'])
                pixels = (decoded[0].clamp(-1,1).add(1).div(2).mul(255).add(.5).clamp(0,255)
                          .permute(1,2,3,0).to('cpu',torch.uint8).numpy())
                if pixels.shape != (49,320,576,3):
                    raise ValueError('unexpected decoded time/space shape')
                save_pixels(directory,pixels,cfg.fps)
                write_json(directory/'sampling_trace.json',dict(policy=POLICY,mode=mode,
                    paired_input_sha256=sha256(existing),rng_unchanged=True,calls=trace,
                    latent_sha256=tensor_hash(sampled),received_anchor_preserved=True))
                receipt(directory,signature,case['name'],arm)
                write_json(directory/'status.json',dict(status='GENERATED',signature=signature))
                finished += 1;newly_done += 1
                eta = (time.monotonic()-start)/newly_done*(total-finished)
                print(f'DONE {finished}/{total} arms ({100*finished/total:.1f}%) elapsed={time.monotonic()-start:.0f}s remaining~{eta:.0f}s',flush=True)
    write_json(output/'generation_summary.json',dict(status='GENERATED',seconds_excluding_load=time.monotonic()-start,
        new_arms=total,reused_legacy_arms=len(protocol['cases']),runtime=runtime))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--stage',choices=['all','generate','evaluate'],default='all')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(repo/'.local/vendor/Open-Sora'))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    print('CHECK received data, frozen baseline receipts and checkpoints; ETA unknown',flush=True)
    cfg,protocol = prepare(repo);signature = fingerprint(protocol)
    output = (args.output or repo/'outputs/receiver_joint_windows_20261008'/f'pilot_{signature[:12]}').resolve()
    if args.check:
        print(f'CHECK PASSED: 4 scenes, 2 paired arms + frozen legacy; output={output}',flush=True)
        return
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol_path = output/'protocol.json'
        if protocol_path.exists() and fingerprint(read_json(protocol_path)) != signature:
            raise ValueError('output protocol changed; choose another output')
        write_json(protocol_path,protocol)
        archive = output/'code';archive.mkdir(exist_ok=True)
        for name in protocol['code']:
            target = archive/Path(name).relative_to(repo)
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes())
        try:
            if args.stage in {'all','generate'}:
                write_json(output/'status.json',dict(status='GENERATING',signature=signature))
                generate(repo,output,cfg,protocol,signature)
            if args.stage in {'all','evaluate'}:
                if not all(complete_arm(output/c['name']/a,signature) for c in protocol['cases'] for a in ARMS):
                    raise ValueError('evaluation needs all completed arms')
                write_json(output/'status.json',dict(status='EVALUATING',signature=signature))
                from .receiver_joint_evaluation import evaluate
                evaluate(output,protocol,signature)
            else:
                write_json(output/'status.json',dict(status='GENERATED_EVALUATION_PENDING',signature=signature))
        except BaseException as error:
            write_json(output/'status.json',dict(status='FAILED',signature=signature,error=str(error)))
            raise


if __name__ == '__main__':
    main()
