"""Experimental unified BSU: paid predictive video followed by learned generation.

Run with an existing Qwen semantic encoding; sender and receiver workers have
different job schemas. The receiver is not passed any source/semantic-run path.
The adopted BSU default is deliberately a separate benchmark reference.
"""
import argparse
import csv
import gc
import itertools
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from . import unified_bsu_core as core
from .artifacts import sha256, write_json
from .qwen35_pairs_video import freeze
from .webvid5 import fingerprint
from .webvid_ablation import Stages
from .wire import pack, unpack

REPO, POLICY = core.REPO, core.POLICY
MODULE = 'semantic_transmission.unified_bsu'
CODE = ('unified_bsu', 'unified_bsu_core', 'unified_bsu_adapter', 'unified_bsu_training',
        'unified_bsu_tokens', 'unified_bsu_sender',
        'unified_bsu_training_policy', 'unified_bsu_training_quality', 'unified_bsu_evaluate', 'official_quality',
        'bsu_dcvc_keys', 'dense_temporal_receiver', 'dense_generation_runtime',
        'precision_text_cache', 'adapter_training_memory')


def event(stage, done=0, total=1, current='통합 BSU'):
    print('COMPARE_PROGRESS '+__import__('json').dumps(dict(stage=stage, current=current,
        overall=f'{done}/{total} ({100*done/total:.1f}%)', state='실행 중'), ensure_ascii=False), flush=True)


def paths():
    local = core.read(REPO / '.local/settings.json')
    preferred = REPO / '.local/unified_bsu_env/bin/python'
    python = str(preferred) if preferred.exists() else local['python']
    vendor = REPO / '.local/vendor/DCVC-RT'
    if not vendor.exists():
        vendor = REPO / '.local/vendor/DCVC/DCVC-family/DCVC-RT'
    return dict(python=python, channel=local['channel_python'], vendor=str(vendor),
                checkpoints=str(REPO / '.local/checkpoints/dcvc_rt'),
                models=core.read(REPO / '.local/model_paths.json'))


def recipe(models):
    from mmengine import Config
    cfg = Config.fromfile(REPO / 'configs/official_opensora.py')
    for name, key in [('stdit', 'model'), ('vae', 'vae'), ('t5', 'text_encoder')]:
        cfg[key]['from_pretrained'] = models[name]
    cfg.vae['vae_2d_path'] = models['vae2d']
    # Portable differentiable PyTorch layernorm, frozen original weights.
    cfg.model['enable_layernorm_kernel'] = False
    return cfg


def registry():
    sys.path.insert(0, str(REPO / '.local/vendor/Open-Sora'))
    import opensora.models
    from opensora.registry import MODELS, build_module
    return MODELS, build_module


def model_inventory(models):
    return {name: {str(p.relative_to(Path(models[name]))): sha256(p)
                   for p in sorted(Path(models[name]).rglob('*'))
                   if p.is_file() and p.suffix in ('.json', '.safetensors', '.bin', '.model')}
            for name in ('stdit', 'vae', 'vae2d', 't5')}


def load_rgb(directory, count=None):
    directory = Path(directory)
    if count is None:
        files = sorted(directory.glob('*.png'), key=lambda p: int(p.stem))
        if [int(p.stem) for p in files] != list(range(len(files))):
            raise ValueError('frame directory has a gap or duplicate index')
    else:
        files = [directory / f'{i}.png' for i in range(count)]
    return core.rgb(np.stack([np.asarray(Image.open(p).convert('RGB')) for p in files]))


def save_rgb(images, directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(images):
        Image.fromarray(frame).save(directory / f'{i:05d}.png')


def recover_base(codec, stream, coding, video):
    import cv2
    from .dense_temporal_receiver import interpolate
    indices, anchors = codec.decode(stream)
    if indices != coding['indices']:
        raise ValueError('decoded indices differ')
    if len(indices) == video['frames']:
        frames = anchors
    else:
        frames = interpolate(anchors, indices, video['frames'], coding['stride'],
            REPO / '.local/vendor/Practical-RIFE', REPO / '.local/checkpoints/rife/v4.25')
    if coding['scale'] == 2:
        frames = np.stack([cv2.resize(f, (video['width'], video['height']), interpolation=cv2.INTER_CUBIC)
                           for f in frames])
    return frames


def sender(j):
    import cv2
    from .bsu_dcvc_keys import Codec
    root = Path(j['output']); p = j['profile']; context = j['context']; v = context['video']
    for name, digest in j['source_hashes'].items():
        if sha256(Path(j['source']) / name) != digest:
            raise ValueError('sender source changed')
    images = load_rgb(j['source'], v['frames'])
    if images.shape[1:3] != (v['height'], v['width']):
        raise ValueError('source geometry differs from semantic context')
    codec = Codec(Path(j['runtime']['vendor']), Path(j['runtime']['checkpoints']))
    records = []; total = len(p['qps'])*len(p['strides'])*len(p['scales'])
    out = root / 'sender'; out.mkdir(parents=True, exist_ok=True)
    for i, (scale, stride, qp) in enumerate(itertools.product(p['scales'], p['strides'], p['qps'])):
        event('송신 압축 후보 검사', i, total)
        indices, motion = core.adaptive_indices(images, stride, context['indices'],
            p['motion_threshold'], p['cut_threshold'])
        selected = images[indices]
        if scale == 2:
            selected = np.stack([cv2.resize(x, (v['width']//2, v['height']//2), interpolation=cv2.INTER_AREA)
                                 for x in selected])
        coding = dict(qp=qp, scale=scale, stride=stride, indices=indices)
        stream = codec.encode(selected, indices, qp)
        packet = core.encode_packet(context, stream, coding, j['decoder'])
        name = f'scale{scale}_stride{stride}_qp{qp}'
        path = out / f'{name}.bin'; path.write_bytes(packet)
        uses = core.channel_uses(packet)
        row = dict(name=name, coding=coding, channel_uses=uses, packet_bytes=len(packet),
                   packet_sha256=sha256(path), fits=uses <= j['budget'], distortion=None)
        if row['fits']:
            base = recover_base(codec, stream, coding, v)
            row['distortion'] = core.proxy_distortion(images, base)
            if p.get('innovation') and j.get('checkpoint'):
                save_rgb(base, out / (name+'_base'))
        row['breakdown'] = core.packet_breakdown(packet)
        records.append(row)
        write_json(out / 'candidates.json', dict(rows=records, budget=j['budget'], complete=False))
    del codec; gc.collect()
    if p.get('innovation') and j.get('checkpoint'):
        import torch
        torch.cuda.empty_cache()
        from .unified_bsu_sender import enhance_candidates
        records = enhance_candidates(j, images, records)
    eligible = [r for r in records if r['fits']]
    if p.get('require_innovation') and j.get('checkpoint'):
        eligible = [r for r in eligible if 'innovation_step' in r]
    chosen = core.choose_candidate(eligible, j['budget'])
    write_json(out / 'candidates.json', dict(rows=records, budget=j['budget'], complete=True))
    (out / 'packet.bin').write_bytes((out / f'{chosen["name"]}.bin').read_bytes())
    write_json(out / 'selection.json', dict(selected=chosen, rows=records, budget=j['budget'],
        complete=True, actual_bytes_not_entropy_estimates=True, sender_local_decode=True,
        receiver_source_access=False, final_generated_quality_used=False))


def prompt_rows(context, p):
    rows = []
    for i, (start, end) in enumerate(core.windows(context['video']['frames'], p['window_frames'], p['window_overlap'])):
        active = [s for a, b, s in zip(context['indices'], context['indices'][1:], context['segments'])
                  if a < end and b >= start]
        text = ' '.join(dict.fromkeys(s['text'] for s in active))
        flow = sum(float(s['flow']) for s in active)/len(active)
        rows.append(dict(path=f'clips/sample/{i:05d}.mp4', text=text, flow=flow))
    return rows


def prepare_text(root, context, p, models):
    from . import precision_text_cache as cache
    root = Path(root); (root / 'receiver').mkdir(parents=True, exist_ok=True)
    with (root / 'receiver/metadata.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'text', 'flow']); writer.writeheader()
        writer.writerows(prompt_rows(context, p))
    cache.prepare(root, REPO, recipe(models).pretty_text, 'fp32')


def text_conditions(root):
    from . import precision_text_cache as cache
    report = core.read(Path(root) / cache.REPORT)
    entries = {e['prompt']: e for e in report['entries']}
    return [cache.load(entries[p], report['contract'], p) for p in report['prompts']]


def padded_tensor(frames, device='cuda'):
    import torch
    if not 1 <= len(frames) <= 17:
        raise ValueError('expected a <=17-frame window')
    padded = np.concatenate((frames, np.repeat(frames[-1:], 17-len(frames), axis=0)))
    return torch.from_numpy(padded.copy()).permute(3, 0, 1, 2)[None].to(device=device, dtype=torch.bfloat16)/127.5-1


def checkpoint_record(path):
    import torch
    path = Path(path).resolve(); c = torch.load(path, map_location='cpu', weights_only=True)
    from .unified_bsu_adapter import POLICY as adapter_policy
    if c.get('policy') != adapter_policy or c.get('steps', 0) < 1:
        raise ValueError('a trained unified BSU adapter is required')
    return dict(path=str(path), sha256=sha256(path), steps=c['steps'],
                training_signature=c['training_signature'], validated_quality=c.get('validated_quality', False),
                smoke_only=c.get('smoke_only', True))


def generate(j):
    import torch
    from .dense_generation_runtime import guide_latents
    from .unified_bsu_adapter import DenseConditionAdapter, sample
    from .adapter_training_memory import independent_chunks
    root = Path(j['output']); side, _ = core.decode_packet(Path(j['packet']).read_bytes())
    p = side['decoder']['profile']; v = side['context']['video']; coding = side['coding']
    if checkpoint_record(j['checkpoint'])['sha256'] != side['decoder']['adapter_sha256']:
        raise ValueError('receiver checkpoint differs from paid identity')
    c = torch.load(j['checkpoint'], map_location='cpu', weights_only=True)
    if c['backbone'] != j['models_inventory']:
        raise ValueError('adapter was trained with a different backbone')
    for key in ('adapter_width', 'window_frames', 'start_noise'):
        if p[key] != c['profile'][key]:
            raise ValueError('receiver/training profile differs: '+key)
    base = load_rgb(root / 'base/frames'); ws = core.windows(len(base), 17, p['window_overlap'])
    modes, build = registry(); cfg = recipe(j['runtime']['models'])
    # Encode/decode phases are separated from STDiT to fit 16GB hardware.
    latents = []
    vae = build(cfg.vae, modes).cuda().to(torch.bfloat16).eval()
    with torch.no_grad():
        for a, b in ws:
            latents.append(guide_latents(vae, padded_tensor(base[a:b])).cpu())
    del vae; gc.collect(); torch.cuda.empty_cache()
    innovation_rois = []
    if side.get('enhancements'):
        from .unified_bsu_tokens import InnovationCodec
        if not c.get('innovation_state') or c.get('innovation_config') != p.get('innovation'):
            raise ValueError('received tokens require matching shared innovation decoder')
        config = c['innovation_config']
        innovation = InnovationCodec(config['tokens'], config['width']).cuda().eval()
        innovation.load_state_dict(c['innovation_state'], strict=True)
        with torch.no_grad():
            for i, (prior, blob) in enumerate(zip(latents, side['enhancements'])):
                restored, roi = innovation.decompress(blob, prior.cuda().float())
                latents[i] = restored.to(torch.bfloat16).cpu(); innovation_rois.append(roi.cpu())
        del innovation; gc.collect(); torch.cuda.empty_cache()
    model = build(dict(cfg.model), modes, input_size=latents[0].shape[2:], in_channels=4,
                  caption_channels=4096, model_max_length=300, enable_sequence_parallelism=False)
    model = model.cuda().to(torch.bfloat16).eval().requires_grad_(False)
    adapter = DenseConditionAdapter(model.hidden_size, p['adapter_width']).cuda().eval()
    adapter.load_state_dict(c['state_dict'], strict=True); handle = adapter.attach(model)
    texts = text_conditions(root / 'text'); generated = []
    if len(texts) != len(ws):
        raise ValueError('window/text condition mismatch')
    try:
        with independent_chunks():
            for i, ((a, b), prior, text) in enumerate(zip(ws, latents, texts)):
                event('수신 Open-Sora 생성 보정', i, len(ws))
                null = model.y_embedder.y_embedding[None, None]
                z = sample(model, adapter, prior.cuda(), text, null, qp=coding['qp'],
                    height=v['height'], width=v['width'], fps=v['fps'], steps=p['diffusion_steps'],
                    start_noise=p['start_noise'], cfg_scale=p['cfg_scale'], seed=p['seed']+i)
                generated.append(z.cpu())
    finally:
        handle.remove()
    del model, adapter; gc.collect(); torch.cuda.empty_cache()
    vae = build(cfg.vae, modes).cuda().to(torch.bfloat16).eval()
    accum = np.zeros_like(base, dtype=np.float32); weight = np.zeros((len(base), 1, 1, 1), np.float32)
    with torch.no_grad():
        for (a, b), z in zip(ws, generated):
            decoded = vae.decode(z.cuda(), num_frames=17)[0, :, :b-a].float()
            array = ((decoded.permute(1, 2, 3, 0).clamp(-1, 1)+1)*127.5).cpu().numpy()
            taper = np.minimum(np.arange(1, b-a+1), np.arange(b-a, 0, -1)).astype(np.float32)[:, None, None, None]
            accum[a:b] += array*taper; weight[a:b] += taper
    if not np.all(weight > 0):
        raise ValueError('generation window coverage failure')
    raw = np.rint(accum/weight).astype(np.uint8)
    final, audit = core.bounded_refinement(base, raw, coding['qp'], p['max_rgb_correction'])
    from .bsu_dense_validation import save_movie
    save_movie(raw, root / 'generated', v['fps']); save_movie(final, root / 'refined', v['fps'])
    write_json(root / 'refined/audit.json', dict(audit, windows=len(ws), diffusion_steps=p['diffusion_steps'],
        checkpoint_sha256=sha256(j['checkpoint']), initialization='received_video_plus_partial_noise',
        one_step_distillation=False, quality_improvement_proven=False,
        paid_innovation_windows=len(innovation_rois), packet_breakdown=core.packet_breakdown(Path(j['packet']).read_bytes())))


def worker(stage, j):
    root = Path(j['output'])
    if stage == 'sender':
        return sender(j)
    if stage == 'channel':
        from .metadata_channel import transmit
        original = (root / 'sender/packet.bin').read_bytes()
        data, report = transmit(original, j['snr_db'], j['channel_seed'], bits_per_symbol=4)
        if data != original or report['bit_errors']:
            raise ValueError('channel failed integrity; no transmitter fallback')
        core.decode_packet(data)
        if report['complex_channel_uses'] != core.channel_uses(original):
            raise ValueError('actual channel use count differs')
        dest = root / 'received'; dest.mkdir(parents=True, exist_ok=True)
        (dest / 'packet.bin').write_bytes(data)
        write_json(dest / 'channel.json', report)
        return
    side, stream = core.decode_packet(Path(j['packet']).read_bytes())
    if stage == 'decode':
        from .bsu_dcvc_keys import Codec
        from .bsu_dense_validation import save_movie
        codec = Codec(Path(j['runtime']['vendor']), Path(j['runtime']['checkpoints']))
        base = recover_base(codec, stream, side['coding'], side['context']['video'])
        save_movie(base, root / 'base', side['context']['video']['fps'])
        write_json(root / 'base/audit.json', dict(packet_sha256=sha256(j['packet']),
            fresh_decoder=True, source_used=False, received_only_conditions=True))
    elif stage == 'text':
        prepare_text(root / 'text', side['context'], side['decoder']['profile'], j['runtime']['models'])
    elif stage == 'generate':
        generate(j)
    else:
        raise ValueError('unknown worker stage')


def execute(args):
    from .bsu_video import launch
    from .etri_60s_check import lock
    root, source, old = args.output.resolve(), args.source_frames.resolve(), args.semantic_run.resolve()
    if any(root == p or root in p.parents or p in root.parents for p in (source, old, REPO / '.local', REPO / 'data')):
        raise ValueError('output must be separate from inputs, models and data')
    p = core.profile(args.profile); runtime = paths()
    h, _ = unpack((old / 'transmitter/metadata.bin').read_bytes())
    context = dict(video=h['video'], indices=[k['index'] for k in h['keyframes']], segments=h['segments'], decoder={})
    core.validate_context(context)
    if args.budget_channel_uses < 1:
        raise ValueError('positive total channel budget required')
    checkpoint = checkpoint_record(args.adapter_checkpoint) if args.adapter_checkpoint else None
    if not args.base_only and not checkpoint:
        raise ValueError('full unified reconstruction requires --adapter-checkpoint')
    inventory = model_inventory(runtime['models'])
    decoder = dict(policy=POLICY, profile=p, adapter_sha256=checkpoint['sha256'] if checkpoint else None,
                   backbone_signature=fingerprint(inventory))
    source_hashes = {f'{i}.png': sha256(source / f'{i}.png') for i in range(context['video']['frames'])}
    code = {n: sha256(REPO / f'src/semantic_transmission/{n}.py') for n in CODE}
    request = dict(policy=POLICY, context=context, source=str(source), source_hashes=source_hashes,
        output=str(root), profile=p, runtime=runtime, decoder=decoder, models_inventory=inventory,
        budget=args.budget_channel_uses, checkpoint=checkpoint, code=code, base_only=args.base_only)
    request['signature'] = fingerprint(request)
    if args.check:
        print(__import__('json').dumps(dict(status='PREFLIGHT_ONLY', signature=request['signature'],
            frames=context['video']['frames'], budget=request['budget'], checkpoint=checkpoint), indent=2)); return
    root.mkdir(parents=True, exist_ok=True)
    with lock(REPO / '.local/etri_60s_check.lock'), lock(root / 'execution.lock'):
        freeze(root / 'protocol.json', request)
        # Only this job can see the source. The other workers receive no source path.
        freeze(root / 'sender_job.json', request)
        freeze(root / 'channel_job.json', dict(output=str(root), snr_db=p['snr_db'], channel_seed=p['channel_seed']))
        receiver = dict(output=str(root), packet=str(root / 'received/packet.bin'), runtime=runtime,
                        checkpoint=checkpoint['path'] if checkpoint else None, models_inventory=inventory)
        freeze(root / 'receiver_job.json', receiver)
        steps = [('sender', ['sender'], 'sender_job.json'),
                 ('channel', ['received'], 'channel_job.json'), ('decode', ['base'], 'receiver_job.json')]
        if not args.base_only:
            steps += [('text', ['text'], 'receiver_job.json'), ('generate', ['generated', 'refined'], 'receiver_job.json')]
        stages = Stages(root, request['signature'])
        for i, (stage, products, job) in enumerate(steps):
            event(stage, i, len(steps))
            python = runtime['channel'] if stage == 'channel' else runtime['python']
            env = {'CUDA_VISIBLE_DEVICES': '-1'} if stage == 'channel' else {}
            stages.step(stage, products, products, lambda log, s=stage, f=job, py=python, e=env:
                launch([py, '-m', MODULE, '--worker', s, '--job', root / f], log, e))
        selection = core.read(root / 'sender/selection.json')
        movie = root / ('base' if args.base_only else 'refined') / 'reconstruction.mp4'
        write_json(root / 'RESULT.json', dict(status='COMPLETE_UNIFIED_BASE' if args.base_only else 'COMPLETE_UNIFIED_RECONSTRUCTION',
            movie=str(movie), movie_sha256=sha256(movie), packet_sha256=sha256(root / 'received/packet.bin'),
            channel_uses=selection['selected']['channel_uses'], budget=request['budget'],
            budget_met=selection['selected']['channel_uses'] <= request['budget'],
            source_at_receiver=False, checkpoint=checkpoint, performance_improvement_proven=False,
            physical_radio_overhead_included=False, profile=p, request_signature=request['signature']))
        event('완료', len(steps), len(steps))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--semantic-run', type=Path); p.add_argument('--source-frames', type=Path)
    p.add_argument('--output', type=Path); p.add_argument('--budget-channel-uses', type=int)
    p.add_argument('--profile', type=Path); p.add_argument('--adapter-checkpoint', type=Path)
    p.add_argument('--base-only', action='store_true', help='prepare received training/base comparison inputs only')
    p.add_argument('--check', action='store_true'); p.add_argument('--worker'); p.add_argument('--job', type=Path)
    args = p.parse_args()
    if args.worker:
        return worker(args.worker, core.read(args.job))
    if any(getattr(args, n) is None for n in ('semantic_run', 'source_frames', 'output', 'budget_channel_uses')):
        p.error('--semantic-run, --source-frames, --output and --budget-channel-uses are required')
    execute(args)


if __name__ == '__main__':
    main()
