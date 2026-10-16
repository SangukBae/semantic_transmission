"""Source-disjoint, resumable training of the unified BSU dense condition adapter.

Preparation uses real DCVC packets and a fresh received-only decoder. Training
freezes STDiT/VAE/T5/DCVC; optional innovation encoder/decoder and the receiver
adapter learn jointly. Target pixels are available ONLY to sender/trainer. An
export records that reconstruction quality is unvalidated until independently
measured. A smoke export can never masquerade as completed main training.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

from . import unified_bsu as u
from . import unified_bsu_core as core
from .artifacts import sha256, write_json
from .qwen35_pairs_video import freeze
from .webvid5 import fingerprint
from .wire import pack, unpack
from .adapter_training_state import atomic_torch_save, commit_checkpoint, load_checkpoint

MODULE = 'semantic_transmission.unified_bsu_training'
DEFAULT_SELECTION = core.REPO / 'outputs/bsu_adapter_training_7h_v1/protocol.json'


def data_plan(args):
    if getattr(args, 'training_config', None):
        from . import unified_bsu_training_policy as policy
        c = policy.config(args.training_config); profile = core.profile(args.profile)
        examples, quality, summary, inputs, excluded = policy.expanded_plan(args.selection, c, profile)
        if c['max_steps'] <= args.codec_warmup_steps+c['min_joint_steps']:
            raise ValueError('maximum steps must include the minimum joint training period')
        runtime = u.paths()
        value = dict(policy=core.POLICY, examples=examples, quality_examples=quality,
            sources=summary['sources'], data_summary=summary, inputs=inputs, excluded_sources=excluded,
            runtime=runtime, selection=str(args.selection.resolve()), selection_sha256=sha256(args.selection),
            profile=profile, budget_per_frame=args.budget_per_frame,
            training_config=c, training_config_sha256=sha256(args.training_config),
            backbone=u.model_inventory(runtime['models']),
            code={name: sha256(core.REPO / f'src/semantic_transmission/{name}.py') for name in u.CODE},
            train=dict(learning_rate=c['learning_rate'], weight_decay=.01, seed=2025,
                max_steps=c['max_steps'], innovation_warmup_steps=args.codec_warmup_steps,
                validate_every=c['validate_every'], save_every=c['save_every'], condition_dropout=.1),
            smoke_only=args.smoke, source_disjoint=True, test_sources_used=False)
        if args.initialize:
            value['initialize'] = dict(path=str(args.initialize.resolve()), sha256=sha256(args.initialize))
        value['signature'] = fingerprint(value)
        return value
    p = core.read(args.selection); rows = []; counts = {'train': 0, 'validation': 0}
    grouped = {}
    for e in p['examples']:
        if e['role'] in counts and e['end']-e['start']+1 >= 17:
            grouped.setdefault((e['role'], e['source_sha256']), []).append(e)
    for (role, source_hash), examples in sorted(grouped.items()):
        limit = args.limit_train if role == 'train' else args.limit_validation
        if limit and counts[role] >= limit:
            continue
        counts[role] += 1
        e = max(examples, key=lambda x: (x['end']-x['start'], -x['start']))
        h, _ = unpack(Path(e['packet']).read_bytes())
        segment = h['segments'][e['segment']]
        starts = sorted(set(np.linspace(e['start'], e['end']-16, args.windows_per_video).round().astype(int).tolist()))
        for n, start in enumerate(starts):
            paths = [Path(e['source_directory']) / f'{i}.png' for i in range(start, start+17)]
            rows.append(dict(name=f'{e["id"]}_{start:05d}', role=role, source_id=e['id'], source_sha256=source_hash,
                source_frames=[str(x) for x in paths], source_hashes=[sha256(x) for x in paths],
                semantic_packet=str(Path(e['packet']).resolve()), semantic_packet_sha256=sha256(e['packet']),
                context=dict(video=dict(h['video'], frames=17), indices=[0, 16], decoder={},
                             segments=[dict(path='clips/sample/0_16.mp4', text=segment['text'], flow=segment['flow'])])))
    if not counts['train'] or not counts['validation']:
        raise ValueError('source-disjoint training and validation sources required')
    train = {r['source_sha256'] for r in rows if r['role'] == 'train'}
    validation = {r['source_sha256'] for r in rows if r['role'] == 'validation'}
    if train & validation:
        raise ValueError('training/validation source leakage')
    # Test-role sources from the existing frozen split are not sampled here.
    profile = core.profile(args.profile); runtime = u.paths()
    value = dict(policy=core.POLICY, examples=rows, sources=counts, runtime=runtime,
        selection=str(args.selection.resolve()), selection_sha256=sha256(args.selection), profile=profile,
        budget_per_frame=args.budget_per_frame, backbone=u.model_inventory(runtime['models']),
        code={name: sha256(core.REPO / f'src/semantic_transmission/{name}.py') for name in u.CODE},
        train=dict(learning_rate=.0001, weight_decay=.01, seed=2025, max_steps=10000,
                   innovation_warmup_steps=args.codec_warmup_steps if profile.get('innovation') else 0,
                   validate_every=50, save_every=10, condition_dropout=.1),
        smoke_only=args.smoke, source_disjoint=True, test_sources_used=False)
    if args.initialize:
        value['initialize'] = dict(path=str(args.initialize.resolve()), sha256=sha256(args.initialize))
    value['signature'] = fingerprint(value)
    return value


def verify_plan(root):
    p = core.read(Path(root) / 'protocol.json')
    if fingerprint({k: v for k, v in p.items() if k != 'signature'}) != p['signature']:
        raise ValueError('training protocol changed')
    for name, digest in p['code'].items():
        if sha256(core.REPO / f'src/semantic_transmission/{name}.py') != digest:
            raise ValueError('training implementation changed; choose a new root')
    if u.model_inventory(p['runtime']['models']) != p['backbone']:
        raise ValueError('frozen backbone changed')
    for name, digest in p.get('inputs', {}).items():
        if sha256(name) != digest:
            raise ValueError('training source manifest changed: '+name)
    return p


def plan(args):
    """Freeze and audit inputs without launching codecs, model inference or training."""
    p = data_plan(args); root = args.root.resolve()
    freeze(root / 'protocol.json', p)
    report = dict(status='PLANNED_DATA_VERIFIED_TRAINING_NOT_STARTED', signature=p['signature'],
                  **p.get('data_summary', dict(sources=p['sources'], examples=len(p['examples']))))
    write_json(root / 'plan.json', report)
    print(json.dumps(report, indent=2))
    return p


def prepare(args):
    from .bsu_video import launch
    from .etri_60s_check import lock
    from .webvid_ablation import Stages
    if args.reuse_prepared:
        return import_prepared(args)
    root = args.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    if args.training_config and (root / 'protocol.json').exists():
        p = verify_plan(root)
        if args.training_config and p.get('training_config') != core.read(args.training_config):
            raise ValueError('training configuration changed; choose a new root')
        if p['selection_sha256'] != sha256(args.selection) or p['profile'] != core.profile(args.profile):
            raise ValueError('training inputs/profile changed; choose a new root')
        if p['smoke_only'] != args.smoke or p['train']['innovation_warmup_steps'] != args.codec_warmup_steps:
            raise ValueError('training mode/warmup changed; choose a new root')
    else:
        p = data_plan(args); freeze(root / 'protocol.json', p)
    for i, e in enumerate(p['examples']):
        u.event('학습용 실제 압축·수신 데이터', i, len(p['examples']), e['name'])
        out = root / 'examples' / e['name']; out.mkdir(parents=True, exist_ok=True)
        source = out / 'source'; source.mkdir(exist_ok=True)
        for t, (file, digest) in enumerate(zip(e['source_frames'], e['source_hashes'])):
            if sha256(file) != digest:
                raise ValueError('source changed during preparation')
            target = source / f'{t}.png'
            if not target.exists():
                target.symlink_to(Path(file).resolve())
            if sha256(target) != digest:
                raise ValueError('source cache mismatch')
        semantic = out / 'semantic/transmitter'; semantic.mkdir(parents=True, exist_ok=True)
        c = e['context']
        data = pack(dict(video=c['video'], keyframes=[dict(index=i) for i in c['indices']],
                         segments=c['segments'], decoder={}))
        target = semantic / 'metadata.bin'
        if target.exists() and target.read_bytes() != data:
            raise ValueError('training semantic context changed')
        target.write_bytes(data)
        profile_path = root / 'profile.json'; freeze(profile_path, p['profile'])
        command = [p['runtime']['python'], '-m', u.MODULE, '--semantic-run', semantic.parent,
                   '--source-frames', source, '--output', out / 'transport', '--base-only',
                   '--profile', profile_path, '--budget-channel-uses',
                   str(len(e['source_frames'])*e.get('budget_per_frame', p['budget_per_frame']))]
        # The transport controller acquires its own GPU lock.
        launch(command, out / f'prepare_transport_{time.time_ns()}.log')
        job = dict(output=str(out), example=e, protocol=p, root=str(root))
        freeze(out / 'job.json', job)
        stages = Stages(out, p['signature'])
        with lock(core.REPO / '.local/etri_60s_check.lock'):
            for stage, products in [('text', ['text']), ('latents', ['cache.pt', 'cache.json'])]:
                stages.step(stage, products, products, lambda log, s=stage:
                    launch([p['runtime']['python'], '-m', MODULE, '--worker', s, '--job', out / 'job.json'], log))
    write_json(root / 'prepared.json', dict(status='COMPLETE_RECEIVED_TRAINING_DATA', signature=p['signature'],
        sources=p['sources'], examples=len(p['examples']), packet_channel_and_fresh_decoder=True,
        caches={e['name']: sha256(root / 'examples' / e['name'] / 'cache.pt') for e in p['examples']}))


def import_prepared(args):
    """Reuse verified target/base caches without rewriting their old receipts."""
    import copy
    import torch
    if getattr(args, 'training_config', None):
        raise ValueError('expanded timeline training requires its own prepared sequences')
    old = args.reuse_prepared.resolve(); root = args.root.resolve()
    if root == old or root in old.parents or old in root.parents:
        raise ValueError('reused training data needs a separate output root')
    p = core.read(old / 'protocol.json'); receipt = core.read(old / 'prepared.json')
    if (fingerprint({k: v for k, v in p.items() if k != 'signature'}) != p['signature']
            or receipt['signature'] != p['signature'] or receipt['status'] != 'COMPLETE_RECEIVED_TRAINING_DATA'):
        raise ValueError('invalid prepared source protocol')
    train_sources = {e['source_sha256'] for e in p['examples'] if e['role'] == 'train'}
    validation_sources = {e['source_sha256'] for e in p['examples'] if e['role'] == 'validation'}
    if not train_sources or not validation_sources or train_sources & validation_sources:
        raise ValueError('training/validation source leakage')
    new = copy.deepcopy(p); new['profile'] = core.profile(args.profile)
    new['code'] = {n: sha256(core.REPO / f'src/semantic_transmission/{n}.py') for n in u.CODE}
    new['prepared_from'] = dict(path=str(old), protocol_sha256=sha256(old / 'protocol.json'),
                               base_codec_profile=p['profile'])
    new['smoke_only'] = p['smoke_only'] or args.smoke
    new['train']['innovation_warmup_steps'] = args.codec_warmup_steps if new['profile'].get('innovation') else 0
    if args.initialize:
        new['initialize'] = dict(path=str(args.initialize.resolve()), sha256=sha256(args.initialize))
    new.pop('signature'); new['signature'] = fingerprint(new)
    root.mkdir(parents=True, exist_ok=True); freeze(root / 'protocol.json', new)
    caches = {}
    for e in p['examples']:
        source = old / 'examples' / e['name']; dest = root / 'examples' / e['name']
        r = core.read(source / 'cache.json'); path = source / 'cache.pt'
        if (r['signature'] != p['signature'] or r['sha256'] != sha256(path)
                or receipt['caches'][e['name']] != sha256(path)):
            raise ValueError('cached training input changed')
        for file, digest in zip(e['source_frames'], e['source_hashes']):
            if sha256(file) != digest:
                raise ValueError('cached source changed')
        x = torch.load(path, map_location='cpu', weights_only=True)
        if x['signature'] != p['signature'] or x['name'] != e['name']:
            raise ValueError('cached latent provenance mismatch')
        x['signature'] = new['signature']; atomic_torch_save(x, dest / 'cache.pt')
        caches[e['name']] = sha256(dest / 'cache.pt')
        write_json(dest / 'cache.json', dict(r, signature=new['signature'], sha256=caches[e['name']],
            reused_from=str(path), reused_from_sha256=sha256(path)))
    write_json(root / 'prepared.json', dict(status='COMPLETE_RECEIVED_TRAINING_DATA', signature=new['signature'],
        sources=new['sources'], examples=len(new['examples']), caches=caches,
        reused_verified_packets_and_latents=True, fresh_channel_run=False))


def prepare_worker(stage, j):
    out = Path(j['output']); p = j['protocol']; e = j['example']
    if stage == 'text':
        return u.prepare_text(out / 'text', e['context'], p['profile'], p['runtime']['models'])
    if stage != 'latents':
        raise ValueError('invalid preparation stage')
    import torch
    from .dense_generation_runtime import guide_latents
    source = u.load_rgb(out / 'source'); base = u.load_rgb(out / 'transport/base/frames')
    side, _ = core.decode_packet((out / 'transport/received/packet.bin').read_bytes())
    modes, build = u.registry()
    vae = build(u.recipe(p['runtime']['models']).vae, modes).cuda().to(torch.bfloat16).eval()
    texts = u.text_conditions(out / 'text'); windows = []
    with torch.no_grad():
        for i, (start, end) in enumerate(core.windows(len(source), overlap=p['profile']['window_overlap'])):
            prior = guide_latents(vae, u.padded_tensor(base[start:end])).cpu()
            target = guide_latents(vae, u.padded_tensor(source[start:end])).cpu()
            windows.append(dict(prior=prior, target=target, text=texts[i], qp=side['coding']['qp'],
                video=dict(e['context']['video'], frames=end-start), start=start))
    value = dict(windows=windows, signature=p['signature'], name=e['name'])
    if not p.get('training_config'):
        value.update(windows[0])
    atomic_torch_save(value, out / 'cache.pt')
    write_json(out / 'cache.json', dict(sha256=sha256(out / 'cache.pt'), signature=p['signature'],
        received_packet_sha256=sha256(out / 'transport/received/packet.bin'),
        base_frames_sha256={f.name: sha256(f) for f in sorted((out / 'transport/base/frames').glob('*.png'))},
        source_hashes=e['source_hashes'], condition_uses_source=False, targets_use_source=True))


class Trainer:
    def __init__(self, root, p):
        import torch
        from .unified_bsu_adapter import DenseConditionAdapter
        from opensora.acceleration.checkpoint import set_grad_checkpoint
        self.root, self.p, self.torch = Path(root), p, torch
        torch.manual_seed(p['train']['seed']); torch.cuda.manual_seed_all(p['train']['seed'])
        modes, build = u.registry(); cfg = u.recipe(p['runtime']['models'])
        self.model = build(dict(cfg.model), modes, input_size=(5, 40, 72), in_channels=4,
            caption_channels=4096, model_max_length=300, enable_sequence_parallelism=False)
        self.model = self.model.cuda().to(torch.bfloat16).eval().requires_grad_(False)
        set_grad_checkpoint(self.model)
        self.adapter = DenseConditionAdapter(self.model.hidden_size, p['profile']['adapter_width']).cuda()
        if p.get('initialize'):
            if sha256(p['initialize']['path']) != p['initialize']['sha256']:
                raise ValueError('initial receiver adapter changed')
            initial = torch.load(p['initialize']['path'], map_location='cpu', weights_only=True)
            if initial['backbone'] != p['backbone']:
                raise ValueError('initial adapter backbone differs')
            self.adapter.load_state_dict(initial['state_dict'], strict=True)
        self.innovation = None
        if p['profile'].get('innovation'):
            from .unified_bsu_tokens import InnovationCodec
            config = p['profile']['innovation']
            self.innovation = InnovationCodec(config['tokens'], config['width']).cuda()
        self.handle = self.adapter.attach(self.model)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=p['train']['learning_rate'],
                                          weight_decay=p['train']['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=.5, patience=3,
            min_lr=p.get('training_config', {}).get('min_learning_rate', 0.))
        self.rng = np.random.default_rng(p['train']['seed']); self.cache = {}
        self.state = dict(step=0, order=[], offset=0, epoch=0, elapsed_seconds=0., history=[], validation=[])

    def parameters(self):
        return list(self.adapter.parameters())+(list(self.innovation.parameters()) if self.innovation else [])

    def load(self, e):
        import torch
        if e['name'] not in self.cache:
            out = self.root / 'examples' / e['name']; receipt = core.read(out / 'cache.json')
            if receipt['signature'] != self.p['signature'] or receipt['sha256'] != sha256(out / 'cache.pt'):
                raise ValueError('training cache integrity mismatch')
            value = torch.load(out / 'cache.pt', map_location='cpu', weights_only=True)
            if value['signature'] != self.p['signature'] or value['name'] != e['name']:
                raise ValueError('wrong training cache identity')
            if len(self.cache) >= 4:
                self.cache.pop(next(iter(self.cache)))
            self.cache[e['name']] = value
        return self.cache[e['name']]

    def loss(self, e, seed, fixed_time=None, window_index=0, return_error=False):
        import torch
        from .unified_bsu_adapter import condition, flow_objective
        cached = self.load(e); x = cached['windows'][window_index] if 'windows' in cached else cached
        target = x['target'].cuda().float(); prior = x['prior'].cuda().float()
        g = torch.Generator(device='cuda').manual_seed(seed)
        noise = torch.randn(target.shape, device='cuda', generator=g)
        t = (torch.rand(1, device='cuda', generator=g)*.999+.001)*1000*self.p['profile']['start_noise']
        if fixed_time is not None:
            t.fill_(fixed_time)
        z = (1-t[:, None, None, None, None]/1000)*target+t[:, None, None, None, None]/1000*noise
        codec_parts = {}
        if self.innovation:
            config = self.p['profile']['innovation']
            step = config['steps'][seed % len(config['steps'])] if fixed_time is None else config['steps'][0]
            prior, codec_parts, _ = self.innovation(target, prior, step)
            if fixed_time is None and self.state['step'] < self.p['train'].get('innovation_warmup_steps', 0):
                pixels = x['video']['frames']*x['video']['height']*x['video']['width']
                value = codec_parts['distortion']+.1*codec_parts['temporal']
                value = value+config['rate_weight']*codec_parts['rate_bits_proxy']/pixels
                result = (value, {f'codec_{k}': float(v.detach()) for k, v in codec_parts.items()})
                return (*result, None) if return_error else result
        values = condition(prior, x['qp'])
        audit_joint = self.innovation is not None and self.state['step'] == self.p['train'].get('innovation_warmup_steps', 0)
        if fixed_time is None and not audit_joint and self.rng.random() < self.p['train']['condition_dropout']:
            values = torch.zeros_like(values)
        self.adapter.received = values
        kwargs = {key: torch.tensor([x['video'][key]], device='cuda', dtype=torch.bfloat16)
                  for key in ('height', 'width', 'fps')}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = self.model(z.to(torch.bfloat16), t, y=x['text']['y'].cuda().to(torch.bfloat16),
                                    mask=x['text']['mask'].cuda(), **kwargs).chunk(2, dim=1)[0].float()
            value, parts = flow_objective(prediction, target, noise, z, t)
        if self.innovation:
            if fixed_time is None and audit_joint:
                grad = torch.autograd.grad(value, self.innovation.analysis[0].weight, retain_graph=True)[0]
                parts['flow_to_sender_grad_norm'] = grad.detach().norm()
            pixels = x['video']['frames']*x['video']['height']*x['video']['width']
            value = value+codec_parts['distortion']+.1*codec_parts['temporal']
            value = value+config['rate_weight']*codec_parts['rate_bits_proxy']/pixels
            parts.update({f'codec_{k}': v for k, v in codec_parts.items()})
        if not torch.isfinite(value):
            raise ValueError('nonfinite unified adapter loss')
        result = (value, {k: float(v.detach()) for k, v in parts.items()})
        error = z+t[:, None, None, None, None]/1000*prediction-target
        return (*result, error) if return_error else result

    def backward_sequence(self, e, seed, accumulation=1):
        """One continuous received sequence, bounded memory across its windows."""
        from .unified_bsu_training_policy import overlap_error_loss
        cached = self.load(e); windows = cached.get('windows', [cached])
        weight = self.p.get('training_config', {}).get('overlap_consistency_weight', 0.)
        total, previous, previous_start, rows = 0., None, 0, []
        for i, window in enumerate(windows):
            loss, parts, error = self.loss(e, seed+i, window_index=i, return_error=True)
            if weight and previous is not None and error is not None:
                seam = overlap_error_loss(previous, error, previous_start, window['start'])
                loss = loss+weight*seam; parts['overlap_error_consistency'] = float(seam.detach())
            (loss/(len(windows)*accumulation)).backward()
            total += float(loss.detach())/len(windows); rows.append(parts)
            previous = error.detach() if error is not None else None
            previous_start = window.get('start', 0)
        keys = set().union(*(r.keys() for r in rows))
        return total, {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}

    def validate(self):
        import torch
        rows = []; self.adapter.eval()
        if self.innovation:
            self.innovation.eval()
        with torch.no_grad():
            for e in self.p['examples']:
                if e['role'] != 'validation':
                    continue
                c = self.p.get('training_config', {})
                seeds = c.get('validation_seeds', [int(fingerprint(e['name'])[:12], 16)])
                times = c.get('validation_noise_fractions', [.5])
                cached = self.load(e); windows = cached.get('windows', [cached]); losses = []; packet_bytes = 0
                for i, x in enumerate(windows):
                    for seed in seeds:
                        for fraction in times:
                            loss, _ = self.loss(e, seed, 1000*fraction*self.p['profile']['start_noise'], i)
                            losses.append(float(loss))
                    if self.innovation:
                        target = x['target'].cuda().float(); prior = x['prior'].cuda().float()
                        blob = self.innovation.compress(target, prior, self.p['profile']['innovation']['steps'][0])
                        decoded, _ = self.innovation.decompress(blob, prior)
                        expected = self.innovation(target, prior, self.p['profile']['innovation']['steps'][0])[0]
                        if not torch.equal(decoded, expected):
                            raise ValueError('training/bitstream quantization mismatch')
                        packet_bytes += len(blob)
                row = dict(source=e['source_sha256'], dataset=e.get('dataset', 'legacy'), loss=float(np.mean(losses)),
                    windows=len(windows), noise_cases=len(seeds)*len(times))
                if self.innovation:
                    row.update(innovation_packet_bytes=packet_bytes, actual_entropy_roundtrip=True)
                rows.append(row)
        self.adapter.train()
        if self.innovation:
            self.innovation.train()
        by_dataset = {d: float(np.mean([np.mean([r['loss'] for r in rows if r['source'] == s])
            for s in {r['source'] for r in rows if r['dataset'] == d}])) for d in {r['dataset'] for r in rows}}
        return dict(step=self.state['step'], loss=float(np.mean(list(by_dataset.values()))),
                    datasets=by_dataset, rows=rows, source_disjoint=True)

    def record_validation(self):
        from .unified_bsu_training_policy import update_monitor
        value = self.validate(); self.state['validation'].append(value)
        self.scheduler.step(value['loss'])
        if self.p.get('training_config'):
            self.state['loss_monitor'], improved = update_monitor(self.state.get('loss_monitor'),
                value['loss'], self.state['step'], self.p['training_config'])
            if improved and self.state['step']:
                receipt = self.save()
                write_json(self.root / 'best_loss.json', dict(receipt, loss=value['loss'],
                    selection='validation_loss_only_not_final_video_quality'))
        return value

    def save(self):
        import torch
        value = dict(step=self.state['step'], state=self.state, adapter=self.adapter.state_dict(),
            innovation=self.innovation.state_dict() if self.innovation else None,
            optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(),
            numpy_rng=json.dumps(self.rng.bit_generator.state), cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all())
        receipt = commit_checkpoint(self.root, value, self.p['signature'])
        if self.state['step']:
            from .unified_bsu_adapter import POLICY
            export = self.root / 'exports' / f'step_{self.state["step"]:07d}.pt'
            atomic_torch_save(dict(policy=POLICY, state_dict={k: v.detach().cpu() for k, v in self.adapter.state_dict().items()},
                steps=self.state['step'], training_signature=self.p['signature'], backbone=self.p['backbone'],
                width=self.p['profile']['adapter_width'], profile=self.p['profile'],
                smoke_only=self.p['smoke_only'], validated_quality=False,
                innovation_state={k: v.detach().cpu() for k, v in self.innovation.state_dict().items()} if self.innovation else None,
                innovation_config=self.p['profile'].get('innovation')), export)
            write_json(self.root / 'export.json', dict(path=str(export), sha256=sha256(export),
                step=self.state['step'], smoke_only=self.p['smoke_only'], validated_quality=False))
        return receipt

    def restore(self, value):
        import torch
        self.adapter.load_state_dict(value['adapter'], strict=True)
        if self.innovation:
            self.innovation.load_state_dict(value['innovation'], strict=True)
        self.optimizer.load_state_dict(value['optimizer']); self.scheduler.load_state_dict(value['scheduler'])
        self.state = value['state']; self.rng.bit_generator.state = json.loads(value['numpy_rng'])
        torch.set_rng_state(value['cpu_rng']); torch.cuda.set_rng_state_all(value['cuda_rng'])
        if value['step'] != self.state['step']:
            raise ValueError('training step mismatch')


def train(args):
    import torch
    from .etri_60s_check import lock
    from .adapter_training_memory import independent_chunks
    root = args.root.resolve(); p = verify_plan(root)
    prepared = core.read(root / 'prepared.json')
    if prepared['signature'] != p['signature'] or prepared['status'] != 'COMPLETE_RECEIVED_TRAINING_DATA':
        raise ValueError('complete verified training data required')
    u.registry()
    with lock(core.REPO / '.local/etri_60s_check.lock'), lock(root / 'execution.lock'):
        trainer = Trainer(root, p); old = load_checkpoint(root, p['signature'])
        if old:
            trainer.restore(old)
        start_step = trainer.state['step']; interrupted = []
        previous = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda signum, frame: interrupted.append(signum))
        examples = [e for e in p['examples'] if e['role'] == 'train']; started = time.monotonic()
        initial_weights = {k: v.detach().cpu().clone() for k, v in trainer.adapter.state_dict().items()}
        status = 'UPDATE_LIMIT_REACHED'
        try:
            with independent_chunks() as memory:
                if not trainer.state['validation']:
                    trainer.record_validation()
                while trainer.state['step'] < p['train']['max_steps']:
                    elapsed = trainer.state['elapsed_seconds']+time.monotonic()-started
                    if args.seconds and elapsed >= args.seconds:
                        status = 'TIME_BUDGET_REACHED'; break
                    if args.stop_after and trainer.state['step']-start_step >= args.stop_after:
                        status = 'BOUNDED_SMOKE_COMPLETE' if p['smoke_only'] else 'REQUESTED_UPDATE_LIMIT'; break
                    s = trainer.state
                    trainer.optimizer.zero_grad(set_to_none=True)
                    c = p.get('training_config', {}); accumulation = c.get('gradient_accumulation', 1)
                    losses, part_rows = [], []
                    with torch.autograd.graph.save_on_cpu(pin_memory=False):
                        for _ in range(accumulation):
                            if s['offset'] >= len(s['order']):
                                from .unified_bsu_training_policy import balanced_order
                                s['order'] = (balanced_order(examples, trainer.rng) if c else
                                              trainer.rng.permutation(len(examples)).tolist())
                                s['offset'] = 0; s['epoch'] += 1
                            e = examples[s['order'][s['offset']]]
                            seed = int(trainer.rng.integers(0, 2**50))
                            loss, parts = trainer.backward_sequence(e, seed, accumulation)
                            losses.append(loss); part_rows.append(parts); s['offset'] += 1
                    parts = {k: float(np.mean([r[k] for r in part_rows if k in r]))
                             for k in set().union(*(r.keys() for r in part_rows))}
                    norm = torch.nn.utils.clip_grad_norm_(trainer.parameters(), 1., error_if_nonfinite=True)
                    sender_grad = sum(float(x.grad.detach().square().sum()) for x in trainer.innovation.parameters()
                                      if x.grad is not None)**.5 if trainer.innovation else None
                    trainer.optimizer.step(); s['step'] += 1
                    s['history'].append(dict(step=s['step'], loss=float(np.mean(losses)), grad_norm=float(norm),
                                            sender_grad_norm=sender_grad, **parts))
                    if s['step'] % p['train']['validate_every'] == 0:
                        trainer.record_validation()
                    phase = '송신 압축 사전학습' if s['step'] <= p['train'].get('innovation_warmup_steps', 0) else '송수신 공동 학습'
                    total = min(p['train']['max_steps'], start_step+args.stop_after) if args.stop_after else p['train']['max_steps']
                    if c:
                        print('COMPARE_PROGRESS '+json.dumps(dict(stage=phase,
                            overall=f'{s["step"]}/{total} updates ({100*s["step"]/total:.1f}% of update cap)',
                            current=f'{e["name"]}; epoch {s["epoch"]}; convergence ETA unknown', state='실행 중')), flush=True)
                    else:
                        u.event(phase, s['step'], total, e['name'])
                    if s['step'] % p['train']['save_every'] == 0 or interrupted:
                        s['elapsed_seconds'] += time.monotonic()-started; started = time.monotonic()
                        trainer.save()
                    if interrupted:
                        status = 'INTERRUPTED_AFTER_COMMITTED_UPDATE'; break
                if not interrupted:
                    if not trainer.state['validation'] or trainer.state['validation'][-1]['step'] != trainer.state['step']:
                        trainer.record_validation()
                trainer.state['elapsed_seconds'] += time.monotonic()-started
                receipt = trainer.save()
                changed = any(not torch.equal(v.cpu(), initial_weights[k]) for k, v in trainer.adapter.state_dict().items())
                write_json(root / 'training_result.json', dict(status=status, step=trainer.state['step'],
                    resumed_from_step=start_step, checkpoint=receipt, source_disjoint=True,
                    adapter_weights_changed_this_run=changed, backbone_frozen=not any(x.requires_grad or x.grad is not None for x in trainer.model.parameters()),
                    sender_codec_jointly_trained=trainer.innovation is not None,
                    innovation_warmup_steps=min(trainer.state['step'], p['train'].get('innovation_warmup_steps', 0)),
                    joint_steps=max(0, trainer.state['step']-p['train'].get('innovation_warmup_steps', 0)),
                    smoke_only=p['smoke_only'], performance_improvement_proven=False,
                    peak_gpu_bytes=torch.cuda.max_memory_allocated(), memory=memory,
                    elapsed_training_seconds=trainer.state['elapsed_seconds'], validation=trainer.state['validation']))
        finally:
            trainer.handle.remove()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        if interrupted:
            raise SystemExit(130)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'prepare', 'train', 'status'], nargs='?', default='status')
    parser.add_argument('--root', type=Path, default=core.REPO / 'outputs/bsu_unified_training_v1')
    parser.add_argument('--selection', type=Path, default=DEFAULT_SELECTION)
    parser.add_argument('--profile', type=Path); parser.add_argument('--budget-per-frame', type=int, default=2048)
    parser.add_argument('--limit-train', type=int, default=0); parser.add_argument('--limit-validation', type=int, default=0)
    parser.add_argument('--windows-per-video', type=int, default=2); parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--seconds', type=int, default=25200); parser.add_argument('--stop-after', type=int, default=0)
    parser.add_argument('--worker'); parser.add_argument('--job', type=Path)
    parser.add_argument('--reuse-prepared', type=Path)
    parser.add_argument('--initialize', type=Path)
    parser.add_argument('--codec-warmup-steps', type=int, default=200)
    parser.add_argument('--training-config', type=Path)
    args = parser.parse_args()
    if args.worker:
        return prepare_worker(args.worker, core.read(args.job))
    if args.seconds < 0 or args.stop_after < 0 or args.budget_per_frame < 1 or args.windows_per_video < 1 or args.codec_warmup_steps < 0:
        parser.error('invalid budget or sampling count')
    if args.action == 'plan':
        plan(args)
    elif args.action == 'prepare':
        prepare(args)
    elif args.action == 'train':
        train(args)
    else:
        root = args.root.resolve()
        print(json.dumps({name: core.read(root / name) for name in ('prepared.json', 'latest.json', 'export.json', 'training_result.json')
                          if (root / name).exists()}, indent=2))


if __name__ == '__main__':
    main()
