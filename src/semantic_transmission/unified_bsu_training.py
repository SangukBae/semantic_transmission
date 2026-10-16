"""Source-disjoint, resumable training of the unified BSU dense condition adapter.

Preparation uses real DCVC packets and a fresh received-only decoder. Training
freezes STDiT/VAE/T5/DCVC; target pixels are available ONLY to the trainer. An
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
                   validate_every=50, save_every=10, condition_dropout=.1),
        smoke_only=args.smoke, source_disjoint=True, test_sources_used=False)
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
    return p


def prepare(args):
    from .bsu_video import launch
    from .etri_60s_check import lock
    from .webvid_ablation import Stages
    root = args.root.resolve(); root.mkdir(parents=True, exist_ok=True)
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
                   '--profile', profile_path, '--budget-channel-uses', str(17*p['budget_per_frame'])]
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
    with torch.no_grad():
        prior = guide_latents(vae, u.padded_tensor(base)).cpu()
        target = guide_latents(vae, u.padded_tensor(source)).cpu()
    text = u.text_conditions(out / 'text')[0]
    value = dict(prior=prior, target=target, text=text, qp=side['coding']['qp'],
        video=e['context']['video'], signature=p['signature'], name=e['name'])
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
        self.handle = self.adapter.attach(self.model)
        self.optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=p['train']['learning_rate'],
                                          weight_decay=p['train']['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=.5, patience=3)
        self.rng = np.random.default_rng(p['train']['seed']); self.cache = {}
        self.state = dict(step=0, order=[], offset=0, epoch=0, elapsed_seconds=0., history=[], validation=[])

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

    def loss(self, e, seed, fixed_time=None):
        import torch
        from .unified_bsu_adapter import condition, flow_objective
        x = self.load(e); target = x['target'].cuda().float(); prior = x['prior'].cuda().float()
        g = torch.Generator(device='cuda').manual_seed(seed)
        noise = torch.randn(target.shape, device='cuda', generator=g)
        t = (torch.rand(1, device='cuda', generator=g)*.999+.001)*1000*self.p['profile']['start_noise']
        if fixed_time is not None:
            t.fill_(fixed_time)
        z = (1-t[:, None, None, None, None]/1000)*target+t[:, None, None, None, None]/1000*noise
        values = condition(prior, x['qp'])
        if fixed_time is None and self.rng.random() < self.p['train']['condition_dropout']:
            values = torch.zeros_like(values)
        self.adapter.received = values
        kwargs = {key: torch.tensor([x['video'][key]], device='cuda', dtype=torch.bfloat16)
                  for key in ('height', 'width', 'fps')}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = self.model(z.to(torch.bfloat16), t, y=x['text']['y'].cuda().to(torch.bfloat16),
                                    mask=x['text']['mask'].cuda(), **kwargs).chunk(2, dim=1)[0].float()
            value, parts = flow_objective(prediction, target, noise, z, t)
        if not torch.isfinite(value):
            raise ValueError('nonfinite unified adapter loss')
        return value, {k: float(v.detach()) for k, v in parts.items()}

    def validate(self):
        import torch
        rows = []; self.adapter.eval()
        with torch.no_grad():
            for e in self.p['examples']:
                if e['role'] != 'validation':
                    continue
                seed = int(fingerprint(e['name'])[:12], 16)
                loss, _ = self.loss(e, seed, 500*self.p['profile']['start_noise'])
                rows.append(dict(source=e['source_sha256'], loss=float(loss)))
        self.adapter.train()
        value = float(np.mean([np.mean([r['loss'] for r in rows if r['source'] == source])
                               for source in {r['source'] for r in rows}]))
        return dict(step=self.state['step'], loss=value, rows=rows, source_disjoint=True)

    def save(self):
        import torch
        value = dict(step=self.state['step'], state=self.state, adapter=self.adapter.state_dict(),
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
                smoke_only=self.p['smoke_only'], validated_quality=False), export)
            write_json(self.root / 'export.json', dict(path=str(export), sha256=sha256(export),
                step=self.state['step'], smoke_only=self.p['smoke_only'], validated_quality=False))
        return receipt

    def restore(self, value):
        import torch
        self.adapter.load_state_dict(value['adapter'], strict=True)
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
                    trainer.state['validation'].append(trainer.validate())
                while trainer.state['step'] < p['train']['max_steps']:
                    elapsed = trainer.state['elapsed_seconds']+time.monotonic()-started
                    if elapsed >= args.seconds:
                        status = 'TIME_BUDGET_REACHED'; break
                    if args.stop_after and trainer.state['step']-start_step >= args.stop_after:
                        status = 'BOUNDED_SMOKE_COMPLETE' if p['smoke_only'] else 'REQUESTED_UPDATE_LIMIT'; break
                    s = trainer.state
                    if s['offset'] >= len(s['order']):
                        s['order'] = trainer.rng.permutation(len(examples)).tolist(); s['offset'] = 0; s['epoch'] += 1
                    e = examples[s['order'][s['offset']]]
                    trainer.optimizer.zero_grad(set_to_none=True)
                    seed = int(trainer.rng.integers(0, 2**50))
                    with torch.autograd.graph.save_on_cpu(pin_memory=False):
                        loss, parts = trainer.loss(e, seed); loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(trainer.adapter.parameters(), 1., error_if_nonfinite=True)
                    trainer.optimizer.step(); s['step'] += 1; s['offset'] += 1
                    s['history'].append(dict(step=s['step'], loss=float(loss.detach()), grad_norm=float(norm), **parts))
                    if s['step'] % p['train']['validate_every'] == 0:
                        v = trainer.validate(); s['validation'].append(v); trainer.scheduler.step(v['loss'])
                    u.event('학습 업데이트', s['step'], p['train']['max_steps'], e['name'])
                    if s['step'] % p['train']['save_every'] == 0 or interrupted:
                        s['elapsed_seconds'] += time.monotonic()-started; started = time.monotonic()
                        trainer.save()
                    if interrupted:
                        status = 'INTERRUPTED_AFTER_COMMITTED_UPDATE'; break
                if not interrupted:
                    v = trainer.validate(); trainer.state['validation'].append(v); trainer.scheduler.step(v['loss'])
                trainer.state['elapsed_seconds'] += time.monotonic()-started
                receipt = trainer.save()
                changed = any(not torch.equal(v.cpu(), initial_weights[k]) for k, v in trainer.adapter.state_dict().items())
                write_json(root / 'training_result.json', dict(status=status, step=trainer.state['step'],
                    resumed_from_step=start_step, checkpoint=receipt, source_disjoint=True,
                    adapter_weights_changed_this_run=changed, backbone_frozen=not any(x.requires_grad or x.grad is not None for x in trainer.model.parameters()),
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
    parser.add_argument('action', choices=['prepare', 'train', 'status'], nargs='?', default='status')
    parser.add_argument('--root', type=Path, default=core.REPO / 'outputs/bsu_unified_training_v1')
    parser.add_argument('--selection', type=Path, default=DEFAULT_SELECTION)
    parser.add_argument('--profile', type=Path); parser.add_argument('--budget-per-frame', type=int, default=2048)
    parser.add_argument('--limit-train', type=int, default=0); parser.add_argument('--limit-validation', type=int, default=0)
    parser.add_argument('--windows-per-video', type=int, default=2); parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--seconds', type=int, default=25200); parser.add_argument('--stop-after', type=int, default=0)
    parser.add_argument('--worker'); parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    if args.worker:
        return prepare_worker(args.worker, core.read(args.job))
    if args.seconds < 1 or args.stop_after < 0 or args.budget_per_frame < 1 or args.windows_per_video < 1:
        parser.error('invalid budget or sampling count')
    if args.action == 'prepare':
        prepare(args)
    elif args.action == 'train':
        train(args)
    else:
        root = args.root.resolve()
        print(json.dumps({name: core.read(root / name) for name in ('prepared.json', 'latest.json', 'export.json', 'training_result.json')
                          if (root / name).exists()}, indent=2))


if __name__ == '__main__':
    main()
