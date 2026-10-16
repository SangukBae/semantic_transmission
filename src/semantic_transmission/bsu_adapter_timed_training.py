"""Continue a verified native adapter checkpoint for a persistent time budget.

The original protocol and caches keep their original signatures. A separate
receipt pins this controller, its budget, and the exact parent checkpoint.
Full-video quality evaluation is deferred; final.pt is the latest, not the best.
"""
import argparse
from contextlib import contextmanager, nullcontext
import fcntl
import math
from pathlib import Path
import signal
import time

from . import adapter_training_data as data
from . import bsu_adapter_training as base
from .adapter_training_state import atomic_torch_save, commit_checkpoint, load_checkpoint, update_plateau
from .artifacts import sha256, write_json
from .terminal_progress import duration

DEFAULT_ROOT = data.common.REPO / 'outputs/bsu_adapter_training_7h_v1'
CODE = ('src/semantic_transmission/bsu_adapter_timed_training.py', 'scripts/train_bsu_adapter_7h.sh',
        'src/semantic_transmission/terminal_progress.py', 'src/semantic_transmission/terminal_gpu_progress.py')


@contextmanager
def output_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'training is still running in {root}; wait for its checkpoint and exit')
        yield


def verify_data(root, p):
    prepared = data.common.read(root / 'data.json')
    if prepared['signature'] != p['signature']:
        raise ValueError('prepared data signature differs')
    expected = {e['name']: e for e in p['examples']}
    if len(prepared['receipts']) != len(expected):
        raise ValueError('incomplete prepared data')
    seen = set()
    for receipt in prepared['receipts']:
        example = receipt['example']; name = example['name']
        path = root / 'data' / f'{name}.pt'
        if name in seen or example != expected.get(name) or receipt['signature'] != p['signature']:
            raise ValueError('prepared example differs')
        if path.resolve() != Path(receipt['path']).resolve() or sha256(path) != receipt['sha256']:
            raise ValueError(f'prepared data changed: {name}')
        if data.common.read(path.with_suffix('.json')) != receipt:
            raise ValueError(f'prepared data receipt changed: {name}')
        seen.add(name)


def prepare(root, source, config, hours):
    """Called under the destination lock. Never modifies source artifacts."""
    if root == source:
        raise ValueError('continuation requires a separate output directory')
    manifest = root / 'continuation.json'
    if not manifest.exists():
        if (root / 'latest.json').exists() or (root / 'protocol.json').exists():
            raise ValueError('destination already contains a different run')
        with output_lock(source):
            p = data.protocol(source, config)
            old = load_checkpoint(source, p['signature'])
            if old is None or old['step'] < 1:
                raise ValueError('a committed training checkpoint is required')
            verify_data(source, p)
            receipt = dict(version=1, source=str(source), parent=data.common.read(source / 'latest.json'),
                protocol_signature=p['signature'], budget_seconds=hours * 3600,
                initial_step=old['step'], resume_optimizer_scheduler_rng=True,
                full_video_evaluation='deferred',
                code={str(data.common.REPO / n): sha256(data.common.REPO / n) for n in CODE})
            receipt['signature'] = data.fingerprint(receipt)
            # Manifest first: interrupted preparation can restart from this pinned parent.
            write_json(manifest, receipt)
    receipt = data.common.read(manifest)
    if data.fingerprint({k: v for k, v in receipt.items() if k != 'signature'}) != receipt['signature']:
        raise ValueError('continuation receipt changed')
    if receipt['source'] != str(source) or receipt['budget_seconds'] != hours * 3600:
        raise ValueError('source/time budget differs; use a new --root')
    for path, digest in receipt['code'].items():
        if sha256(path) != digest:
            raise ValueError(f'continuation code changed: {path}; use a new --root')
    parent = receipt['parent']
    if sha256(parent['path']) != parent['sha256']:
        raise ValueError('parent checkpoint changed')
    if not (root / 'protocol.json').exists():
        write_json(root / 'protocol.json', data.common.read(source / 'protocol.json'))
    p = data.protocol(root, config)
    if p['signature'] != receipt['protocol_signature']:
        raise ValueError('parent protocol differs')
    cache = root / 'data'
    if not cache.exists():
        cache.symlink_to(source / 'data', target_is_directory=True)
    if not (root / 'data.json').exists():
        write_json(root / 'data.json', data.common.read(source / 'data.json'))
    verify_data(root, p)
    current = load_checkpoint(root, p['signature'])
    if current is None:
        import torch
        current = torch.load(parent['path'], map_location='cpu', weights_only=True)
        if current['signature'] != p['signature'] or current['step'] != parent['step']:
            raise ValueError('parent payload differs')
        current['continuation'] = dict(signature=receipt['signature'], elapsed_seconds=0.)
        commit_checkpoint(root, current, p['signature'])
    if current.get('continuation', {}).get('signature') != receipt['signature']:
        raise ValueError('checkpoint continuation identity differs')
    base.export_adapter(root, current, p['signature'])
    return p, receipt, current


class Budget:
    def __init__(self, seconds, used=0., started=None):
        if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(used) or used < 0:
            raise ValueError('invalid time budget')
        self.seconds, self.used = seconds, used
        self.started = time.monotonic() if started is None else started

    def elapsed(self):
        return self.used + time.monotonic() - self.started

    def expired(self):
        return self.elapsed() >= self.seconds


class ValidationStopped(Exception):
    pass


class TimedTrainer(base.NativeTrainer):
    def validate_until(self, should_stop):
        self.validation_stop = should_stop
        try:
            return super().validate()
        finally:
            self.validation_stop = None
            self.adapter.train()

    def loss(self, *args, **kwargs):
        if getattr(self, 'validation_stop', None) and self.validation_stop():
            raise ValidationStopped()
        return super().loss(*args, **kwargs)


def progress(root, budget, step, initial, stage, state='실행 중'):
    elapsed = budget.elapsed(); remaining = max(0., budget.seconds - elapsed)
    base.event(root, stage,
        overall=f'시간 예산 {duration(elapsed)}/{duration(budget.seconds)} ({min(100., 100*elapsed/budget.seconds):.1f}%)',
        current=f'추가 {step-initial}회 · 누적 {step}회 · 예산 종료까지 {duration(remaining)} + 진행 중 업데이트 저장',
        state=state)


def finish(root, p, receipt, current):
    path = base.export_adapter(root, current, p['signature'])
    import torch
    final = root / 'final.pt'
    atomic_torch_save(torch.load(path, map_location='cpu', weights_only=True), final)
    report = dict(status='TRAINING_BUDGET_COMPLETE_EVALUATION_PENDING',
        stop_reason='WALL_TIME_BUDGET', signature=receipt['signature'],
        initial_step=receipt['initial_step'], steps=current['step'],
        additional_steps=current['step']-receipt['initial_step'],
        budget_seconds=receipt['budget_seconds'], elapsed_seconds=current['continuation']['elapsed_seconds'],
        final_checkpoint=str(final), final_checkpoint_sha256=sha256(final),
        latest=data.common.read(root / 'latest.json'),
        full_video_evaluation='not run', sufficient_training_proven=False,
        production_default_changed=False)
    write_json(root / 'final.json', dict(path=str(final), sha256=sha256(final), step=current['step']))
    write_json(root / 'RESULT.json', report)
    return report


def train(root, p, receipt, current, budget, interrupted, trainer_factory=TimedTrainer):
    import torch
    from .adapter_training_memory import independent_chunks
    config = p['config']; initial = receipt['initial_step']
    should_stop = lambda: bool(interrupted) or budget.expired()
    progress(root, budget, current['step'], initial, '검증된 체크포인트에서 모델 로딩')
    trainer = trainer_factory(root, p)
    trainer.restore(current)
    print(f'RESUME: verified weights/optimizer/scheduler/RNG/data at step {current["step"]}; '
          f'{duration(max(0., budget.seconds-budget.used))} budget remaining', flush=True)
    examples = [e for e in p['examples'] if e['role'] == 'train']
    accounted = time.monotonic()

    def save():
        nonlocal accounted, current
        now = time.monotonic()
        trainer.state['training_seconds'] += now - accounted; accounted = now
        current = trainer.snapshot()
        current['continuation'] = dict(signature=receipt['signature'], elapsed_seconds=budget.elapsed())
        commit_checkpoint(root, current, p['signature'])
        base.export_adapter(root, current, p['signature'])

    try:
        with independent_chunks():
            while not should_stop():
                step = trainer.state['step']
                # Resume a validation interrupted by Ctrl-C before advancing again.
                if (step % config['validate_every'] == 0 and
                        not any(v['step'] == step for v in trainer.state['validation'])):
                    try:
                        v = trainer.validate_until(should_stop)
                    except ValidationStopped:
                        break
                    trainer.state['validation'].append(v)
                    trainer.state['loss_plateau'], _ = update_plateau(
                        trainer.state['loss_plateau'], v['loss'], config['relative_min_delta'])
                    trainer.scheduler.step(v['loss'])
                    trainer.state['best_loss'] = min(trainer.state['best_loss'], v['loss'])
                    write_json(root / 'validation_loss.json', dict(rows=trainer.state['validation']))
                    save()
                    if should_stop():
                        break
                trainer.optimizer.zero_grad(set_to_none=True); loss_sum = 0.
                if step < config['warmup_steps']:
                    for group in trainer.optimizer.param_groups:
                        group['lr'] = config['learning_rate']*(step+1)/config['warmup_steps']
                progress(root, budget, step, initial, 'optimizer update 진행 중')
                # Finish all accumulated microbatches to preserve exact optimizer/RNG order.
                for _ in range(config['gradient_accumulation']):
                    e = trainer.choose(examples)
                    seed = int(trainer.rng.integers(0, 2**50))
                    drop = trainer.rng.random() < config['condition_dropout']
                    offload = torch.autograd.graph.save_on_cpu(pin_memory=False) if config['cpu_activation_offload'] else nullcontext()
                    with offload:
                        loss = trainer.loss(e, seed, drop=drop) / config['gradient_accumulation']
                        loss.backward()
                    loss_sum += float(loss.detach()); trainer.state['microbatches'] += 1
                norm = torch.nn.utils.clip_grad_norm_(trainer.adapter.parameters(), 1., error_if_nonfinite=True)
                trainer.optimizer.step(); trainer.state['step'] += 1
                trainer.state['loss_history'].append(dict(step=trainer.state['step'], loss=loss_sum,
                    grad_norm=float(norm), lr=trainer.optimizer.param_groups[0]['lr']))
                # Small adapter state: persist every update, limiting lost work on hard failure.
                save()
                progress(root, budget, trainer.state['step'], initial, 'optimizer update 저장 완료',
                    state=f'체크포인트 저장 완료 · loss={loss_sum:.5f}')
            save()
    finally:
        trainer.handle.remove()
    if interrupted and not budget.expired():
        progress(root, budget, current['step'], initial, '중단 · 남은 시간은 같은 명령으로 재개',
            state='중단 · 가중치/옵티마이저/RNG/남은 시간 저장 완료')
        return 130
    finish(root, p, receipt, current)
    progress(root, budget, current['step'], initial, '시간 예산 종료 · final.pt 저장 완료',
        state='추가 학습 완료 · 전체 영상 품질 평가 대기')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'prepare', 'run', 'status'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--source', type=Path, default=data.DEFAULT_ROOT)
    parser.add_argument('--config', type=Path, default=data.DEFAULT_CONFIG)
    parser.add_argument('--hours', type=float, default=7.)
    args = parser.parse_args()
    if not math.isfinite(args.hours) or args.hours <= 0:
        parser.error('--hours must be positive and finite')
    root, source, config = args.root.resolve(), args.source.resolve(), args.config.resolve()
    if args.command == 'status':
        for name in ('status.json', 'continuation.json', 'latest.json', 'final.json', 'RESULT.json'):
            if (root / name).exists():
                print(name+'\n'+(root / name).read_text())
        if not root.exists():
            print('NOT_STARTED: '+str(root))
        return
    started = time.monotonic(); interrupted = []
    handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        for s in handlers:
            signal.signal(s, lambda signum, frame: interrupted.append(signum))
        with output_lock(root):
            p, receipt, current = prepare(root, source, config, args.hours)
            if args.command != 'run':
                print(f'CHECK: verified step={current["step"]}, optimizer/RNG/cache retained, '
                      f'budget={duration(receipt["budget_seconds"])}, used={duration(current["continuation"]["elapsed_seconds"])}, '
                      f'root={root}; training not started', flush=True)
                return 0
            if (root / 'RESULT.json').exists():
                report = data.common.read(root / 'RESULT.json')
                if report['signature'] != receipt['signature'] or sha256(report['final_checkpoint']) != report['final_checkpoint_sha256']:
                    raise ValueError('completed continuation changed')
                print('RESUME: budget already complete; '+str(root / 'final.pt'), flush=True)
                return 0
            budget = Budget(receipt['budget_seconds'], current['continuation']['elapsed_seconds'], started)
            if budget.expired() or interrupted:
                current['continuation']['elapsed_seconds'] = budget.elapsed()
                commit_checkpoint(root, current, p['signature'])
                if budget.expired():
                    finish(root, p, receipt, current)
                    return 0
                return 130
            return train(root, p, receipt, current, budget, interrupted)
    finally:
        for s, handler in handlers.items():
            signal.signal(s, handler)


if __name__ == '__main__':
    raise SystemExit(main())
