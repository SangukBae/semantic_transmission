"""Checkpoint transactions and stopping rules, independent of model loading."""
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import uuid
from .artifacts import sha256, write_json


def validate_config(c):
    for k in ('min_steps','max_steps','gradient_accumulation','save_every','validate_every',
              'quality_every','loss_patience','quality_patience','warmup_steps'):
        if type(c[k]) is not int or c[k] < 1:
            raise ValueError(f'{k} must be a positive integer')
    if c['min_steps'] > c['max_steps'] or c['quality_every'] % c['validate_every']:
        raise ValueError('invalid minimum/maximum steps or validation intervals')
    if c['validate_every'] % c['save_every']:
        raise ValueError('validate_every must be divisible by save_every')
    if not 0 < c['min_learning_rate'] <= c['learning_rate'] or not 0 <= c['condition_dropout'] < 1:
        raise ValueError('invalid learning rate or condition dropout')
    if not 0 < c['relative_min_delta'] < 1 or not c['validation_seeds']:
        raise ValueError('invalid plateau threshold or validation seeds')
    if any(not 0 < t < 1000 for t in c['validation_timesteps']) or not c['validation_timesteps']:
        raise ValueError('validation timesteps must lie strictly between 0 and 1000')
    return c


def update_plateau(state, value, relative_delta):
    if not math.isfinite(value) or value < 0:
        raise ValueError('invalid validation objective')
    state = copy.deepcopy(state or dict(best=None, bad=0, evaluations=0))
    improved = state['best'] is None or value < state['best'] * (1-relative_delta)
    if improved:
        state.update(best=value, bad=0)
    else:
        state['bad'] += 1
    state['evaluations'] += 1
    return state, improved


def stop_reason(step, config, loss, quality):
    if (step >= config['min_steps'] and loss and quality
            and loss['bad'] >= config['loss_patience']
            and quality['bad'] >= config['quality_patience']):
        return 'VALIDATION_PLATEAU'
    if step >= config['max_steps']:
        return 'MAX_STEPS_WITHOUT_CONFIRMED_PLATEAU'
    return None


def atomic_torch_save(value, path):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.checkpoint-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            torch.save(value, f); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def commit_checkpoint(root, value, signature):
    """An interrupted write can leave an orphan, never a partially updated head."""
    root = Path(root)
    path = root/'checkpoints'/f'step_{value["step"]:07d}_{uuid.uuid4().hex[:10]}.pt'
    atomic_torch_save(dict(value, signature=signature), path)
    receipt = dict(path=str(path), sha256=sha256(path), signature=signature, step=value['step'])
    write_json(root/'latest.json', receipt)
    return receipt


def load_checkpoint(root, signature):
    import torch
    path = Path(root)/'latest.json'
    if not path.exists():
        return None
    head = json.loads(path.read_text())
    if head['signature'] != signature or sha256(head['path']) != head['sha256']:
        raise ValueError('checkpoint signature/hash differs; use a new output directory')
    data = torch.load(head['path'], map_location='cpu', weights_only=True)
    if data['signature'] != signature or data['step'] != head['step']:
        raise ValueError('checkpoint payload and receipt differ')
    return data
