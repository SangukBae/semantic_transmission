import copy
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from semantic_transmission import bsu_adapter_timed_training as timed
from semantic_transmission import adapter_training_memory as memory
from semantic_transmission.adapter_training_state import commit_checkpoint, load_checkpoint
from semantic_transmission.artifacts import sha256, write_json


class TinyTrainer:
    """Real CPU AdamW/RNG state, with a clock standing in for native backward cost."""
    def __init__(self, root, p):
        self.p = p
        self.adapter = torch.nn.Linear(1, 1, bias=False)
        self.optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.rng = np.random.default_rng(42)
        self.handle = SimpleNamespace(remove=lambda: None)
        self.state = dict(step=1, microbatches=4, validation=[dict(step=0, loss=1.)],
            loss_history=[], loss_plateau=dict(best=1., bad=0, evaluations=1), best_loss=1., training_seconds=0.)

    def snapshot(self):
        return copy.deepcopy(dict(step=self.state['step'], adapter=self.adapter.state_dict(),
            optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(), training=self.state,
            numpy_rng_json=json.dumps(self.rng.bit_generator.state), cpu_rng=torch.get_rng_state()))

    def restore(self, old):
        self.adapter.load_state_dict(old['adapter']); self.optimizer.load_state_dict(old['optimizer'])
        self.scheduler.load_state_dict(old['scheduler']); self.state=copy.deepcopy(old['training'])
        self.rng.bit_generator.state=json.loads(old['numpy_rng_json']); torch.set_rng_state(old['cpu_rng'])

    def choose(self, examples):
        return examples[int(self.rng.integers(len(examples)))]

    def loss(self, *args, **kwargs):
        self.p['on_micro']()
        return self.adapter(torch.rand(1, 1)).square().mean()

    def validate_until(self, should_stop):
        self.p['on_validation']()
        if should_stop():
            raise timed.ValidationStopped()
        return dict(step=self.state['step'], loss=.9)


@pytest.fixture
def simulation(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(timed.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(memory, 'independent_chunks', nullcontext)
    config = json.loads(timed.data.DEFAULT_CONFIG.read_text())
    config.update(cpu_activation_offload=False, warmup_steps=1)
    p = dict(signature='original', config=config, examples=[dict(role='train', name='one')],
        on_micro=lambda: clock.__setitem__(0, clock[0]+1), on_validation=lambda: None)
    current = TinyTrainer(tmp_path, p).snapshot()
    current['continuation'] = dict(signature='continued', elapsed_seconds=0.)
    receipt = dict(signature='continued', initial_step=1, budget_seconds=10.)
    commit_checkpoint(tmp_path, current, p['signature'])
    return tmp_path, p, receipt, current, clock


def test_deadline_finishes_accumulated_update_and_exports_latest(simulation):
    root,p,r,current,clock = simulation
    code = timed.train(root,p,r,current,timed.Budget(10),[],TinyTrainer)
    saved = load_checkpoint(root,p['signature'])
    assert code == 0 and saved['step'] == 4
    assert saved['training']['microbatches'] == 16  # never saves a partial update
    report = json.loads((root/'RESULT.json').read_text())
    assert report['elapsed_seconds'] == 12 and report['additional_steps'] == 3
    assert not report['sufficient_training_proven'] and report['full_video_evaluation'] == 'not run'
    exported = torch.load(root/'final.pt',weights_only=True)
    torch.testing.assert_close(exported['state_dict']['weight'],saved['adapter']['weight'],rtol=0,atol=0)


def test_interrupt_resume_preserves_remaining_budget_and_exact_updates(simulation,tmp_path):
    root,p,r,current,clock = simulation
    uninterrupted = root/'reference';uninterrupted.mkdir()
    timed.train(uninterrupted,p,r,current,timed.Budget(10),[],TinyTrainer)
    reference = load_checkpoint(uninterrupted,p['signature'])
    clock[0] = 0.; interrupted = []
    def on_micro():
        clock[0] += 1
        interrupted.append(15)
    p['on_micro'] = on_micro
    assert timed.train(root,p,r,current,timed.Budget(10),interrupted,TinyTrainer) == 130
    first = load_checkpoint(root,p['signature'])
    assert first['step'] == 2 and first['continuation']['elapsed_seconds'] == 4
    assert not (root/'RESULT.json').exists()
    clock[0] += 10000  # stopped time is excluded
    p['on_micro'] = lambda: clock.__setitem__(0,clock[0]+1)
    assert timed.train(root,p,r,first,timed.Budget(10,used=4),[],TinyTrainer) == 0
    final = load_checkpoint(root,p['signature'])
    assert final['step'] == reference['step'] and final['continuation']['elapsed_seconds'] == 12
    torch.testing.assert_close(final['adapter']['weight'],reference['adapter']['weight'],rtol=0,atol=0)
    assert final['numpy_rng_json'] == reference['numpy_rng_json']
    assert torch.equal(final['cpu_rng'], reference['cpu_rng'])


def test_validation_expiry_does_not_commit_partial_validation(simulation):
    root,p,r,current,clock = simulation
    current['step'] = current['training']['step'] = 250
    p['on_validation'] = lambda: clock.__setitem__(0,11.)
    timed.train(root,p,r,current,timed.Budget(10),[],TinyTrainer)
    saved = load_checkpoint(root,p['signature'])
    assert saved['step'] == 250 and len(saved['training']['validation']) == 1
    assert saved['scheduler'] == current['scheduler']


def test_failure_mid_update_keeps_last_committed_state(simulation):
    root,p,r,current,clock = simulation
    def fail():
        clock[0] += 1
        if clock[0] == 6:
            raise RuntimeError('simulated OOM')
    p['on_micro'] = fail
    with pytest.raises(RuntimeError,match='simulated OOM'):
        timed.train(root,p,r,current,timed.Budget(10),[],TinyTrainer)
    saved = load_checkpoint(root,p['signature'])
    assert saved['step'] == 2 and saved['training']['microbatches'] == 8
    assert saved['continuation']['elapsed_seconds'] == 4 and not (root/'final.pt').exists()


def test_expired_budget_never_starts_another_update(simulation):
    root,p,r,current,clock = simulation
    timed.train(root,p,r,current,timed.Budget(10,used=10),[],TinyTrainer)
    assert load_checkpoint(root,p['signature'])['step'] == current['step']


def test_prepare_pins_parent_and_reuses_verified_cache(simulation,monkeypatch):
    source,p,r,current,clock = simulation
    # Remove test-only callbacks before writing the protocol.
    protocol = {k:v for k,v in p.items() if not k.startswith('on_')}
    write_json(source/'protocol.json',protocol)
    monkeypatch.setattr(timed.data,'protocol',lambda root,config: json.loads((root/'protocol.json').read_text()))
    code = source/'controller.py';code.write_text('pinned code')
    monkeypatch.setattr(timed,'CODE',(str(code),))
    sample = source/'data/one.pt';sample.parent.mkdir();sample.write_bytes(b'native latent cache')
    row = dict(path=str(sample),sha256=sha256(sample),signature='original',example=p['examples'][0])
    write_json(sample.with_suffix('.json'),row)
    write_json(source/'data.json',dict(signature='original',receipts=[row]))
    parent_bytes=(source/'latest.json').read_bytes()
    destination = source/'continued';destination.mkdir()
    _,receipt,copied = timed.prepare(destination,source,timed.data.DEFAULT_CONFIG,7.)
    assert copied['step'] == current['step'] and receipt['budget_seconds'] == 25200
    assert (source/'latest.json').read_bytes() == parent_bytes
    assert copied['numpy_rng_json'] == current['numpy_rng_json']
    assert copied['optimizer'] == current['optimizer'] and copied['scheduler'] == current['scheduler']
    assert torch.equal(copied['adapter']['weight'],current['adapter']['weight'])
    assert (destination/'data').is_symlink()
    timed.prepare(destination,source,timed.data.DEFAULT_CONFIG,7.)
    with pytest.raises(ValueError,match='budget differs'):
        timed.prepare(destination,source,timed.data.DEFAULT_CONFIG,8.)
    sample.write_bytes(b'changed')
    with pytest.raises(ValueError,match='data changed'):
        timed.prepare(destination,source,timed.data.DEFAULT_CONFIG,7.)


def test_continuation_rejects_active_source_lock(tmp_path):
    source=tmp_path/'source';target=tmp_path/'target';target.mkdir()
    with timed.output_lock(source):
        with pytest.raises(ValueError,match='still running'):
            timed.prepare(target,source,timed.data.DEFAULT_CONFIG,7.)
    assert not (target/'continuation.json').exists()


@pytest.mark.parametrize('seconds,used',[(0,0),(-1,0),(float('nan'),0),(1,-1),(1,float('inf'))])
def test_invalid_time_budget_rejected(seconds,used):
    with pytest.raises(ValueError):
        timed.Budget(seconds,used)


def test_validation_stop_restores_adapter_training_mode(monkeypatch):
    trainer=object.__new__(timed.TimedTrainer)
    trainer.adapter=torch.nn.Linear(1,1)
    def validate(self):
        self.adapter.eval()
        self.loss(None,0)
    monkeypatch.setattr(timed.base.NativeTrainer,'validate',validate)
    with pytest.raises(timed.ValidationStopped):
        trainer.validate_until(lambda: True)
    assert trainer.adapter.training and trainer.validation_stop is None


def test_cli_completed_budget_is_idempotent_without_loading_model(simulation,monkeypatch):
    import sys
    root,p,r,current,clock=simulation
    current['continuation']['elapsed_seconds']=r['budget_seconds']
    monkeypatch.setattr(timed,'prepare',lambda *args:(p,r,current))
    def no_training(*args,**kwargs):
        raise AssertionError('completed budget must not load or train a model')
    monkeypatch.setattr(timed,'train',no_training)
    monkeypatch.setattr(sys,'argv',['timed','run','--root',str(root)])
    assert timed.main()==0
    first=(root/'final.pt').read_bytes()
    assert timed.main()==0 and (root/'final.pt').read_bytes()==first
