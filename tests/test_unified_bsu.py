import copy
from types import SimpleNamespace

import numpy as np
import pytest

from semantic_transmission import unified_bsu_core as core
from semantic_transmission.wire import pack, unpack


def context(frames=17):
    return dict(video=dict(frames=frames, height=64, width=96, fps=24), indices=[0, frames-1],
        segments=[dict(path=f'clips/sample/0_{frames-1}.mp4', text='A person walks.', flow=1.)], decoder={})


def packet():
    indices = list(range(0, 17, 2))
    stream = pack(dict(indices=indices, height=64, width=96), b'encoded-video')
    return core.encode_packet(context(), stream, dict(indices=indices, qp=32, stride=2, scale=1),
                              dict(policy=core.POLICY))


def test_packet_accounts_all_conditions_and_fails_on_corruption():
    data = packet(); side, stream = core.decode_packet(data)
    assert side['context'] == context()
    assert core.channel_uses(data) >= core.channel_uses(stream)
    with pytest.raises(ValueError):
        core.decode_packet(data[:-1]+bytes([data[-1] ^ 1]))
    h, p = unpack(data); h['stream_sha256'] = '0'*64
    with pytest.raises(ValueError, match='integrity'):
        core.decode_packet(pack(h, p))


def test_reject_packet_with_unpaid_or_wrong_timeline():
    side, stream = core.decode_packet(packet())
    side['coding']['indices'] = [0, 3, 16]
    malformed = core.encode_packet(side['context'], stream, side['coding'], side['decoder'])
    with pytest.raises(ValueError):
        core.decode_packet(malformed)


def test_adaptive_sampler_protects_small_fast_motion():
    x = np.zeros((5, 64, 96, 3), np.uint8)
    x[1, :8, :8] = 255
    assert np.abs(x[1].astype(float)-x[0]).mean() < 12
    indices, scores = core.adaptive_indices(x, 4, [0, 4])
    assert {0, 1, 2, 4} <= set(indices)
    assert scores[0] > 12
    static, _ = core.adaptive_indices(np.zeros_like(x), 4)
    assert static == [0, 4]


@pytest.mark.parametrize('n', [2, 16, 17, 18, 29, 30, 60, 721])
def test_generation_windows_cover_full_timeline(n):
    count = np.zeros(n, int)
    for start, end in core.windows(n):
        assert 0 <= start < end <= n and end-start <= 17
        count[start:end] += 1
    assert count.min() > 0


def test_budget_selection_does_not_silently_exceed_cap():
    rows = [dict(name='sharp', channel_uses=120, distortion=dict(score=1)),
            dict(name='small', channel_uses=80, distortion=dict(score=2))]
    assert core.choose_candidate(rows, 100)['name'] == 'small'
    with pytest.raises(ValueError, match='no candidate'):
        core.choose_candidate(rows, 50)


def test_projection_rejects_structure_disagreement_and_bounds_changes():
    rng = np.random.default_rng(42)
    base = rng.integers(0, 256, (3, 32, 32, 3), dtype=np.uint8)
    identity, audit = core.bounded_refinement(base, base, 32)
    assert np.array_equal(identity, base)
    result, audit = core.bounded_refinement(base, 255-base, 16)
    assert np.abs(result.astype(int)-base.astype(int)).max() <= 8
    assert audit['source_used'] is False and audit['semantic_preservation_proven'] is False


def test_received_adapter_identity_then_real_gradient_update():
    import torch
    from semantic_transmission.unified_bsu_adapter import DenseConditionAdapter, condition
    torch.manual_seed(10)
    model = DenseConditionAdapter(hidden=8, width=4)
    prior = torch.randn(1, 4, 5, 5, 7)
    x = condition(prior, 24)
    value = model(x)
    assert value.shape == (1, 5*3*4, 8)
    assert torch.count_nonzero(value) == 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    (value-1).square().mean().backward(); optimizer.step()
    assert torch.count_nonzero(model(x)) > 0


def test_partial_sampler_uses_received_prior_and_multiple_steps():
    import torch
    from semantic_transmission.unified_bsu_adapter import sample
    class Constant(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.zeros(())); self.calls = []
        def forward(self, x, time, **kwargs):
            self.calls.append((x.clone(), time.clone()))
            return torch.zeros(x.shape[0], 8, *x.shape[2:])
    model = Constant(); adapter = SimpleNamespace(received=None)
    prior = torch.ones(1, 4, 5, 4, 4)
    text = dict(y=torch.zeros(1, 1, 3, 8), mask=torch.ones(1, 3))
    result = sample(model, adapter, prior, text, text['y'], qp=24, height=32, width=32, fps=24,
                    steps=4, start_noise=.3, seed=2025)
    noise = torch.randn(prior.shape, generator=torch.Generator().manual_seed(2025))
    assert torch.allclose(result, .7*prior+.3*noise)
    assert len(model.calls) == 4
    assert [float(x[1][0]) for x in model.calls] == [300, 225, 150, 75]


def test_flow_supervision_has_zero_loss_for_exact_velocity():
    import torch
    from semantic_transmission.unified_bsu_adapter import flow_objective
    target = torch.randn(1, 4, 5, 4, 4); noise = torch.randn_like(target)
    time = torch.tensor([200.]); z = .8*target+.2*noise
    loss, _ = flow_objective(target-noise, target, noise, z, time)
    assert float(loss) < 1e-6
