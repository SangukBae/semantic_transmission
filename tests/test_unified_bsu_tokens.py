import copy

import numpy as np
import pytest
import torch

from semantic_transmission import unified_bsu_tokens as tokens
from semantic_transmission import unified_bsu_core as core
from semantic_transmission.wire import pack, unpack


@pytest.mark.parametrize('step', range(4))
def test_integer_entropy_roundtrip_and_temporal_prediction(step):
    rng = np.random.default_rng(21)
    q = rng.integers(-64, 65, (5, 16, 8), dtype=np.int16)
    roi = rng.integers(0, 2, (5, 9), dtype=np.uint8)
    data = tokens.encode_symbols(q, roi, step)
    restored, mask, decoded_step = tokens.decode_symbols(data)
    assert np.array_equal(restored, q) and np.array_equal(mask, roi) and decoded_step == step
    repeated = np.repeat(q[:1], 5, axis=0)
    assert len(tokens.encode_symbols(repeated, roi, step)) < len(data)
    for broken in (data[:-1], data+b'\0', b'BAD!'+data[4:]):
        with pytest.raises(ValueError):
            tokens.decode_symbols(broken)


def test_learned_sender_gradient_and_received_only_decoder(tmp_path):
    torch.manual_seed(1)
    codec = tokens.InnovationCodec(tokens=8)
    prior = torch.randn(1, 4, 5, 16, 24)
    target = prior+.2*torch.randn_like(prior)
    restored, parts, roi = codec(target, prior)
    blob = codec.compress(target, prior)
    other = tokens.InnovationCodec(tokens=8)
    other.load_state_dict(codec.state_dict())
    received, paid_roi = other.decompress(blob, prior)
    assert torch.equal(received, restored)
    assert torch.equal(roi, paid_roi)
    loss = parts['distortion']+.1*parts['temporal']+1e-5*parts['rate_bits_proxy']
    loss.backward()
    assert codec.analysis[0].weight.grad.abs().sum() > 0
    assert codec.queries.grad.abs().sum() > 0
    assert codec.synthesis[-1].weight.grad.abs().sum() > 0
    assert torch.isfinite(parts['rate_bits_proxy'])


def test_all_enhancement_bytes_paid_and_packet_validates_boundaries():
    context = dict(video=dict(frames=17, height=64, width=96, fps=24), indices=[0, 16],
                   segments=[dict(path='clips/sample/0_16.mp4', text='A person walks.', flow=1.)], decoder={})
    coding = dict(indices=list(range(0, 17, 2)), qp=24, stride=2, scale=1)
    stream = pack(dict(indices=coding['indices'], height=64, width=96), b'video-data'*100)
    decoder = dict(policy=core.POLICY, profile=dict(compact_wire=True, policy=core.POLICY,
        window_frames=17, window_overlap=5, references={'unused': 'not-needed-at-receiver'},
        innovation=dict(tokens=8, width=32, steps=[1], rate_weight=.01)))
    blob = tokens.encode_symbols(np.zeros((5, 8, 8), np.int16), np.zeros((5, 9), np.uint8), 1)
    old = core.encode_packet(context, stream, coding, decoder)
    data = core.encode_packet(context, stream, coding, decoder, [blob])
    side, received_stream = core.decode_packet(data)
    assert side['enhancements'] == [blob] and received_stream == stream
    assert 'references' not in side['decoder']['profile']
    assert len(data) > len(old)
    report = core.packet_breakdown(data)
    assert report['innovation_bytes'] == len(blob)
    assert report['packet_bytes'] == sum(report[k] for k in ('header_bytes', 'side_bytes', 'video_stream_bytes', 'innovation_bytes'))
    h, payload = unpack(data); h['innovation_lengths'][0] += 1
    with pytest.raises(ValueError, match='innovation'):
        core.decode_packet(pack(h, payload))
    h, payload = unpack(data)
    with pytest.raises(ValueError, match='innovation'):
        core.decode_packet(pack(h, payload[:-1]+bytes([payload[-1] ^ 1])))


def test_roi_cost_is_real_and_no_semantic_labels_needed():
    prior = torch.zeros(1, 4, 5, 40, 72); target = prior.clone()
    target[..., 8:16, 16:24] = 5
    roi = tokens.importance(target, prior)
    assert roi.sum() == 9
    assert roi[0, 0, 0, 1, 2] == 1
    assert not roi.requires_grad


def test_budget_rejects_added_information_that_exceeds_total_limit():
    rows = [dict(name='base', channel_uses=100, distortion=dict(score=2)),
            dict(name='tokens', channel_uses=200, distortion=dict(score=1))]
    assert core.choose_candidate(rows, 100)['name'] == 'base'


def test_joint_checkpoint_restores_codec_optimizer_rng_and_data_cursor(tmp_path, monkeypatch):
    from semantic_transmission.unified_bsu_training import Trainer
    from semantic_transmission.adapter_training_state import load_checkpoint
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda states: None)
    trainer = Trainer.__new__(Trainer)
    trainer.root = tmp_path
    trainer.p = dict(signature='joint-checkpoint-test', backbone={}, smoke_only=True,
                     profile=dict(adapter_width=4, innovation=dict(tokens=8, width=32)))
    trainer.adapter = torch.nn.Linear(2, 2); trainer.innovation = torch.nn.Linear(2, 2)
    trainer.optimizer = torch.optim.AdamW(trainer.parameters(), lr=.01)
    trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer)
    trainer.rng = np.random.default_rng(42)
    trainer.state = dict(step=1, order=[1, 0], offset=1, epoch=1, elapsed_seconds=2.)
    trainer.adapter(trainer.innovation(torch.ones(1, 2))).square().sum().backward()
    trainer.optimizer.step(); trainer.save()
    saved = load_checkpoint(tmp_path, trainer.p['signature'])
    expected = [p.detach().clone() for p in trainer.parameters()]
    cpu_rng = torch.get_rng_state().clone(); next_numpy = trainer.rng.random()
    with torch.no_grad():
        for p in trainer.parameters():
            p.add_(10)
    trainer.state = dict(step=999); torch.rand(20); trainer.rng.random(20)
    trainer.restore(saved)
    assert all(torch.equal(a, b) for a, b in zip(expected, trainer.parameters()))
    assert trainer.state['step'] == 1 and trainer.state['offset'] == 1 and trainer.state['order'] == [1, 0]
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert trainer.rng.random() == next_numpy
    assert all(int(s['step']) == 1 for s in trainer.optimizer.state.values())
