import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

from semantic_transmission import bsu_predict_jscc as model
from semantic_transmission import bsu_predict as base
from semantic_transmission import bsu_metadata_compression as compression
from semantic_transmission import dense_temporal_receiver as temporal
from semantic_transmission import metadata_channel
from semantic_transmission import ntscc_plus_codec as codec
from semantic_transmission.artifacts import write_json
from semantic_transmission.packets import accounting
from semantic_transmission.wire import pack, unpack


def packet(frames=5):
    indices = sorted(set(range(0, frames, 2)) | {frames - 1})
    rates = bytes([1] * 32)  # Padded 128 x 64, 32 sites at rate 4.
    header = dict(schema=model.POLICY, video=dict(frames=frames, height=16, width=16, fps=24),
        visual_codec=codec.SPEC, checkpoint_sha256=codec.CHECKPOINT_SHA256,
        complex_dtype='little_endian_complex64', temporal=base.temporal_identity(),
        keyframes=[dict(index=i, complex_offset=k * 64, complex_count=64,
            average_power=.5, rate_offset=k * 32, rate_count=32, rate_bytes=32)
            for k, i in enumerate(indices)])
    return pack(header, rates * len(indices))


def setup_sender(root, frames=5):
    dest = root / 'sender'; dest.mkdir(parents=True)
    original = packet(frames); header, _ = model.parse_packet(original)
    wire = compression.encode(original)
    (dest / 'metadata_wire.bin').write_bytes(wire)
    values = np.ones(64 * len(header['keyframes']), dtype='<c8')
    values.tofile(dest / 'visual.c64')
    return original, wire, values


def test_direct_visual_symbols_and_compressed_metadata_are_separate(tmp_path, monkeypatch):
    from semantic_transmission.qwen35_pairs_comparison import frame_noise
    original, wire, sent = setup_sender(tmp_path)
    calls = []
    def transmit(data, *args, **kwargs):
        calls.append(data)
        return data, dict(**accounting(len(data)), bit_errors=0)
    monkeypatch.setattr(metadata_channel, 'transmit', transmit)
    model.channel(dict(output=str(tmp_path), sender=str(tmp_path / 'sender')))
    assert calls == [wire]  # The visual symbols NEVER enter LDPC.
    assert (tmp_path / 'received/metadata.bin').read_bytes() == original
    report = base.read(tmp_path / 'received/channel.json')
    assert report['complex_channel_uses'] == len(sent) + accounting(len(wire))['complex_channel_uses']
    assert report['visual_ldpc'] is False
    header, _ = model.parse_packet(original)
    received = model.read_symbols(tmp_path / 'received/visual.c64', header)
    for k in header['keyframes']:
        a, n = k['complex_offset'], k['complex_count']
        np.testing.assert_array_equal(received[a:a+n], sent[a:a+n] + frame_noise(k['index'], n, 42, 10))
    assert not np.array_equal(sent, received)


@pytest.mark.parametrize('fault', ['schema', 'checkpoint', 'codec', 'temporal', 'count', 'offset',
    'rate_offset', 'rate_count', 'power', 'index', 'trailing_rates', 'alphabet', 'crc'])
def test_bad_received_metadata_fails_before_model_load(fault):
    header, rates = unpack(packet())
    if fault == 'schema': header['schema'] = 'bsu_predict_v1'
    if fault == 'checkpoint': header['checkpoint_sha256'] = 'wrong'
    if fault == 'codec': header['visual_codec']['eta'] = .1
    if fault == 'temporal': header['temporal']['stride'] = 4
    if fault == 'count': header['keyframes'][0]['complex_count'] += 1
    if fault == 'offset': header['keyframes'][1]['complex_offset'] = 0
    if fault == 'rate_offset': header['keyframes'][1]['rate_offset'] = 0
    if fault == 'rate_count': header['keyframes'][0]['rate_count'] = 31
    if fault == 'power': header['keyframes'][0]['average_power'] = 0
    if fault == 'index': header['keyframes'][1]['index'] = 4
    if fault == 'trailing_rates': rates += b'\x01'
    if fault == 'alphabet': rates = b'\x1a' + rates[1:]
    data = pack(header, rates)
    if fault == 'crc': data = data[:-1] + bytes([data[-1] ^ 1])
    with pytest.raises(ValueError): model.parse_packet(data)


@pytest.mark.parametrize('fault', ['short', 'extra_byte', 'nonfinite'])
def test_visual_length_and_nonfinite_symbols_rejected(tmp_path, fault):
    original, _, values = setup_sender(tmp_path)
    path = tmp_path / 'sender/visual.c64'
    if fault == 'short': values[:-1].tofile(path)
    if fault == 'extra_byte': path.write_bytes(path.read_bytes() + b'x')
    if fault == 'nonfinite':
        values[0] = np.nan; values.tofile(path)
    with pytest.raises(ValueError): model.read_symbols(path, model.parse_packet(original)[0])


def test_noisy_metadata_aborts_without_received_products(tmp_path, monkeypatch):
    setup_sender(tmp_path)
    monkeypatch.setattr(metadata_channel, 'transmit', lambda data, *a, **kw:
        (data, dict(bit_errors=1)))
    with pytest.raises(ValueError, match='integrity failure'):
        model.channel(dict(output=str(tmp_path), sender=str(tmp_path / 'sender')))
    assert not (tmp_path / 'received').exists()


def test_streaming_selection_matches_original_scene_cut_policy(tmp_path):
    frames = np.array([np.full((16, 16, 3), x, dtype=np.uint8) for x in [0, 1, 2, 240, 241, 242]])
    for i, frame in enumerate(frames): Image.fromarray(frame).save(tmp_path / f'{i}.png')
    source = base.inspect_frames(tmp_path, 24)
    assert model.select_source_indices(source) == temporal.select_indices(frames, 2)
    Image.fromarray(frames[0]).save(tmp_path / '3.png')
    with pytest.raises(ValueError, match='source frame changed'): model.select_source_indices(source)


def test_profile_and_status_need_no_models(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(model, 'preflight', lambda _: pytest.fail('unexpected preflight'))
    model.main(['profile'])
    value = json.loads(capsys.readouterr().out)
    assert value['visual_ldpc'] is False and value['metadata_ldpc'] is True
    assert value['temporal_latent_prediction'] is False
    model.main(['status', '--output', str(tmp_path / 'missing')])
    assert json.loads(capsys.readouterr().out)['status'] == 'NOT_STARTED'
    assert not (tmp_path / 'missing').exists()


def test_resume_and_received_only_jobs(tmp_path, monkeypatch):
    source = tmp_path / 'source'; source.mkdir()
    for i in range(5): Image.new('RGB', (16, 16), (i * 10,) * 3).save(source / f'{i}.png')
    root = tmp_path / 'run'; calls = []
    record = dict(output=str(root), signature='test-jscc', profile=model.profile(),
        source=base.inspect_frames(source, 24),
        runtimes=dict(codec=sys.executable, channel=sys.executable, receiver=sys.executable))
    def launch(stage, job, *_):
        calls.append(stage); job = base.read(job)
        if stage in ('decode', 'restore'):
            assert 'source' not in job and 'sender' not in job
        if stage == 'sender': setup_sender(root)
        elif stage == 'channel': model.channel(job)
        elif stage == 'decode':
            (root / 'decoded').mkdir()
            Image.new('RGB', (16, 16)).save(root / 'decoded/00000.png')
            write_json(root / 'decoded.json', {'test': 'decoder stub'})
        else:
            (root / 'reconstruction').mkdir()
            (root / 'reconstruction/reconstruction.mp4').write_bytes(b'test movie stub')
    monkeypatch.setattr(model, 'launch', launch)
    monkeypatch.setattr(metadata_channel, 'transmit', lambda data, *a, **kw:
        (data, dict(**accounting(len(data)), bit_errors=0)))
    model.execute(record); model.execute(record)
    assert calls == ['sender', 'channel', 'decode', 'restore']
    assert base.read(root / 'RESULT.json')['status'] == model.COMPLETE
    (root / 'received/visual.c64').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='artifacts changed'): model.execute(record)
    assert base.read(root / 'status.json')['status'] == 'FAILED'
    assert not (root / 'RESULT.json').exists()


def test_completed_sender_reused_after_interruption(tmp_path, monkeypatch):
    source = tmp_path / 'source'; source.mkdir()
    for i in range(2): Image.new('RGB', (16, 16)).save(source / f'{i}.png')
    root = tmp_path / 'run'
    record = dict(output=str(root), signature='interruption-jscc', profile=model.profile(),
        source=base.inspect_frames(source, 24), runtimes=dict(codec='stub', channel='stub', receiver='stub'))
    def launch(stage, *_):
        if stage == 'sender': setup_sender(root, 2)
        else: raise KeyboardInterrupt('simulated safe stop')
    monkeypatch.setattr(model, 'launch', launch)
    with pytest.raises(KeyboardInterrupt): model.execute(record)
    assert base.read(root / 'status.json')['status'] == 'INTERRUPTED'
    assert base.read(root / 'stages/sender.json')['status'] == 'PASSED'
    assert not (root / 'RESULT.json').exists()
