from pathlib import Path
import random
import zlib

import pytest

from semantic_transmission import bsu_predict as model
from semantic_transmission import bsu_predict_compression as compression
from semantic_transmission import bsu_predict_compressed as runner
from semantic_transmission.artifacts import sha256
from semantic_transmission.packets import accounting
from semantic_transmission.wire import pack, unpack
from test_bsu_predict import codec_packet, fake_models, record


def real_shape_packet(payload_bytes=20000):
    codec, _ = unpack(codec_packet(list(range(0, 239, 2)) + [239], width=576, height=320))
    payload = random.Random(42).randbytes(payload_bytes)
    return model.make_packet(dict(frames=240, width=576, height=320, fps=24), pack(codec, payload))


@pytest.mark.parametrize('payload_bytes,saved_blocks', [(19800, 1), (20000, 0)])
def test_exact_headers_and_entropy_stream_with_actual_wire_accounting(payload_bytes, saved_blocks):
    original = real_shape_packet(payload_bytes); wire = compression.encode(original)
    assert len(wire) < len(original)
    assert compression.decode(wire) == original
    _, stream = compression.split(original)
    assert wire.endswith(stream)  # No recompression or modification of DCVC bits.
    assert accounting(len(original))['blocks'] - accounting(len(wire))['blocks'] == saved_blocks
    assert accounting(len(original))['complex_channel_uses'] - accounting(len(wire))['complex_channel_uses'] == saved_blocks * 2304
    assert compression.encode(original, 'raw') == original
    assert compression.decode(original) == original


@pytest.mark.parametrize('mutation', ['crc', 'tail', 'truncated', 'version', 'size', 'prefix', 'zip_tail'])
def test_bad_envelopes_rejected_before_neural_decoder(mutation):
    wire = compression.encode(real_shape_packet()); fields = list(compression.HEADER.unpack(wire[:compression.HEADER.size]))
    if mutation == 'crc': wire = wire[:-1] + bytes([wire[-1] ^ 1])
    if mutation == 'tail': wire += b'extra'
    if mutation == 'truncated': wire = wire[:10]
    if mutation == 'version': fields[1] = 2
    if mutation == 'size': fields[4] = compression.MAX_PACKET + 1
    if mutation == 'prefix': fields[5] = 33
    if mutation in ('version', 'size', 'prefix'):
        wire = compression.HEADER.pack(*fields) + wire[compression.HEADER.size:]
    if mutation == 'zip_tail':
        boundary = compression.HEADER.size + fields[6]
        wire = wire[:boundary] + b'tail' + wire[boundary:]
        fields[6] += 4
        wire = compression.HEADER.pack(*fields) + wire[compression.HEADER.size:]
    with pytest.raises(ValueError): compression.decode(wire)


def test_raw_fallback_includes_envelope_overhead(monkeypatch):
    original = real_shape_packet(); compress = zlib.compress
    monkeypatch.setattr(compression.zlib, 'compress', lambda value, _: compress(value, 0))
    assert compression.encode(original) == original


def test_received_only_pipeline_matches_raw_and_resumes(tmp_path, monkeypatch, fake_models):
    results = []
    for mode in ('raw', 'auto'):
        directory = tmp_path / mode; directory.mkdir()
        request = record(directory); root = Path(request['output']); calls = []
        def launch(stage, job, *_):
            calls.append(stage)
            value = model.read(job)
            if stage in ('decode', 'restore'): assert 'source' not in value
            runner.worker(value, stage, mode)
        monkeypatch.setattr(model, 'launch', launch)
        model.execute(request)
        assert calls == ['sender', 'channel', 'decode', 'restore']
        assert (root / 'sender/packet_original.bin').read_bytes() == (root / 'received/packet.bin').read_bytes()
        wire = root / 'sender/packet.bin'; channel = model.read(root / 'received/channel.json')
        assert channel['complex_channel_uses'] == accounting(wire.stat().st_size)['complex_channel_uses']
        model.execute(request); assert len(calls) == 4
        results.append(root)
    a, b = results
    assert (a / 'sender/packet_original.bin').read_bytes() == (b / 'sender/packet_original.bin').read_bytes()
    for p in (a / 'reconstruction/frames').glob('*.png'):
        assert sha256(p) == sha256(b / 'reconstruction/frames' / p.name)
    assert sha256(a / 'reconstruction/reconstruction.mp4') == sha256(b / 'reconstruction/reconstruction.mp4')


def test_failed_wire_channel_does_not_write_decoder_packet(tmp_path, monkeypatch, fake_models):
    from semantic_transmission import metadata_channel
    request = record(tmp_path); root = Path(request['output']); root.mkdir()
    runner.worker(dict(output=str(root), source=request['source']), 'sender', 'auto')
    monkeypatch.setattr(metadata_channel, 'transmit', lambda *_a, **_k: (b'bad', dict(bit_errors=1)))
    with pytest.raises(ValueError, match='channel integrity'):
        runner.worker(dict(output=str(root), packet=str(root / 'sender/packet.bin')), 'channel', 'auto')
    assert not (root / 'received/packet.bin').exists()


def test_profile_dispatch_is_scoped_and_registered():
    original = model.profile(); module = model.MODULE
    with runner.execution_profile('auto'):
        assert model.profile()['metadata_transport']['mode'] == 'auto'
        assert model.MODULE == runner.MODULE
        assert 'bsu_predict_compression' in model.CODE
    assert model.profile() == original and model.MODULE == module
