import pytest

from semantic_transmission import sparc_transport as t
from semantic_transmission.packets import accounting
from semantic_transmission.wire import pack, unpack


def example():
    video = dict(frames=9, height=16, width=16, fps=24)
    packets = [pack(dict(t.CODEC, indices=indices, height=16, width=16), bytes(range(128)) * 3)
               for indices in ([0, 3, 4], [5, 8])]
    return t.make(video, packets, [5]), packets


def test_scene_streams_are_unchanged_after_compression_and_channel_framing():
    original, packets = example()
    wire = t.compress(original)
    assert len(wire) < len(original)
    assert accounting(len(wire))['complex_channel_uses'] <= accounting(len(original))['complex_channel_uses']
    assert t.decompress(wire) == original
    assert list(t.decoder_packets(t.decompress(wire))) == packets


@pytest.mark.parametrize('fault', ['codec', 'temporal', 'offset', 'length', 'cut', 'gap', 'duplicate', 'empty', 'trailing'])
def test_invalid_packet_rejected_before_neural_decode(fault):
    original, _ = example(); header, payload = unpack(original)
    if fault == 'codec': header['codec']['weights']['inter'] = 'wrong'
    if fault == 'temporal': header['temporal']['weights'] = 'wrong'
    if fault == 'offset': header['chunks'][1]['offset'] = 0
    if fault == 'length': header['chunks'][0]['bytes'] += 1
    if fault == 'cut': header['cuts'] = []
    if fault == 'gap': header['chunks'][0]['indices'] = [0]
    if fault == 'duplicate': header['chunks'][0]['indices'] = [0, 3, 3, 4]
    if fault == 'empty': header['chunks'] = []
    if fault == 'trailing': payload += b'junk'
    with pytest.raises(ValueError): t.parse(pack(header, payload))


@pytest.mark.parametrize('fault', ['crc', 'truncated', 'trailing', 'size', 'method', 'prefix'])
def test_bad_compression_envelopes_fail_closed(fault):
    original, _ = example(); wire = t.compress(original)
    fields = list(t.ENVELOPE.unpack(wire[:t.ENVELOPE.size]))
    if fault == 'crc': wire = wire[:-1] + bytes([wire[-1] ^ 1])
    if fault == 'truncated': wire = wire[:-1]
    if fault == 'trailing': wire += b'x'
    if fault in ('size', 'method', 'prefix'):
        fields[{'size': 4, 'method': 2, 'prefix': 5}[fault]] += 1
        wire = t.ENVELOPE.pack(*fields) + wire[t.ENVELOPE.size:]
    with pytest.raises(ValueError): t.decompress(wire)


def test_entropy_stream_is_not_recompressed():
    original, _ = example(); _, payload = unpack(original)
    wire = t.compress(original)
    assert wire.endswith(payload)
    assert t.decompress(original) == original
