import json
import random
import struct
import zlib

import pytest

from semantic_transmission import bsu_metadata_compression as codec
from semantic_transmission.wire import pack, unpack, HEADER
from semantic_transmission.packets import accounting


def example(count=864):
    rates = bytes(i % 26 for i in range(count))
    header = dict(visual_codec=dict(name='NTSCC+', rate_index_encoding='uint8'),
        keyframes=[dict(rate_count=count, rate_bytes=count, rate_offset=0)],
        segments=[dict(text='공을 잡고 다시 놓는다. '*8)], video=dict(frames=37))
    return pack(header, rates)


@pytest.mark.parametrize('count', [1,2,3,7,8,9,25,26,863,864,865,1728])
def test_all_alphabet_values_and_byte_boundaries(count):
    rng = random.Random(count)
    values = bytes(rng.randrange(26) for _ in range(count))
    encoded = codec.pack5(values)
    assert len(encoded) == (count*5+7)//8
    assert codec.unpack5(encoded, count) == values


@pytest.mark.parametrize('method', ['raw','pack5','zlib','pack5_zlib','auto'])
def test_exact_original_bytes_including_noncanonical_json_and_unicode(method):
    header, payload = unpack(example(865))
    metadata = json.dumps(header, ensure_ascii=False, indent=3).encode()
    body = metadata+payload
    original = HEADER.pack(b'LGV1', len(metadata),len(payload),zlib.crc32(body))+body
    encoded = codec.encode(original, method)
    assert codec.decode(encoded) == original
    assert codec.decode(codec.decode(encoded)) == original


def test_auto_counts_wrapper_and_cannot_increase_transmitted_blocks():
    for count in (1,7,864,10000):
        original = example(count)
        choices = codec.variants(original)
        selected = codec.encode(original)
        assert len(selected) == min(map(len, choices.values()))
        assert accounting(len(selected))['complex_channel_uses'] <= accounting(len(original))['complex_channel_uses']


@pytest.mark.parametrize('method', ['raw','pack5','zlib','pack5_zlib'])
def test_every_single_byte_corruption_truncation_and_trailing_data_rejected(method):
    encoded = codec.encode(example(26),method)
    for index in range(len(encoded)):
        damaged = bytearray(encoded); damaged[index] ^= 1
        with pytest.raises(ValueError):
            codec.decode(bytes(damaged))
    for damaged in (encoded[:-1], encoded[:5], encoded+b'\0'):
        with pytest.raises(ValueError):
            codec.decode(damaged)


def test_invalid_rate_range_padding_and_descriptors_rejected():
    for rates in (b'', bytes([26]), bytes([255])):
        with pytest.raises(ValueError): codec.pack5(rates)
    for data,count in ((b'\x01',1),(b'\xd0',1),(b'\x00',0),(b'\x00',True),(b'\x00',2)):
        with pytest.raises(ValueError): codec.unpack5(data,count)
    header, rates = unpack(example())
    for field,value in [('rate_offset',1),('rate_count',-1),('rate_count',True),('rate_bytes',863)]:
        bad=json.loads(json.dumps(header));bad['keyframes'][0][field]=value
        with pytest.raises(ValueError): codec.encode(pack(bad,rates))
    with pytest.raises(ValueError): codec.encode(pack(header,rates[:-1]))
    with pytest.raises(ValueError): codec.encode(pack(header,rates[:-1]+bytes([26])))
    header['visual_codec']['name']='NTSCC'
    with pytest.raises(ValueError): codec.encode(pack(header,rates))


def test_oversized_and_concatenated_streams_rejected_before_inner_decode():
    original = example()
    encoded = codec.encode(original,'zlib')
    fields=list(codec.HEADER.unpack(encoded[:codec.HEADER.size]))
    fields[-2]=codec.MAX_PACKET+1
    with pytest.raises(ValueError): codec.decode(codec.HEADER.pack(*fields)+encoded[codec.HEADER.size:])
    fields[-2]=len(original)
    with pytest.raises(ValueError): codec.decode(codec.HEADER.pack(*fields)+zlib.compress(b'x'*(len(original)+100)))
    with pytest.raises(ValueError): codec.decode(encoded+zlib.compress(b'trailing'))
