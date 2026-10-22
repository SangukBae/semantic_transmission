"""Experimental UF packets: paid timestamps, model identity and chunk padding."""
import struct
import zlib

from .bsu_predict import validate_video
from .sparc_selection import validate_anchors
from .sparc_transport import temporal_identity
from .wire import pack, unpack

UPSTREAM = 'cbdae87a5445114cdc7f48816da63ea80bdeac40'
SCHEMA = 'sparc_uf_hts_pilot_v1'
LIMIT = 256 * 1024 * 1024
PREFIX_LIMIT = 1024 * 1024
ENVELOPE = struct.Struct('!4sIII')


def groups(indices, keys, mode):
    if (not indices or indices != sorted(set(indices)) or not keys or
            keys != sorted(set(keys)) or not set(keys) <= set(indices) or
            keys[0] != indices[0] or keys[-1] != indices[-1]):
        raise ValueError('invalid frozen selection')
    if mode == 'scene':
        return [indices]
    if mode != 'interval':
        raise ValueError('unknown grouping')
    # Every anchor is sent once. The right endpoint starts the next interval.
    return [[i for i in indices if a <= i < b]
            for a, b in zip(keys, keys[1:])] + [[keys[-1]]]


def padding(count):
    if type(count) is not int or count < 1:
        raise ValueError('empty group')
    return (-(count - 1)) % 8


def make(video, keys, mode, coded, weights, qp):
    chunks, payload = [], bytearray()
    for indices, stream in coded:
        chunks.append(dict(indices=indices, offset=len(payload), bytes=len(stream),
                           padded_frames=padding(len(indices))))
        payload.extend(stream)
    header = dict(schema=SCHEMA, upstream=UPSTREAM, weights=weights, model='hts',
                  video=video, keys=keys, mode=mode, qp=qp, max_gap=4, cuts=[],
                  temporal=temporal_identity(4), chunks=chunks)
    original = pack(header, payload)
    prefix = original[:-len(payload)]
    compressed = zlib.compress(prefix, 9)
    packet = ENVELOPE.pack(b'SPUF', len(original), len(prefix), len(compressed)) + compressed + payload
    packet = packet if len(packet) < len(original) else original
    parse(packet, weights)
    return packet


def parse(packet, weights):
    if not isinstance(packet, bytes) or not 16 < len(packet) <= LIMIT:
        raise ValueError('invalid UF packet size')
    if packet[:4] == b'SPUF':
        _, total, prefix_size, compressed_size = ENVELOPE.unpack(packet[:ENVELOPE.size])
        if (not 16 < prefix_size <= PREFIX_LIMIT or not prefix_size < total <= LIMIT or
                compressed_size < 1 or
                len(packet) != ENVELOPE.size + compressed_size + total - prefix_size):
            raise ValueError('invalid UF envelope')
        inflater = zlib.decompressobj()
        try:
            prefix = inflater.decompress(packet[16:16+compressed_size], prefix_size+1)
        except zlib.error as error:
            raise ValueError('invalid compressed metadata') from error
        if (len(prefix) != prefix_size or not inflater.eof or inflater.unused_data or
                inflater.unconsumed_tail):
            raise ValueError('invalid UF metadata boundary')
        original = prefix + packet[16+compressed_size:]
    elif packet[:4] == b'LGV1':
        original = packet
        prefix_size = None
    else:
        raise ValueError('unknown UF envelope')
    header, payload = unpack(original)
    expected = {'schema', 'upstream', 'weights', 'model', 'video', 'keys', 'mode',
                'qp', 'max_gap', 'cuts', 'temporal', 'chunks'}
    if (set(header) != expected or header['schema'] != SCHEMA or
            header['upstream'] != UPSTREAM or header['weights'] != weights or
            header['model'] != 'hts' or header['max_gap'] != 4 or header['cuts'] != [] or
            header['temporal'] != temporal_identity(4) or
            type(header['qp']) is not int or not 0 <= header['qp'] <= 63):
        raise ValueError('unsupported UF model or protocol')
    validate_video(header['video'])
    if not isinstance(header['chunks'], list) or not header['chunks']:
        raise ValueError('missing UF chunks')
    indices, offset, actual = [], 0, []
    for chunk in header['chunks']:
        if (set(chunk) != {'indices', 'offset', 'bytes', 'padded_frames'} or
                type(chunk['offset']) is not int or chunk['offset'] != offset or
                type(chunk['bytes']) is not int or chunk['bytes'] < 1 or
                not isinstance(chunk['indices'], list) or not chunk['indices'] or
                chunk['padded_frames'] != padding(len(chunk['indices']))):
            raise ValueError('invalid UF chunk descriptor')
        indices.extend(chunk['indices'])
        actual.append(chunk['indices'])
        offset += chunk['bytes']
    validate_anchors(indices, header['video']['frames'], 4, [])
    if actual != groups(indices, header['keys'], header['mode']) or offset != len(payload):
        raise ValueError('UF grouping or payload mismatch')
    if (len(original)-len(payload) > PREFIX_LIMIT or
            prefix_size is not None and len(original)-len(payload) != prefix_size):
        raise ValueError('UF prefix boundary mismatch')
    return header, payload
