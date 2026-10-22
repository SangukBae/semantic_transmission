"""Paid DCVC scene streams and lossless SPARC metadata compression.

DCVC entropy-coded bytes are never modified by the metadata compressor. All
sample-dependent receiver inputs are inside this one LDPC-protected packet.
"""
import struct
import zlib

from .bsu_predict import validate_video
from . import dense_temporal_receiver as temporal
from .predictive_codec import WEIGHTS
from .sparc_selection import scene_groups, validate_anchors, validate_gap
from .wire import pack, unpack

SCHEMA = 'sparc_dcvc_scene_streams_v1'
CODEC = dict(schema='bsu_dcvc_rt_keys_v1', weights=WEIGHTS,
    color='full_range_ycbcr444_bt709', upstream='cbdae87a5445114cdc7f48816da63ea80bdeac40')
ENVELOPE = struct.Struct('!4sBBHIIII')
MAX_PACKET = 256 * 1024 * 1024
MAX_PREFIX = 1024 * 1024


def temporal_identity(max_gap):
    validate_gap(max_gap)
    return dict(policy=temporal.VARIABLE_POLICY if max_gap is None else temporal.POLICY,
                stride=max_gap, weights=temporal.WEIGHT_SHA,
                network=temporal.NETWORK_SHA, warp=temporal.WARP_SHA)


def parse(packet):
    if not isinstance(packet, bytes) or not 0 < len(packet) <= MAX_PACKET:
        raise ValueError('invalid SPARC packet size')
    header, payload = unpack(packet)
    fields = {'schema', 'video', 'codec', 'temporal', 'max_gap', 'cuts', 'chunks'}
    if (set(header) != fields or header['schema'] != SCHEMA or header['codec'] != CODEC
            or header['temporal'] != temporal_identity(header['max_gap'])):
        raise ValueError('unsupported SPARC packet/model identity')
    validate_video(header['video'])
    if not isinstance(header['chunks'], list) or not header['chunks']:
        raise ValueError('missing DCVC scene chunks')
    offset = 0; indices = []; groups = []
    for chunk in header['chunks']:
        if (not isinstance(chunk, dict) or set(chunk) != {'indices', 'offset', 'bytes'}
                or type(chunk['offset']) is not int or chunk['offset'] != offset
                or type(chunk['bytes']) is not int or chunk['bytes'] <= 0
                or not isinstance(chunk['indices'], list) or not chunk['indices']):
            raise ValueError('invalid DCVC scene descriptor')
        groups.append(chunk['indices']); indices.extend(chunk['indices']); offset += chunk['bytes']
    if offset != len(payload):
        raise ValueError('DCVC payload length mismatch')
    validate_anchors(indices, header['video']['frames'], header['max_gap'], header['cuts'])
    if groups != scene_groups(indices, header['cuts']):
        raise ValueError('DCVC reset points disagree with scene cuts')
    if len(packet) - len(payload) > MAX_PREFIX:
        raise ValueError('SPARC metadata exceeds the registered bound')
    return header, payload


def make(video, coded_groups, cuts, max_gap=4):
    chunks = []; payload = bytearray()
    for coded in coded_groups:
        header, stream = unpack(coded)
        expected = dict(CODEC, indices=header.get('indices'), width=video['width'], height=video['height'])
        if header != expected or not stream:
            raise ValueError('DCVC scene identity or dimensions changed')
        chunks.append(dict(indices=header['indices'], offset=len(payload), bytes=len(stream)))
        payload.extend(stream)
    packet = pack(dict(schema=SCHEMA, video=video, codec=CODEC,
        temporal=temporal_identity(max_gap), max_gap=max_gap, cuts=cuts, chunks=chunks), payload)
    parse(packet)
    return packet


def decoder_packets(packet):
    header, payload = parse(packet)
    for chunk in header['chunks']:
        codec_header = dict(CODEC, indices=chunk['indices'],
                            height=header['video']['height'], width=header['video']['width'])
        yield pack(codec_header, payload[chunk['offset']:chunk['offset'] + chunk['bytes']])


def compress(packet):
    _, payload = parse(packet)
    prefix = packet[:-len(payload)]
    body = zlib.compress(prefix, 9)
    encoded = ENVELOPE.pack(b'SPRC', 1, 1, 0, len(packet), len(prefix), len(body),
                            zlib.crc32(packet)) + body + payload
    return encoded if len(encoded) < len(packet) else packet


def decompress(packet):
    if not isinstance(packet, bytes) or not 0 < len(packet) <= MAX_PACKET:
        raise ValueError('invalid SPARC envelope size')
    if packet[:4] == b'LGV1':
        parse(packet); return packet
    if len(packet) < ENVELOPE.size:
        raise ValueError('truncated SPARC envelope')
    magic, version, method, reserved, size, prefix_size, body_size, crc = ENVELOPE.unpack(packet[:ENVELOPE.size])
    if (magic != b'SPRC' or version != 1 or method != 1 or reserved or
            not 16 < prefix_size <= MAX_PREFIX or not prefix_size < size <= MAX_PACKET or
            body_size < 1 or len(packet) != ENVELOPE.size + body_size + size - prefix_size):
        raise ValueError('invalid SPARC envelope header')
    inflater = zlib.decompressobj()
    try:
        prefix = inflater.decompress(packet[ENVELOPE.size:ENVELOPE.size + body_size], prefix_size + 1)
    except zlib.error as error:
        raise ValueError('invalid compressed SPARC metadata') from error
    if (len(prefix) != prefix_size or not inflater.eof or inflater.unused_data or inflater.unconsumed_tail):
        raise ValueError('invalid SPARC metadata boundary')
    original = prefix + packet[ENVELOPE.size + body_size:]
    if zlib.crc32(original) != crc:
        raise ValueError('SPARC checksum mismatch')
    _, payload = parse(original)
    if len(original) - len(payload) != prefix_size:
        raise ValueError('SPARC prefix length disagrees with framing')
    return original
