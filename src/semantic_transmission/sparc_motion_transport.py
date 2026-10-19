"""Experimental long-gap key stream and separately paid sparse motion packet.

The v1 SPARC short-gap contract is deliberately not relaxed. This new schema
registers a separate, untrained long-gap RIFE experiment, capped at 64 frames.
"""
import hashlib
import json
import struct
import zlib

from .bsu_predict import validate_video
from .qwen35_semantic_selection import validate_keys
from .sparc_transport import CODEC
from . import dense_temporal_receiver as rife
from . import wire

POLICY = 'sparc_sparse_motion_pilot_v1'
HEADER = struct.Struct('!4sIII')
MAX_BYTES = 256 * 1024 * 1024
MAX_METADATA = 1024 * 1024
RECEIVER = dict(policy='rife425_long_gap_64_experimental_v1', weights=rife.WEIGHT_SHA,
                network=rife.NETWORK_SHA, warp=rife.WARP_SHA)


def validate_timeline(video, indices):
    validate_video(video)
    validate_keys(indices, video['frames'])
    if max(b-a for a,b in zip(indices, indices[1:])) > 64:
        raise ValueError('experimental RIFE gap exceeds 64 frames')


def pack(header, payload=b''):
    raw = json.dumps(header, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(raw) > MAX_METADATA: raise ValueError('motion metadata too large')
    metadata = zlib.compress(raw, 9)
    body = metadata + payload
    result = HEADER.pack(b'SMP1', len(metadata), len(payload), zlib.crc32(body)) + body
    unpack(result)
    return result


def unpack(packet):
    if not isinstance(packet, bytes) or not HEADER.size < len(packet) <= MAX_BYTES:
        raise ValueError('invalid motion packet length')
    magic, metadata_size, payload_size, checksum = HEADER.unpack(packet[:HEADER.size])
    body = packet[HEADER.size:]
    if (magic != b'SMP1' or metadata_size < 1 or metadata_size + payload_size != len(body)
            or zlib.crc32(body) != checksum):
        raise ValueError('invalid motion packet framing/checksum')
    inflater = zlib.decompressobj()
    try:
        raw = inflater.decompress(body[:metadata_size], MAX_METADATA + 1)
    except zlib.error as error:
        raise ValueError('invalid compressed motion metadata') from error
    if (len(raw) > MAX_METADATA or not inflater.eof or inflater.unused_data or inflater.unconsumed_tail):
        raise ValueError('invalid motion metadata boundary')
    header = json.loads(raw)
    if not isinstance(header, dict) or header.get('policy') != POLICY or header.get('kind') not in ('keys', 'motion'):
        raise ValueError('unsupported motion packet policy')
    return header, body[metadata_size:]


def make_keys(video, coded):
    packet = pack(dict(policy=POLICY, kind='keys', video=video, receiver=RECEIVER), coded)
    parse_keys(packet)
    return packet


def parse_keys(packet):
    header, coded = unpack(packet)
    if set(header) != {'policy', 'kind', 'video', 'receiver'} or header['kind'] != 'keys' or header['receiver'] != RECEIVER:
        raise ValueError('invalid key packet identity')
    codec, payload = wire.unpack(coded)
    video = header['video']
    if codec != dict(CODEC, indices=codec.get('indices'), height=video.get('height'), width=video.get('width')) or not payload:
        raise ValueError('invalid DCVC key stream')
    validate_timeline(video, codec['indices'])
    return header, coded, codec['indices']


def make_motion(keys, windows):
    header, _, _ = parse_keys(keys)
    packet = pack(dict(policy=POLICY, kind='motion', video=header['video'],
                       key_packet_sha256=hashlib.sha256(keys).hexdigest(), windows=windows))
    parse_motion(packet, keys)
    return packet


def parse_motion(packet, keys):
    from .sparse_trajectory import decode
    key_header, _, indices = parse_keys(keys)
    header, payload = unpack(packet)
    if (set(header) != {'policy', 'kind', 'video', 'key_packet_sha256', 'windows'} or
            header['kind'] != 'motion' or payload or header['video'] != key_header['video'] or
            header['key_packet_sha256'] != hashlib.sha256(keys).hexdigest() or not isinstance(header['windows'], list)):
        raise ValueError('motion does not match received keys')
    intervals = set(zip(indices, indices[1:])); decoded = {}
    for row in header['windows']:
        if not isinstance(row, dict) or set(row) != {'start', 'end', 'trajectory'}:
            raise ValueError('invalid motion window')
        a, b = row['start'], row['end']
        if type(a) is not int or type(b) is not int or (a,b) not in intervals or (a,b) in decoded:
            raise ValueError('invalid or duplicate motion interval')
        decoded[a,b] = decode(row['trajectory'], b-a+1)
    return decoded
