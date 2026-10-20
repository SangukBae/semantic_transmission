"""Lossless predictive-packet header compression; DCVC bytes stay verbatim.

Both nested LGV1 headers (including their original JSON and checksums) form a
contiguous prefix. Only that prefix is compressed. RX restores the entire
original packet before the existing predictive parser or neural decoder runs.
"""
import struct
import zlib

from .wire import HEADER as INNER_HEADER, unpack

POLICY = 'bsu_predict_lossless_headers_v1'
HEADER = struct.Struct('!4sBBHIIII')
MAGIC = b'BSPC'
MAX_PACKET = 256 * 1024 * 1024
MAX_PREFIX = 1024 * 1024


def split(packet):
    from .bsu_predict import unpack_packet
    if not isinstance(packet, bytes) or not INNER_HEADER.size < len(packet) <= MAX_PACKET:
        raise ValueError('invalid predictive packet size/type')
    _, coded = unpack_packet(packet)
    _, stream = unpack(coded)
    prefix = len(packet) - len(stream)
    if not stream or not 2 * INNER_HEADER.size < prefix <= MAX_PREFIX:
        raise ValueError('invalid predictive metadata prefix')
    return packet[:prefix], stream


def encode(packet, mode='auto'):
    prefix, stream = split(packet)
    if mode == 'raw':
        return packet
    if mode != 'auto':
        raise ValueError('compression mode must be raw or auto')
    body = zlib.compress(prefix, 9)
    compressed = HEADER.pack(MAGIC, 1, 1, 0, len(packet), len(prefix),
                             len(body), zlib.crc32(packet)) + body + stream
    # Include the envelope overhead; never increase on-wire bytes or LDPC cost.
    return compressed if len(compressed) < len(packet) else packet


def decode(packet):
    if not isinstance(packet, bytes) or not 0 < len(packet) <= MAX_PACKET:
        raise ValueError('invalid compressed packet size/type')
    if packet[:4] == b'LGV1':
        split(packet)
        return packet
    if len(packet) < HEADER.size:
        raise ValueError('truncated predictive compression envelope')
    magic, version, method, reserved, size, prefix_size, body_size, crc = HEADER.unpack(packet[:HEADER.size])
    if (magic != MAGIC or version != 1 or method != 1 or reserved or
            not 2 * INNER_HEADER.size < prefix_size <= MAX_PREFIX or
            not prefix_size < size <= MAX_PACKET or body_size < 1 or
            len(packet) != HEADER.size + body_size + size - prefix_size):
        raise ValueError('invalid predictive compression envelope')
    inflater = zlib.decompressobj()
    try:
        prefix = inflater.decompress(packet[HEADER.size:HEADER.size + body_size], prefix_size + 1)
    except zlib.error as error:
        raise ValueError('invalid compressed predictive headers') from error
    if (len(prefix) != prefix_size or not inflater.eof or inflater.unused_data or
            inflater.unconsumed_tail):
        raise ValueError('invalid predictive header length or trailing stream')
    original = prefix + packet[HEADER.size + body_size:]
    if zlib.crc32(original) != crc:
        raise ValueError('restored predictive packet checksum mismatch')
    expected_prefix, _ = split(original)
    if len(expected_prefix) != prefix_size:
        raise ValueError('predictive prefix boundary mismatch')
    return original


def method_name(packet):
    return 'raw' if packet[:4] == b'LGV1' else 'zlib_nested_headers'
