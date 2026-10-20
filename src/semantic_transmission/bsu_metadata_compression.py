"""Opt-in lossless transport envelope for existing NTSCC+ metadata packets.

The receiver restores the exact original LGV1 bytes BEFORE normal parsing,
trajectory hash checks or neural decoding. This module changes no default.
"""
import struct
import zlib

from .wire import HEADER as INNER_HEADER, unpack

HEADER = struct.Struct('!4sBBHII')
MAGIC = b'BSMC'
VERSION = 1
MAX_PACKET = 16 * 1024 * 1024
METHODS = {'raw': 0, 'pack5': 1, 'zlib': 2, 'pack5_zlib': 3}


def validate_original(packet):
    if not isinstance(packet, bytes) or not INNER_HEADER.size < len(packet) <= MAX_PACKET:
        raise ValueError('invalid original packet size/type')
    header, rates = unpack(packet)
    codec = header.get('visual_codec', {})
    if codec.get('name') != 'NTSCC+' or codec.get('rate_index_encoding') != 'uint8':
        raise ValueError('compression requires NTSCC+ uint8 rate indices')
    keys = header.get('keyframes')
    if not isinstance(keys, list) or not keys:
        raise ValueError('missing keyframe descriptors')
    offset = 0
    for key in keys:
        if not isinstance(key, dict):
            raise ValueError('invalid keyframe descriptor')
        count = key.get('rate_count')
        if (type(count) is not int or count < 1 or type(key.get('rate_bytes')) is not int
                or type(key.get('rate_offset')) is not int or key['rate_bytes'] != count
                or key['rate_offset'] != offset):
            raise ValueError('noncontiguous or invalid uint8 rate descriptors')
        offset += count
    if offset != len(rates) or any(v >= 26 for v in rates):
        raise ValueError('invalid NTSCC+ rate payload')
    return header, rates


def pack5(rates):
    if not isinstance(rates, bytes) or not rates or any(v >= 26 for v in rates):
        raise ValueError('rates must be nonempty bytes in [0,25]')
    output = bytearray()
    acc = bits = 0
    for value in rates:
        acc = (acc << 5) | value
        bits += 5
        if bits >= 8:
            bits -= 8
            output.append(acc >> bits)
            acc &= (1 << bits) - 1
    if bits:
        output.append(acc << (8-bits))
    return bytes(output)


def unpack5(data, count):
    if type(count) is not int or not 1 <= count <= MAX_PACKET or len(data) != (count*5+7)//8:
        raise ValueError('invalid packed rate length')
    output = bytearray()
    acc = bits = 0
    for value in data:
        acc = (acc << 8) | value
        bits += 8
        while bits >= 5 and len(output) < count:
            bits -= 5
            rate = acc >> bits
            if rate >= 26:
                raise ValueError('out-of-alphabet packed rate')
            output.append(rate)
            acc &= (1 << bits)-1
    if len(output) != count or acc:
        raise ValueError('invalid packed rate padding')
    return bytes(output)


def _packed(packet, rates):
    # Keep the original framing, JSON bytes, CRC and offsets verbatim.
    return packet[:-len(rates)] + pack5(rates)


def _unpacked(data, original_size):
    if len(data) < INNER_HEADER.size:
        raise ValueError('truncated packed metadata')
    magic, metadata_size, count, _ = INNER_HEADER.unpack(data[:INNER_HEADER.size])
    prefix = INNER_HEADER.size + metadata_size
    if magic != b'LGV1' or prefix+count != original_size or prefix >= len(data):
        raise ValueError('invalid packed metadata lengths')
    return data[:prefix] + unpack5(data[prefix:], count)


def _inflate(data, limit):
    decoder = zlib.decompressobj()
    try:
        output = decoder.decompress(data, limit+1)
    except zlib.error as error:
        raise ValueError('invalid compressed stream') from error
    if (len(output) > limit or not decoder.eof or decoder.unused_data
            or decoder.unconsumed_tail):
        raise ValueError('oversized, incomplete or trailing compressed stream')
    return output


def variants(packet):
    _, rates = validate_original(packet)
    packed = _packed(packet, rates)
    contents = {'pack5': packed, 'zlib': zlib.compress(packet, 9),
                'pack5_zlib': zlib.compress(packed, 9)}
    result = {'raw': packet}
    for method, body in contents.items():
        result[method] = HEADER.pack(MAGIC, VERSION, METHODS[method], 0,
                                     len(packet), zlib.crc32(packet)) + body
    return result


def encode(packet, method='auto'):
    choices = variants(packet)
    if method == 'auto':
        # Smallest on-wire size also minimizes fixed LDPC block accounting.
        # Including raw ensures incompressible packets cannot grow.
        return min(choices.values(), key=len)
    if method not in choices:
        raise ValueError('unknown compression method')
    return choices[method]


def decode(packet):
    if not isinstance(packet, bytes) or len(packet) > MAX_PACKET+2*HEADER.size:
        raise ValueError('invalid envelope size/type')
    if packet[:4] == b'LGV1':
        validate_original(packet)
        return packet
    if len(packet) < HEADER.size:
        raise ValueError('truncated compression header')
    magic, version, method, reserved, size, crc = HEADER.unpack(packet[:HEADER.size])
    if (magic != MAGIC or version != VERSION or reserved or method not in (1, 2, 3)
            or not INNER_HEADER.size < size <= MAX_PACKET):
        raise ValueError('invalid compression header')
    body = packet[HEADER.size:]
    if method in (2, 3):
        body = _inflate(body, size)
    if method in (1, 3):
        body = _unpacked(body, size)
    if len(body) != size or zlib.crc32(body) != crc:
        raise ValueError('restored packet checksum/length mismatch')
    validate_original(body)
    return body


def method_name(packet):
    if packet[:4] == b'LGV1':
        return 'raw'
    if len(packet) < HEADER.size:
        raise ValueError('truncated compression header')
    number = HEADER.unpack(packet[:HEADER.size])[2]
    return next((name for name, value in METHODS.items() if value == number), 'invalid')
