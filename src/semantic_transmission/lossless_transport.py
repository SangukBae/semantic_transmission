"""Opt-in lossless coding around existing LGV1 digital packets.

Visual JSCC symbols are never compressed or counted as digital file bytes here.
The receiver restores the original packets before any model or trajectory code
runs. Compression decisions use sender bytes and the actual padded LDPC cost.
"""
import lzma
import struct
import zlib

from .packets import accounting
from .wire import unpack

ENVELOPE = struct.Struct('!4sB3xIII')
BUNDLE = struct.Struct('!4sII')
MAX_BYTES = 16 * 1024 * 1024
METHODS = {'raw': 0, 'zlib': 1, 'xz': 2}


def _encode(raw, method):
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError('invalid lossless payload size')
    if method == 'zlib':
        coded = zlib.compress(raw, 9)
    elif method == 'xz':
        coded = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=6)
    elif method == 'raw':
        return raw
    else:
        raise ValueError('unsupported lossless method')
    return ENVELOPE.pack(b'BSL1', METHODS[method], len(raw), len(coded), zlib.crc32(raw)) + coded


def _decode(packet):
    if len(packet) > MAX_BYTES + ENVELOPE.size:
        raise ValueError('lossless packet too large')
    if not packet.startswith(b'BSL1'):
        return packet
    if len(packet) < ENVELOPE.size:
        raise ValueError('truncated lossless envelope')
    _, method, size, coded_size, crc = ENVELOPE.unpack(packet[:ENVELOPE.size])
    if packet[5:8] != b'\0\0\0' or not 0 < size <= MAX_BYTES:
        raise ValueError('invalid lossless envelope')
    body = packet[ENVELOPE.size:]
    if len(body) != coded_size:
        raise ValueError('lossless body size mismatch')
    try:
        if method == METHODS['zlib']:
            decoder = zlib.decompressobj()
            raw = decoder.decompress(body, size + 1)
            tail = decoder.unconsumed_tail
        elif method == METHODS['xz']:
            decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=128 * 1024 * 1024)
            raw = decoder.decompress(body, max_length=size + 1)
            tail = b''
        else:
            raise ValueError('unsupported lossless method')
    except (zlib.error, lzma.LZMAError) as error:
        raise ValueError('invalid compressed stream') from error
    if (len(raw) != size or not decoder.eof or decoder.unused_data or tail
            or zlib.crc32(raw) != crc):
        raise ValueError('lossless size/checksum/end-of-stream mismatch')
    return raw


def encode_packet(packet, method='xz'):
    """An expanding compression falls back to the original self-framed packet."""
    unpack(packet)
    coded = _encode(packet, method)
    return coded if len(coded) < len(packet) else packet


def decode_packet(packet):
    raw = _decode(packet)
    unpack(raw)
    return raw


def restore(messages):
    """Decode using received messages alone; no source, dictionary or TX cache."""
    if len(messages) not in (1, 2):
        raise ValueError('expected one bundle/base packet or two separate packets')
    first = _decode(messages[0])
    if first.startswith(b'BSB1'):
        if len(messages) != 1 or len(first) < BUNDLE.size:
            raise ValueError('invalid bundle framing')
        _, base_size, trajectory_size = BUNDLE.unpack(first[:BUNDLE.size])
        if not base_size or not trajectory_size or len(first) != BUNDLE.size + base_size + trajectory_size:
            raise ValueError('invalid bundle lengths')
        base = first[BUNDLE.size:BUNDLE.size + base_size]
        trajectory = first[BUNDLE.size + base_size:]
        unpack(base); unpack(trajectory)
        return {'metadata': base, 'trajectory': trajectory}
    unpack(first)
    result = {'metadata': first}
    if len(messages) == 2:
        result['trajectory'] = decode_packet(messages[1])
    return result


def cost(messages, **channel):
    return sum(accounting(len(packet), **channel)['complex_channel_uses'] for packet in messages)


def choose(metadata, trajectory=None, *, method='xz', bundle=False, **channel):
    """Minimize paid LDPC symbols, with raw fallback and no quality-based tuning."""
    original = [metadata] + ([trajectory] if trajectory is not None else [])
    for packet in original:
        unpack(packet)
    candidates = [('legacy', original),
                  ('separate_' + method, [encode_packet(p, method) for p in original])]
    if bundle and trajectory is not None:
        raw = BUNDLE.pack(b'BSB1', len(metadata), len(trajectory)) + metadata + trajectory
        candidates.append(('bundle_' + method, [_encode(raw, method)]))
    # Equal symbol cost prefers fewer bytes, then original order. Never expand
    # paid bandwidth, including the new envelope, lengths, CRC, padding/parity.
    name, messages = min(candidates, key=lambda row: (cost(row[1], **channel), sum(map(len, row[1]))))
    expected = {'metadata': metadata}
    if trajectory is not None:
        expected['trajectory'] = trajectory
    if restore(messages) != expected:
        raise ValueError('sender lossless self-check failed')
    return name, messages


def transmit(metadata, trajectory, snr_db, seed, *, method='xz', bundle=False, **channel):
    """Actual LDPC/QAM/AWGN; fail closed before giving bytes to the receiver."""
    from .metadata_channel import transmit as channel_transmit
    name, messages = choose(metadata, trajectory, method=method, bundle=bundle, **channel)
    received, reports = [], []
    for offset, packet in enumerate(messages):
        raw, report = channel_transmit(packet, snr_db, seed + offset, **channel)
        received.append(raw); reports.append(report)
    # Framing/decompression/CRC validation is receiver-only. Equality below is
    # an experiment assertion, not a receiver dependency or free retransmission.
    restored = restore(received)
    expected = {'metadata': metadata}
    if trajectory is not None:
        expected['trajectory'] = trajectory
    if restored != expected:
        raise ValueError('channel did not preserve the digital model inputs')
    return restored, {'mode': name, 'reports': reports,
                      'legacy_complex_channel_uses': cost([metadata] + ([trajectory] if trajectory else []), **channel),
                      'complex_channel_uses': sum(r['complex_channel_uses'] for r in reports),
                      'exact_match': True}
