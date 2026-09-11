"""Versioned reconstruction metadata and packed NTSCC rate indices."""
import json
import struct
import zlib

HEADER = struct.Struct("!4sIII")


def pack(header, payload=b""):
    metadata = json.dumps(header, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    body = metadata + payload
    return HEADER.pack(b"LGV1", len(metadata), len(payload), zlib.crc32(body)) + body


def unpack(packet):
    if len(packet) < HEADER.size:
        raise ValueError("truncated transport header")
    magic, metadata_size, payload_size, crc = HEADER.unpack(packet[:HEADER.size])
    body = packet[HEADER.size:]
    if magic != b"LGV1" or len(body) != metadata_size + payload_size or zlib.crc32(body) != crc:
        raise ValueError("invalid transport framing or checksum")
    header = json.loads(body[:metadata_size].decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("transport metadata must be an object")
    return header, body[metadata_size:]


def pack_indices(indices):
    values = list(indices)
    if not values or any(not isinstance(v, int) or not 0 <= v < 16 for v in values):
        raise ValueError("rate indices must be integers in [0,15]")
    if len(values) % 2:
        values.append(0)
    return bytes((a << 4) | b for a, b in zip(values[::2], values[1::2]))


def unpack_indices(data, count):
    if count < 1 or len(data) != (count + 1) // 2:
        raise ValueError("rate-index payload length mismatch")
    if count % 2 and data[-1] & 15:
        raise ValueError("nonzero rate-index padding")
    return [v for byte in data for v in (byte >> 4, byte & 15)][:count]
