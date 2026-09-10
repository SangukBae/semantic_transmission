"""Framed UTF-8 semantic metadata; damaged packets never reach the decoder."""

import json
import math
from pathlib import PurePosixPath
import struct
import zlib

MAGIC = b"STX1"
HEADER = struct.Struct("!4sII")
MAX_PAYLOAD = 16 * 1024 * 1024


def validate_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("metadata must contain at least one segment")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "text", "flow"}:
            raise ValueError("each segment requires exactly path, text and flow")
        path = row["path"]
        if (not isinstance(path, str) or not path or "\\" in path
                or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts):
            raise ValueError("segment paths must be relative receiver paths")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError("segment captions must be nonempty strings")
        if isinstance(row["flow"], bool) or not isinstance(row["flow"], (int, float)):
            raise ValueError("flow must be a finite number")
        if not math.isfinite(row["flow"]) or row["flow"] < 0:
            raise ValueError("flow must be finite and nonnegative")
    return rows


def encode(rows):
    validate_rows(rows)
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("semantic metadata exceeds the packet size limit")
    return HEADER.pack(MAGIC, len(payload), zlib.crc32(payload)) + payload


def decode(packet):
    if len(packet) < HEADER.size:
        raise ValueError("truncated metadata packet header")
    magic, size, crc = HEADER.unpack(packet[:HEADER.size])
    payload = packet[HEADER.size:]
    if magic != MAGIC or size > MAX_PAYLOAD or size != len(payload):
        raise ValueError("invalid metadata packet framing")
    if zlib.crc32(payload) != crc:
        raise ValueError("metadata packet CRC mismatch")
    return validate_rows(json.loads(payload.decode("utf-8")))


def accounting(packet_bytes, *, k=6144, n=9216, bits_per_symbol=4):
    if not isinstance(packet_bytes, int) or packet_bytes <= 0:
        raise ValueError("packet_bytes must be a positive integer")
    if not (0 < k < n and bits_per_symbol > 0 and n % bits_per_symbol == 0):
        raise ValueError("invalid LDPC/modulation parameters")
    bits = packet_bytes * 8
    blocks = (bits + k - 1) // k
    return {"packet_bytes": packet_bytes, "information_bits": bits, "ldpc_k": k,
            "ldpc_n": n, "bits_per_symbol": bits_per_symbol, "blocks": blocks,
            "padding_bits": blocks * k - bits, "coded_bits": blocks * n,
            "complex_channel_uses": blocks * n // bits_per_symbol}
