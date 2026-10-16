import struct
import unittest
import zlib

from semantic_transmission.lossless_transport import (
    BUNDLE, ENVELOPE, MAX_BYTES, _encode, choose, cost, decode_packet, encode_packet, restore,
)
from semantic_transmission.wire import pack


class LosslessTransportTests(unittest.TestCase):
    def setUp(self):
        self.base = pack({'text': '사람이 일어서고 열차가 지나간다. ' * 300,
                          'flow': 0.12345678901234568, 'indices': [0, 31, 80]}, bytes(range(256)) * 8)
        self.trajectory = pack({'base_sha': 'a' * 64, 'windows': ['abcdef' * 100] * 4})

    def test_roundtrip_is_byte_exact_for_each_codec(self):
        for method in ('raw', 'zlib', 'xz'):
            with self.subTest(method=method):
                self.assertEqual(decode_packet(encode_packet(self.base, method)), self.base)

    def test_bundle_restores_both_original_packets_without_side_data(self):
        for method in ('zlib', 'xz'):
            name, messages = choose(self.base, self.trajectory, method=method, bundle=True)
            self.assertLess(cost(messages), cost([self.base, self.trajectory]))
            self.assertEqual(restore(messages), {'metadata': self.base, 'trajectory': self.trajectory})

    def test_raw_fallback_does_not_expand_tiny_packet(self):
        packet = pack({'x': 1})
        for method in ('zlib', 'xz'):
            self.assertEqual(encode_packet(packet, method), packet)
            self.assertLessEqual(cost(choose(packet, method=method)[1]), cost([packet]))

    def test_corruption_truncation_trailing_streams_and_header_damage_rejected(self):
        for method in ('zlib', 'xz'):
            packet = encode_packet(self.base, method)
            damaged = [packet[:-1], packet + b'junk', packet[:5] + b'\1' + packet[6:],
                       packet[:4] + b'\x7f' + packet[5:], packet[:12] + b'\0\0\0\0' + packet[16:]]
            for offset in (8, 16, ENVELOPE.size + 5, len(packet) - 2):
                value = bytearray(packet); value[offset] ^= 1; damaged.append(bytes(value))
            body = packet[ENVELOPE.size:] + packet[ENVELOPE.size:]
            fields = list(ENVELOPE.unpack(packet[:ENVELOPE.size])); fields[3] = len(body)
            damaged.append(ENVELOPE.pack(*fields) + body)
            for bad in damaged:
                with self.subTest(method=method, prefix=bad[:20]), self.assertRaises(ValueError):
                    decode_packet(bad)

    def test_decompression_size_bound(self):
        for method in ('zlib', 'xz'):
            packet = _encode(b'x' * 10000, method)
            for size in (0, 1, 9999, 10001, MAX_BYTES + 1):
                fields = list(ENVELOPE.unpack(packet[:ENVELOPE.size])); fields[2] = size
                with self.assertRaises(ValueError):
                    decode_packet(ENVELOPE.pack(*fields) + packet[ENVELOPE.size:])

    def test_invalid_bundles_rejected(self):
        body = self.base + self.trajectory
        for a, b in ((0, len(body)), (len(self.base), 0), (len(self.base) - 1, len(self.trajectory)),
                     (len(self.base) + 1, len(self.trajectory) - 1)):
            with self.assertRaises(ValueError):
                restore([_encode(BUNDLE.pack(b'BSB1', a, b) + body, 'xz')])
        bundle = _encode(BUNDLE.pack(b'BSB1', len(self.base), len(self.trajectory)) + body, 'xz')
        with self.assertRaises(ValueError):
            restore([bundle, self.trajectory])

    def test_legacy_compatibility_and_no_trajectory(self):
        self.assertEqual(restore([self.base]), {'metadata': self.base})
        self.assertEqual(restore([self.base, self.trajectory]),
                         {'metadata': self.base, 'trajectory': self.trajectory})
        self.assertEqual(restore(choose(self.base, bundle=True)[1]), {'metadata': self.base})
        for messages in ([], [self.base] * 3):
            with self.assertRaises(ValueError):
                restore(messages)

    def test_accounting_includes_padding_and_all_messages(self):
        self.assertEqual(cost([b'a']), 2304)
        self.assertEqual(cost([b'a' * 769]), 4608)
        self.assertEqual(cost([b'a', b'b']), 4608)
        self.assertEqual(cost([b'a'], bits_per_symbol=2), 4608)


if __name__ == '__main__':
    unittest.main()
