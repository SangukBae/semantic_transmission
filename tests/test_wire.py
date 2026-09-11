import unittest
from semantic_transmission.wire import pack, unpack, pack_indices, unpack_indices


class WireTests(unittest.TestCase):
    def test_binary_rate_payload_and_unicode_metadata_roundtrip(self):
        indices = list(range(16)) + [9]
        header = {"caption": '사람, 자동차 "둘"\nnext', "rate_count": len(indices)}
        packet = pack(header, pack_indices(indices))
        restored, payload = unpack(packet)
        self.assertEqual(restored, header)
        self.assertEqual(unpack_indices(payload, len(indices)), indices)
        self.assertEqual(len(payload), 9)

    def test_transport_damage_cannot_change_decoder_control(self):
        packet = pack({"frames": 100, "fps": 10}, b"\x12\x34")
        for bad in (packet[:-1], packet + b"extra", b"BAD!" + packet[4:],
                    packet[:20] + bytes([packet[20] ^ 1]) + packet[21:]):
            with self.assertRaises(ValueError):
                unpack(bad)

    def test_invalid_rates_and_padding_rejected(self):
        for values in ([], [-1], [16], [2.5]):
            with self.assertRaises(ValueError):
                pack_indices(values)
        with self.assertRaises(ValueError):
            unpack_indices(b"\x12", 1)
        with self.assertRaises(ValueError):
            unpack_indices(b"\x10", 4)


if __name__ == "__main__":
    unittest.main()
