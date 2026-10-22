import unittest

from semantic_transmission import sparc_uf_transport as t
from semantic_transmission.packets import accounting
from semantic_transmission.wire import pack


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.video = dict(frames=17,width=64,height=64,fps=24.)
        self.indices = [0,2,4,6,8,10,12,14,16]
        self.keys = [0,8,16]
        self.weights = {'image':'a'*64,'video':'b'*64}

    def packet(self, mode):
        chunks = t.groups(self.indices,self.keys,mode)
        return t.make(self.video,self.keys,mode,[(g,bytes(range(256))*4) for g in chunks],
                      self.weights,32)

    def test_interval_assigns_endpoints_once_and_pays_padding(self):
        groups = t.groups(self.indices,self.keys,'interval')
        self.assertEqual(groups,[[0,2,4,6],[8,10,12,14],[16]])
        self.assertEqual(sum(groups,[]),self.indices)
        self.assertEqual([t.padding(len(g)) for g in groups],[5,5,0])
        self.assertEqual(t.padding(9),0)
        self.assertEqual(t.padding(10),7)

    def test_packet_round_trip_and_total_rate(self):
        for mode in ('scene','interval'):
            packet = self.packet(mode)
            header,payload = t.parse(packet,self.weights)
            self.assertEqual(sum((c['indices'] for c in header['chunks']),[]),self.indices)
            self.assertEqual(payload,bytes(range(256))*4*len(header['chunks']))
            rate = accounting(len(packet))
            self.assertGreaterEqual(rate['coded_bits'],len(packet)*8*1.5)
            self.assertEqual(rate['complex_channel_uses']*4,rate['coded_bits'])

    def test_receiver_rejects_damage_wrong_models_and_timeline(self):
        packet = self.packet('interval')
        for damaged in (packet[:-1],packet+b'x',packet[:-1]+bytes([packet[-1]^1])):
            with self.assertRaises(ValueError):
                t.parse(damaged,self.weights)
        with self.assertRaises(ValueError):
            t.parse(packet,{'wrong':'model'})
        header,payload = t.parse(packet,self.weights)
        header['chunks'][1]['indices'][0] = 6
        with self.assertRaises(ValueError):
            t.parse(pack(header,payload),self.weights)

    def test_padding_and_offset_cannot_be_forged(self):
        for field in ('padded_frames','offset','bytes'):
            header,payload = t.parse(self.packet('interval'),self.weights)
            header['chunks'][0][field] += 1
            with self.assertRaises(ValueError):
                t.parse(pack(header,payload),self.weights)


if __name__ == '__main__':
    unittest.main()
