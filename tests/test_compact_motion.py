import unittest
import numpy as np

from semantic_transmission import compact_motion as m
from semantic_transmission.wire import pack,unpack
from semantic_transmission.packets import accounting


class MotionPacketTests(unittest.TestCase):
    def test_quantized_roundtrip_and_real_packet_capacity(self):
        rng=np.random.default_rng(42);values=rng.uniform(-.5,.5,(15,12));confidence=rng.uniform(0,1,15)
        encoded=m.encode(values,confidence,384);decoded=m.decode(encoded,384)
        self.assertLessEqual(np.max(np.abs(values-decoded['values'])),.5/m.SCALE)
        self.assertLessEqual(np.max(np.abs(confidence-decoded['confidence'])),.5/255)
        packet=pack(dict(video=dict(frames=384)),b'unchanged rate indices')
        enriched=m.add_to_packet(packet,encoded,len(packet)+582)
        self.assertEqual(unpack(enriched)[1],b'unchanged rate indices')
        self.assertLessEqual(len(enriched)-len(packet),582)
        self.assertEqual(accounting(1722)['blocks'],accounting(1722+len(enriched)-len(packet))['blocks'])
        with self.assertRaises(ValueError):m.add_to_packet(packet,encoded,len(packet)+100)

    def test_reject_invalid_or_misaligned_payload(self):
        encoded=m.encode(np.zeros((15,12)),np.ones(15),49)
        with self.assertRaises(ValueError):m.decode(encoded,384)
        with self.assertRaises(ValueError):m.decode(encoded[:-4])
        with self.assertRaises(ValueError):m.encode(np.full((15,12),5),np.ones(15),49)
        packet=bytearray(m.add_to_packet(pack(dict(video=dict(frames=49))),encoded));packet[-1]^=1
        with self.assertRaises(ValueError):unpack(bytes(packet))

    def test_known_camera_translation(self):
        import cv2
        rng=np.random.default_rng(20)
        image=rng.integers(0,256,(160,288,3),dtype=np.uint8)
        frames=np.stack([cv2.warpAffine(image,np.float32([[1,0,i],[0,1,0]]),(288,160)) for i in range(12)])
        encoded,audit=m.extract(frames);result=m.decode(encoded)
        self.assertAlmostEqual(float(result['values'][-1,2])*288,11,delta=.8)
        self.assertLess(audit['failed_transitions'],3)


class AdapterTests(unittest.TestCase):
    def test_actual_two_key_bsu_conditioning(self):
        import torch
        from pathlib import Path
        from semantic_transmission.condition_collision import apply_guard,upstream_functions
        functions=upstream_functions(Path(__file__).resolve().parents[1])
        functions.apply_mask_strategy.__globals__['torch']=torch
        z=torch.zeros(1,4,113,2,2)
        refs=[[torch.ones(4,1,2,2),torch.full((4,1,2,2),2.)]]
        mask,records=apply_guard(z,refs,['0;0,1,0,-1,1'],0,5,2,functions)
        self.assertFalse(records[0]['repaired'])
        self.assertEqual(records[0]['operations'][-1]['target'],110)
        self.assertTrue(torch.all(z[:,:,0]==1))
        self.assertTrue(torch.all(z[:,:,110]==2))
        self.assertEqual(int((mask==0).sum()),2)

    def test_microblock_timeline(self):
        from semantic_transmission.motion_condition_adapter import latent_times
        self.assertEqual(len(latent_times(384)),113)
        self.assertEqual(len(latent_times(49)),14)
        self.assertEqual(len(latent_times(16)),4)

    def test_zero_motion_preserves_identical_endpoint_prior(self):
        import torch
        from semantic_transmission.motion_condition_adapter import features,MotionPrior
        first=torch.rand(1,4,8,12)
        data=m.decode(m.encode(np.zeros((15,12)),np.ones(15),49))
        x=features(first,first,data,49)
        result=MotionPrior()(x)
        torch.testing.assert_close(result,first.unsqueeze(2).expand_as(result),atol=1e-5,rtol=1e-5)

    def test_guidance_is_not_multiplied_by_cfg(self):
        import torch
        from semantic_transmission.motion_condition_adapter import guidance_hook
        z=torch.zeros(2,4,3,4,4);out=torch.zeros(2,8,3,4,4);out[0,:4]=2;out[1,:4]=1
        result=guidance_hook(torch.ones(1,4,3,4,4),.15)(None,(z,torch.tensor([500.,500.])),out)
        baseline=1+7*(2-1)
        expected=baseline+.15*(1-.5*baseline)/.5
        guided=result[1,:4]+7*(result[0,:4]-result[1,:4])
        torch.testing.assert_close(guided,torch.full_like(guided,expected))
        self.assertTrue(torch.equal(result[:,4:],out[:,4:]))


if __name__=='__main__':unittest.main()
