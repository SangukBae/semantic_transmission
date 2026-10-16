import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from semantic_transmission.key_reduction_selection import guard, proposals, splat, subset_packet
from semantic_transmission.wire import pack, unpack
from semantic_transmission.webvid_ablation import Stages


class KeyReductionTests(unittest.TestCase):
    def packet(self):
        header = dict(keyframes=[dict(index=i, complex_offset=n*2, complex_count=2,
            rate_offset=n, rate_bytes=1, rate_count=2, average_power=1.) for n, i in enumerate([0, 4, 8])],
            decoder=dict(seed=2025), segments=[dict(text='a'), dict(text='b')])
        return pack(header, b'abc'), bytes(range(48))

    def test_subset_preserves_exact_codec_payloads_and_rebases_offsets(self):
        packet, visual = self.packet()
        changed, values = subset_packet(packet, visual, [0, 8], [dict(text='merged')], 2026)
        header, rates = unpack(changed)
        self.assertEqual(rates, b'ac')
        self.assertEqual(values, visual[:16]+visual[32:])
        self.assertEqual([k['complex_offset'] for k in header['keyframes']], [0, 2])
        self.assertEqual([k['rate_offset'] for k in header['keyframes']], [0, 1])
        self.assertEqual(header['decoder']['seed'], 2026)
        self.assertEqual(unpack(packet)[0]['decoder']['seed'], 2025)

    def test_rejects_missing_endpoint_duplicate_or_short_symbols(self):
        packet, visual = self.packet()
        for keys in [[4, 8], [0, 4], [0, 4, 4, 8], [0, 7, 8]]:
            with self.assertRaises(ValueError):
                subset_packet(packet, visual, keys, [{}]*(len(keys)-1), 2025)
        with self.assertRaises(ValueError):
            subset_packet(packet, visual[:-1], [0, 8], [{}], 2025)

    def sample(self):
        q = dict(psnr_db=20., ssim=.7, lpips_vgg=.3, dists=.13, clip=.96)
        return dict(quality=q.copy(), nonkeys=q.copy(), temporal=dict(flow_error_mean=1.,
            cadence_error_mean=.3, false_freeze_rate=.02), boundary_error=2., channel_uses=1000)

    def test_guards_reject_rate_increase_freezing_nonkey_loss_and_nan(self):
        a = self.sample(); b = copy.deepcopy(a); b['channel_uses'] = 900
        self.assertTrue(guard(a, b)['passed'])
        bad = copy.deepcopy(b); bad['nonkeys']['psnr_db'] -= .2
        self.assertIn('nonkeys.psnr_db', guard(a, bad)['failures'])
        bad = copy.deepcopy(b); bad['temporal']['false_freeze_rate'] += .006
        self.assertIn('false_freeze_rate', guard(a, bad)['failures'])
        bad = copy.deepcopy(b); bad['quality']['lpips_vgg'] = float('nan')
        self.assertFalse(guard(a, bad)['passed'])
        self.assertFalse(guard(a, a)['passed'])

    def test_zero_flow_splat_is_identity(self):
        frame = np.random.default_rng(1).integers(0, 255, (12, 16, 3), dtype=np.uint8)
        output, weight = splat(frame, np.zeros((12, 16, 2)), .5)
        np.testing.assert_array_equal(output, frame)
        np.testing.assert_array_equal(weight, np.ones((12, 16)))

    def test_occlusion_unknown_and_endpoints_are_protected(self):
        frame = np.random.default_rng(2).integers(0, 255, (160, 288, 3), dtype=np.uint8)
        keys = [0, 4, 8, 12, 16]
        decisions = [dict(candidate=k, selected=True, result=dict(kind=kind))
                     for k, kind in [(4, 'occlusion'), (8, 'pose')]]
        result = proposals(keys, decisions, lambda _: frame)
        self.assertEqual(result['protected_keys'], [0, 4, 12, 16])
        self.assertEqual(result['removed'], [8])
        self.assertEqual(result['single'], [0, 4, 12, 16])

    def test_two_endpoint_case_needs_no_flow(self):
        result = proposals([0, 239], [], lambda _: self.fail('should not read pixels'))
        self.assertEqual(result['removed'], [])

    def test_asset_alias_requires_identical_path_policy_and_bytes(self):
        from semantic_transmission.bsu_key_reduction import canonical_asset
        from semantic_transmission.artifacts import sha256
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); real = root/'real'; real.mkdir(); (real/'weight').write_bytes(b'weight')
            alias = root/'alias'; alias.symlink_to(real)
            expected = dict(path=str(real), files={'weight': sha256(real/'weight')}, rank=32)
            original = dict(expected, path=str(alias))
            self.assertEqual(canonical_asset(original, expected), expected)
            self.assertEqual(original['path'], str(alias))
            with self.assertRaises(ValueError):
                canonical_asset(dict(original, rank=16), expected)
            (real/'weight').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                canonical_asset(original, expected)

    def test_long_gap_is_not_merged(self):
        decision = dict(candidate=80, selected=True, result=dict(kind='position'))
        result = proposals([0, 80, 160], [decision], lambda _: self.fail('protected long gap'))
        self.assertEqual(result['removed'], [])

    def test_receipt_reuses_verified_inputs_and_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); calls = []
            def init(_):
                (root/'receiver').mkdir(); (root/'receiver/input').write_text('original'); calls.append(1)
            Stages(root, 'protocol').step('channel', ['receiver'], ['receiver/input'], init)
            (root/'receiver/new_output').write_text('later stage')
            Stages(root, 'protocol').step('channel', ['receiver'], ['receiver/input'], init)
            self.assertEqual(calls, [1])
            (root/'receiver/input').write_text('corrupt')
            with self.assertRaises(ValueError):
                Stages(root, 'protocol').step('channel', ['receiver'], ['receiver/input'], init)


if __name__ == '__main__':
    unittest.main()
