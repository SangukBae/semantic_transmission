import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from semantic_transmission import sparc_selector_distributed as d


class PartitionTests(unittest.TestCase):
    def test_balances_work_and_keeps_partial_measurements_on_remote(self):
        items = [('long', 288), ('medium', 144)] + [(f'short{i}', 24) for i in range(30)]
        assigned, weight = d.partition(items, ['short5'])
        self.assertIn('short5', assigned['remote'])
        self.assertEqual(set(assigned['remote']) | set(assigned['local']), {i for i, _ in items})
        self.assertFalse(set(assigned['remote']) & set(assigned['local']))
        self.assertLessEqual(abs(weight['remote']-weight['local']), 24)
        self.assertEqual(d.partition(items, ['short5']), (assigned, weight))

    def test_plan_rejects_overlap_even_with_valid_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = dict(prepared=['a', 'b'], completed=[], assignments=dict(local=['a'], remote=['a', 'b']))
            p['signature'] = d.digest(p)
            path = Path(tmp)/'plan.json'
            d.save(path, p)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                d.checked_plan(path)


class PublicationTests(unittest.TestCase):
    def fixture(self, root):
        row = dict(id='clip1', parent='parent1', split='train')
        protocol = dict(signature='frozen-protocol', snr_db=10., channel_seed=42)
        d.save(root/'sources.json', [row])
        d.save(root/'protocol.json', dict(supervision_protocol=protocol))
        plan = dict(prepared=['clip1'], completed=[], assignments=dict(local=['clip1'], remote=[]),
                    source_manifest_sha256=d.sha(root/'sources.json'),
                    protocol_sha256=d.sha(root/'protocol.json'),
                    orchestrator_sha256=d.sha(Path(d.__file__)))
        plan['signature'] = d.digest(plan)
        plan_path = root/'.distributed/plan.json'
        d.save(plan_path, plan)
        source = root/'.distributed/incoming/clips/clip1'
        identity = dict(protocol=protocol['signature'], indices=[0, 2], source_sha256='frames')
        name = d.digest(identity)[:24]
        obs = source/'observations'/name
        obs.mkdir(parents=True)
        for f in ['sent.bin', 'received.bin', 'delivered.mp4']:
            (obs/f).write_bytes(b'video' if f.endswith('mp4') else b'packet')
        result = dict(identity=identity, artifacts={f: d.sha(obs/f) for f in
            ['sent.bin', 'received.bin', 'delivered.mp4']},
            sender=dict(complex_channel_uses=10), channel=dict(bit_errors=0, complex_channel_uses=10),
            receiver=dict(source_pixels_at_receiver=False),
            quality=dict(mse=.1, lpips_vgg=.2, temporal_mse=.3, local_mae95=.4, psnr_db=10.))
        d.save(obs/'result.json', result)
        np.savez(source/'labels.npz', x=np.ones((1, 4), np.float32), y=np.ones((1, 5), np.float32))
        receipt = dict(input=row, protocol=protocol['signature'], samples=1,
            records=[dict(before_observation=name, after_observation=name)], sha256=d.sha(source/'labels.npz'))
        d.save(source/'labels.json', receipt)
        return row, protocol, plan_path, source, obs

    def test_corrupt_video_never_publishes_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row, protocol, plan, source, obs = self.fixture(root)
            (obs/'delivered.mp4').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'artifact checksum'):
                d.import_clips(root, plan, ['clip1'])
            self.assertFalse((root/'clips/clip1/labels.json').exists())
            self.assertFalse((root/'.distributed/imports.json').exists())

    def test_import_is_verified_and_idempotent_after_retransfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row, protocol, plan, source, obs = self.fixture(root)
            backup = root/'transfer-backup'
            shutil.copytree(source, backup)
            d.import_clips(root, plan, ['clip1'])
            target = root/'clips/clip1'
            before = d.sha(target/'labels.json')
            self.assertEqual(d.verify_clip(target, row, protocol)['samples'], 1)
            shutil.copytree(backup, source, dirs_exist_ok=True)
            d.import_clips(root, plan, ['clip1'])
            self.assertEqual(d.sha(target/'labels.json'), before)
            self.assertEqual(len(d.read(root/'.distributed/imports.json')), 1)

    def test_recovers_publication_after_observation_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row, protocol, plan, source, obs = self.fixture(root)
            target = root/'clips/clip1/observations'/obs.name
            shutil.copytree(obs, target)
            d.import_clips(root, plan, ['clip1'])
            d.verify_clip(root/'clips/clip1', row, protocol)

    def test_conflicting_existing_label_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row, protocol, plan, source, obs = self.fixture(root)
            target = root/'clips/clip1'
            shutil.copytree(source, target)
            np.savez(target/'labels.npz', x=np.zeros((1, 4)), y=np.zeros((1, 5)))
            receipt = d.read(target/'labels.json')
            receipt['sha256'] = d.sha(target/'labels.npz')
            d.save(target/'labels.json', receipt)
            before = d.sha(target/'labels.json')
            with self.assertRaisesRegex(ValueError, 'existing different labels'):
                d.import_clips(root, plan, ['clip1'])
            self.assertEqual(before, d.sha(target/'labels.json'))

    def test_wrong_owner_and_wrong_protocol_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row, protocol, plan, source, obs = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'outside local assignment'):
                d.import_clips(root, plan, ['unassigned'])
            receipt = d.read(source/'labels.json')
            receipt['protocol'] = 'different'
            d.save(source/'labels.json', receipt)
            with self.assertRaisesRegex(ValueError, 'source/protocol'):
                d.import_clips(root, plan, ['clip1'])

    def test_nonfinite_labels_rejected_even_with_updated_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row, protocol, plan, source, obs = self.fixture(root)
            np.savez(source/'labels.npz', x=np.ones((1, 4)), y=np.full((1, 5), np.nan))
            receipt = d.read(source/'labels.json')
            receipt['sha256'] = d.sha(source/'labels.npz')
            d.save(source/'labels.json', receipt)
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                d.import_clips(root, plan, ['clip1'])

    def rejection(self, root):
        row, protocol, plan, source, obs = self.fixture(root)
        (source/'labels.json').unlink()
        sent = (obs/'sent.bin').read_bytes()
        (obs/'failed_received.bin').write_bytes(bytes([sent[0] ^ 1])+sent[1:])
        r = dict(status='REJECTED_CHANNEL', input=row, protocol=protocol['signature'],
            observation=obs.name, use_for_training=False,
            report=dict(bit_errors=1, snr_db=10., seed=42, bits_per_symbol=4, ldpc_k=6144, ldpc_n=9216),
            artifacts={k:d.sha(obs/k) for k in ['sent.bin', 'failed_received.bin']})
        d.save(source/'label_failure.json', r)
        return row, protocol, plan, source, obs

    def test_channel_failure_is_published_separately_without_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row, protocol, plan, source, obs = self.rejection(root)
            d.import_clips(root, plan, ['clip1'])
            d.verify_rejection(root/'clips/clip1', row, protocol)
            self.assertFalse((root/'clips/clip1/labels.json').exists())
            self.assertEqual(d.read(root/'.distributed/imports.json')['clip1']['status'], 'REJECTED_CHANNEL')

    def test_rejection_with_invented_error_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row, protocol, plan, source, obs = self.rejection(root)
            r = d.read(source/'label_failure.json')
            r['report']['bit_errors'] = 2
            d.save(source/'label_failure.json', r)
            with self.assertRaisesRegex(ValueError, 'unverified channel rejection'):
                d.import_clips(root, plan, ['clip1'])

    def test_rejection_cannot_replace_existing_valid_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row, protocol, plan, source, obs = self.fixture(root)
            shutil.copytree(source, root/'clips/clip1')
            (source/'labels.json').unlink()
            sent = (obs/'sent.bin').read_bytes()
            (obs/'failed_received.bin').write_bytes(bytes([sent[0] ^ 1])+sent[1:])
            r = dict(status='REJECTED_CHANNEL', input=row, protocol=protocol['signature'],
                observation=obs.name, use_for_training=False,
                report=dict(bit_errors=1, snr_db=10., seed=42, bits_per_symbol=4, ldpc_k=6144, ldpc_n=9216),
                artifacts={k:d.sha(obs/k) for k in ['sent.bin', 'failed_received.bin']})
            d.save(source/'label_failure.json', r)
            with self.assertRaisesRegex(ValueError, 'replace valid labels'):
                d.import_clips(root, plan, ['clip1'])


if __name__ == '__main__':
    unittest.main()
