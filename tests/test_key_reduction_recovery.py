import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from semantic_transmission import bsu_key_reduction as run
from semantic_transmission import bsu_key_reduction_recovery as recovery
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.webvid5 import fingerprint
from semantic_transmission.webvid_ablation import Stages


class ManifestTests(unittest.TestCase):
    def test_actual_frozen_separator_roundtrips_all_paths_and_labels(self):
        import yaml
        config = yaml.safe_load((run.REPO/'07_downstream/timesformer/TimeSformer_divST_8x32_224_TEST.yaml').read_text())
        separator = config['DATA']['PATH_LABEL_SEPARATOR']
        rows = [dict(path='/video with spaces/a.mp4', label=208), dict(path='/video/b.mp4', label=0)]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'test.csv'; run.write_action_manifest(path, rows, separator)
            actual = [line.split(separator) for line in path.read_text().splitlines()]
            self.assertEqual(actual, [[r['path'], str(r['label'])] for r in rows])
            legacy = '/video/a.mp4 208'
            self.assertNotEqual(len(legacy.split(separator)), 2)

    def test_uses_configured_separator_instead_of_hardcoding_comma(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'test.csv'
            run.write_action_manifest(p, [dict(path='/video/a.mp4', label=399)], '\t')
            self.assertEqual(p.read_text(), '/video/a.mp4\t399\n')

    def test_rejects_unrepresentable_rows_before_overwriting_existing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'test.csv'; p.write_text('preserve')
            for row in [dict(path='/a,b.mp4', label=1), dict(path='/a\nb.mp4', label=1),
                        dict(path='/a.mp4', label=400), dict(path='/a.mp4', label=True)]:
                with self.assertRaises(ValueError):
                    run.write_action_manifest(p, [dict(path='/valid.mp4', label=0), row], ',')
                self.assertEqual(p.read_text(), 'preserve')


class RecoveryTests(unittest.TestCase):
    def test_missing_or_failed_stage_never_runs_or_archives_partial_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); partial = root/'partial'; partial.write_text('keep')
            for status in [None, 'FAILED', 'RUNNING']:
                if status:
                    write_json(root/'stages/reconstruct.json', dict(status=status))
                with self.assertRaises(ValueError):
                    recovery.CompletedOnlyStages(root, 'frozen').step('reconstruct', ['partial'], ['partial'],
                        lambda _: self.fail('must not generate'))
                self.assertEqual(partial.read_text(), 'keep')
                self.assertFalse((root/'failed_attempts').exists())

    def test_complete_stage_is_verified_and_corruption_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            Stages(root, 'frozen').step('evaluate', ['quality'], ['quality'], lambda _: (root/'quality').write_text('done'))
            recovery.CompletedOnlyStages(root, 'frozen').step('evaluate', ['quality'], ['quality'], lambda _: self.fail())
            (root/'quality').write_text('changed')
            with self.assertRaises(ValueError):
                recovery.CompletedOnlyStages(root, 'frozen').step('evaluate', ['quality'], ['quality'], lambda _: self.fail())

    def test_only_exact_known_code_replacement_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); changed = root/recovery.FILE; changed.parent.mkdir(parents=True)
            changed.write_text('fixed'); other = root/'other.py'; other.write_text('frozen')
            plan = dict(code={recovery.FILE: recovery.LEGACY_SHA, 'other.py': sha256(other)})
            plan['signature'] = fingerprint(plan)
            with patch.object(recovery, 'FIXED_SHA', sha256(changed)):
                recovery.verify_code(plan, root)
                other.write_text('unrelated edit')
                with self.assertRaises(ValueError):
                    recovery.verify_code(plan, root)

    def test_action_accuracy_cannot_disagree_with_logits(self):
        row = dict(id='clip', variant='baseline_2025', label=0)
        result = dict(records=[dict(input=row, summed_logits=[0.]*400, top5=[0,1,2,3,4],
            top1_correct=True, top5_correct=True)], labels_used_for_key_selection=False,
            summary={'baseline_2025':dict(videos=1, top1_correct=1, top5_correct=1)})
        recovery.validate_action(dict(rows=[row]), result)
        bad = copy.deepcopy(result); bad['records'][0]['top1_correct'] = False
        with self.assertRaises(ValueError):
            recovery.validate_action(dict(rows=[row]), bad)


if __name__ == '__main__':
    unittest.main()
