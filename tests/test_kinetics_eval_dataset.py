"""Selection, real video normalization and portable integrity verification."""
import csv
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('kinetics_builder',Path(__file__).parents[1]/'scripts/prepare_kinetics_eval.py')
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class DatasetTests(unittest.TestCase):
    def test_selection_is_class_balanced_deterministic_and_excludes_parent_ids(self):
        rows = [dict(label=label,youtube_id=f'{label}{i:010}',time_start='0',time_end='10')
                for label in ('a','b') for i in range(8)]
        duplicate = dict(rows[0],time_start='20',time_end='30')
        excluded = {rows[1]['youtube_id']}
        a = builder.rank_candidates(rows+[duplicate],excluded,42,3)
        b = builder.rank_candidates(list(reversed(rows))+[duplicate],excluded,42,3)
        self.assertEqual(a,b)
        self.assertEqual({len(v) for v in a.values()},{3})
        for items in a.values():
            ids=[r['youtube_id'] for r in items]
            self.assertEqual(len(ids),len(set(ids)))
            self.assertFalse(set(ids)&excluded)

    def test_real_normalization_relative_manifest_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'candidates').mkdir()
            row=dict(label='fixture',youtube_id='fixture0001',time_start='0',time_end='3')
            source=root/'candidates'/builder.identity(row)
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=64x36:rate=10',
                '-t','3','-c:v','libx264','-threads','1',str(source)],check=True)
            p=dict(inputs={},labels=['fixture'],target_videos=1,per_class=1,excluded_sha256=[],excluded_youtube_ids=[],
                preprocessing=dict(min_duration_seconds=2,max_duration_seconds=16,
                    filter='fps=24,scale=576:320:flags=lanczos,setsar=1',codec='libx264',crf=18,preset='medium'))
            builder.save(root/'plan.json',p)
            result=builder.normalize(root,row,p)
            self.assertEqual(result['frames'],72)
            self.assertFalse(Path(result['path']).is_absolute())
            self.assertEqual(result,builder.normalize(root,row,p))
            builder.save(root/'manifest.json',dict(plan_sha256=builder.sha(root/'plan.json'),rows=[result]))
            self.assertEqual(builder.verify(root)['videos'],1)
            (root/result['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):builder.verify(root)

    def test_verify_rejects_reused_youtube_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            builder.save(root/'plan.json',dict(inputs={},target_videos=2,labels=['a','b'],per_class=1,
                excluded_youtube_ids=[],excluded_sha256=[]))
            rows=[dict(id=label,label=label,youtube_id='same-source',raw_sha256=label,sha256=label) for label in ('a','b')]
            builder.save(root/'manifest.json',dict(rows=rows))
            with self.assertRaisesRegex(ValueError,'duplicate youtube_id'):builder.verify(root)


if __name__=='__main__':unittest.main()
