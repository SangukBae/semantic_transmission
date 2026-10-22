import csv
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from semantic_transmission import sparc_kinetics_evaluate as evaluation
from semantic_transmission.sparc_selector_data import save, sha
from semantic_transmission.official_quality import KEYS, PROFILE


class KineticsComparisonTests(unittest.TestCase):
    def test_source_hash_overlap_is_rejected_before_any_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'preprocess_records').mkdir()
            manifest=root/'manifest.csv'
            manifest.write_text('filename,kinetics_label_id,sha256\na.mp4,0,training-source\n')
            save(root/'preprocess_records/a.mp4.json',dict(sha256='processed-source'))
            with self.assertRaisesRegex(ValueError,'overlaps'):
                evaluation.source_rows(root,manifest,[dict(sha256='training-source')])

    def test_final_metrics_resume_aggregates_real_receipts_and_detects_movie_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[]
            for i in range(2):
                rid=f'kinetics_{i}';record=dict(id=rid)
                for role,base in [('baseline',1.),('selected',2.)]:
                    movie=root/f'{rid}_{role}.mp4';movie.write_bytes((rid+role).encode())
                    record[role+'_video']=str(movie);record[role+'_video_sha256']=sha(movie)
                    save(root/'quality'/rid/(role+'.json'),dict(
                        identity=dict(protocol='p',source_sha256=rid,movie_sha256=sha(movie),profile=PROFILE),
                        summary={key:base+i for key in KEYS}))
                rows.append(record)
            with (root/'all14_per_video.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            save(root/'all14_summary.json',dict(frames=20,budget_feasible=True,
                baseline=dict(channel_uses=100,aggregate_cbr=.01),selected=dict(channel_uses=80,aggregate_cbr=.008)))
            p=dict(signature='p',target_cbr=.01,evaluation_mode='test',rows=[dict(id=f'kinetics_{i}',sha256=f'kinetics_{i}') for i in range(2)])
            with mock.patch.object(evaluation.shared,'decode_input',side_effect=AssertionError('should reuse verified metric receipt')):
                result=evaluation.final_metrics(root,p,lambda *a,**k:None)
            self.assertEqual(result['baseline']['quality']['psnr_db'],1.5)
            self.assertEqual(result['selected']['quality']['psnr_db'],2.5)
            self.assertAlmostEqual(result['channel_reduction_percent'],20.)
            Path(rows[0]['selected_video']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'reconstruction changed'):
                evaluation.final_metrics(root,p,lambda *a,**k:None)

    def test_stop_never_signals_unrelated_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save(root/'launcher.json',dict(pid=123,start_ticks='5'))
            with mock.patch.object(evaluation.distributed,'active',return_value=True), \
                 mock.patch.object(Path,'read_bytes',return_value=b'python\0other\0'), \
                 mock.patch.object(evaluation.os,'kill') as kill:
                with self.assertRaisesRegex(ValueError,'identity mismatch'):
                    evaluation.stop(root)
                kill.assert_not_called()


if __name__=='__main__':unittest.main()
