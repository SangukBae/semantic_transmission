import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from semantic_transmission import sparc_webvid_evaluate as evaluation
from semantic_transmission.sparc_selector_data import save, sha
from test_sparc_selector_v3 import observation


class WebVidEvaluationTests(unittest.TestCase):
    def test_existing_mp4_pixels_and_timeline_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'prepared.mp4'
            frames=np.random.default_rng(4).integers(0,256,(7,16,32,3),dtype=np.uint8)
            subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','32x16',
                '-r','24','-i','-','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(path)],
                input=frames.tobytes(),check=True)
            row=dict(path=str(path),sha256=sha(path),frames=7,height=16,width=32)
            decoded=evaluation.decode_input(row)
            reference=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),
                '-vsync','0','-f','rawvideo','-pix_fmt','rgb24','-'])
            self.assertEqual(decoded.tobytes(),reference)
            self.assertEqual(decoded.shape,frames.shape)
            row['frames']=6
            with self.assertRaisesRegex(ValueError,'frame count'):
                evaluation.decode_input(row)
            row['sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'source changed'):
                evaluation.decode_input(row)

    def test_cache_receipt_detects_modified_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);f=folder/'result.json';save(f,{'a':1})
            hashes={f.name:sha(f)};evaluation.check_hashes(folder,hashes)
            save(f,{'a':2})
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                evaluation.check_hashes(folder,hashes)

    def test_cohort_allocation_and_infeasible_outputs_are_explicit(self):
        clips=[dict(id='a',pixels=100,baseline=observation([0,4],100,.01),
            observations=[observation([0,4],100,.01),observation([0,2,4],50,.02)])]
        cohort,picked=evaluation.output_choice(clips,.2)
        self.assertTrue(cohort['feasible']);self.assertEqual(picked[0]['channel']['complex_channel_uses'],50)
        cohort,picked=evaluation.output_choice(clips,.1)
        self.assertFalse(cohort['feasible']);self.assertEqual(picked[0]['channel']['complex_channel_uses'],50)

    def test_existing_published_video_is_never_silently_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.mp4';source.write_bytes(b'correct')
            dest=root/'out/video.mp4';evaluation.publish_video(source,dest)
            self.assertEqual(source.stat().st_ino,dest.stat().st_ino)
            dest.unlink();dest.write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError,'frozen choice'):
                evaluation.publish_video(source,dest)

    def test_stop_refuses_unrelated_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save(root/'launcher.json',dict(pid=123,start_ticks='99'))
            with mock.patch.object(evaluation.distributed,'active',return_value=True), \
                 mock.patch.object(Path,'read_bytes',return_value=b'python\0unrelated\0'), \
                 mock.patch.object(evaluation.os,'kill') as kill:
                with self.assertRaisesRegex(ValueError,'identity mismatch'):evaluation.stop(root)
                kill.assert_not_called()


if __name__=='__main__':unittest.main()
