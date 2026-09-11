import copy
import json
from pathlib import Path
import tempfile
import unittest

from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.resume import STAGES, completed_runs, copy_completed


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "old"
        self.run = self.root / "01_test"
        self.cfg = dict(width=512, height=256, frames=2, fps=10, steps=50)
        self.source = {"id": "01_test", "path": "/source.mp4", "sha256": "source_hash"}
        self.record = {"id": "01_test", "status": "PASSED",
                       "stages": [{"stage": s, "status": "PASSED", "returncode": 0} for s in STAGES]}
        self.manifest = {"status": "FAILED", "code": {"commit": "original", "dirty": False},
                         "inputs": [self.source], "runs": [self.record]}
        write_json(self.root / "profile.json", self.cfg)
        write_json(self.root / "batch_manifest.json", self.manifest)
        write_json(self.run / "run_manifest.json", self.record)
        write_json(self.run / "run_config.json", dict(self.cfg, input=self.source["path"]))
        video = self.run / "receiver/reconstruction/sample_0000.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"verified-video-fixture")
        frames = video.parent / "sample_0000_frames"
        frames.mkdir()
        for i in range(2):
            (frames / f"{i:05d}.png").write_bytes(b"frame-fixture")
        write_json(self.run / "quality.json", {"status": "PASSED", "video": self.cfg,
                   "source_sha256": self.source["sha256"], "video_sha256": sha256(video)})
        tx = self.run / "transmitter"
        tx.mkdir()
        for name in ("metadata.bin", "visual.c64"):
            (tx / name).write_bytes(name.encode())
        write_json(self.run / "sender_accounting.json", {"transmitter_files": {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in tx.iterdir()}})
        for name in ("channel_accounting.json", "receiver_accounting.json"):
            write_json(self.run / name, {"status": "PASSED"})

    def reuse(self):
        return completed_runs([self.root], self.cfg, [self.source])

    def test_completed_video_survives_interrupted_batch_and_keeps_execution_provenance(self):
        item = self.reuse()["01_test"]
        copied = copy_completed(item, Path(self.temp.name) / "new")
        self.assertEqual(copied["execution_code"]["commit"], "original")
        self.assertEqual(copied["reused_from"], str(self.run))
        self.assertNotIn("reused_from", json.loads((self.run / "run_manifest.json").read_text()))

    def test_partial_video_is_regenerated_even_when_artifacts_exist(self):
        self.manifest["runs"][0]["status"] = "FAILED"
        write_json(self.root / "batch_manifest.json", self.manifest)
        self.assertEqual(self.reuse(), {})

    def test_modified_payload_video_or_settings_cannot_count_as_verified(self):
        for filename in ("transmitter/metadata.bin", "receiver/reconstruction/sample_0000.mp4"):
            with self.subTest(filename=filename):
                path = self.run / filename
                original = path.read_bytes()
                path.write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    self.reuse()
                path.write_bytes(original)
        cfg = copy.deepcopy(self.cfg)
        cfg["steps"] = 10
        with self.assertRaises(ValueError):
            completed_runs([self.root], cfg, [self.source])

    def test_success_label_cannot_hide_failed_stage(self):
        self.record["stages"][1]["returncode"] = 1
        write_json(self.run / "run_manifest.json", self.record)
        with self.assertRaises(ValueError):
            self.reuse()


if __name__ == "__main__":
    unittest.main()
