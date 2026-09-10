import csv
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from semantic_transmission.decoder_runner import run, split_rows
from semantic_transmission.packets import accounting, decode, encode
from semantic_transmission.temporal import segment_lengths, trim_prefix, conditioning_indices


class MetadataTests(unittest.TestCase):
    def test_unicode_and_csv_punctuation_survive_packet(self):
        rows = [{"path": "clips/video/0000.mp4", "text": '배, 자동차 "둘"\nnext line', "flow": 1.25}]
        self.assertEqual(decode(encode(rows)), rows)

    def test_damage_and_truncation_are_rejected(self):
        packet = encode([{"path": "clips/video/0.mp4", "text": "boat", "flow": 0}])
        for damaged in (packet[:-1], b"BAD!" + packet[4:], packet[:-1] + bytes([packet[-1] ^ 1])):
            with self.assertRaises(ValueError):
                decode(damaged)

    def test_sender_absolute_paths_and_nonfinite_flow_rejected(self):
        for path, flow in (("/sender/private/a.mp4", 0), ("../a.mp4", 0), ("a/b.mp4", float("nan"))):
            with self.assertRaises(ValueError):
                encode([{"path": path, "text": "boat", "flow": flow}])

    def test_channel_uses_include_short_final_ldpc_block(self):
        report = accounting(769)  # one bit beyond a full 6144-bit block, rounded to bytes
        self.assertEqual(report["blocks"], 2)
        self.assertEqual(report["complex_channel_uses"], 4608)
        self.assertEqual(report["padding_bits"], 6136)


class TemporalTests(unittest.TestCase):
    def test_short_and_long_segments_supply_valid_overlap(self):
        for count in (1, 5, 17, 30, 34):
            indices = conditioning_indices(count)
            self.assertEqual(len(indices), 17)
            self.assertEqual(indices[-1], count - 1)
            self.assertEqual(sorted(indices), indices)
            self.assertTrue(all(0 <= index < count for index in indices))
        self.assertEqual(conditioning_indices(5)[-5:], list(range(5)))
        self.assertEqual(conditioning_indices(30), list(range(13, 30)))

    def test_adjacent_segments_preserve_exact_source_frame_count(self):
        for indices in ([0, 16], [0, 16, 32], [0, 7, 22, 47]):
            lengths = segment_lengths(indices)
            emitted = sum(length - trim_prefix(i) for i, length in enumerate(lengths))
            self.assertEqual(emitted, indices[-1] + 1)

    def test_missing_start_or_repeated_keyframes_rejected(self):
        for indices in ([2, 16], [0, 0, 16], [0], [0, 16, 8]):
            with self.assertRaises(ValueError):
                segment_lengths(indices)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.csv = self.root / "input.csv"
        with self.csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["path", "text", "flow"])
            writer.writeheader()
            writer.writerows([
                {"path": "clips/first/0.mp4", "text": 'boats, "cars"\nwind', "flow": 1},
                {"path": "clips/second/0.mp4", "text": "other", "flow": 0},
                {"path": "clips/first/1.mp4", "text": "later", "flow": 2},
            ])

    def tearDown(self):
        self.temporary.cleanup()

    def test_quoted_multiline_csv_and_interleaved_videos(self):
        rows = split_rows(self.csv)
        self.assertEqual(len(rows["first"]), 2)
        self.assertEqual(rows["first"][0]["text"], 'boats, "cars"\nwind')

    def test_decoder_failure_is_recorded_and_propagated(self):
        decoder = self.root / "fail.py"
        decoder.write_text("raise SystemExit(37)\n")
        output = self.root / "output"
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            run(self.csv, output, "method", self.root, decoder=decoder, config=self.root / "config.py")
        self.assertEqual(caught.exception.returncode, 37)
        report = json.loads((output / "decoder_run.json").read_text())
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(len(report["videos"]), 1)
        self.assertTrue((output / "inputs/00000.csv").exists())

    def test_empty_success_is_failure_and_cannot_resume_as_success(self):
        decoder = self.root / "empty.py"
        decoder.write_text("pass\n")
        output = self.root / "output"
        with self.assertRaisesRegex(RuntimeError, "no video"):
            run(self.csv, output, "method", self.root, decoder=decoder, config=self.root / "config.py")
        with self.assertRaises(FileExistsError):
            run(self.csv, output, "method", self.root, decoder=decoder, config=self.root / "config.py")

    def test_preexisting_video_cannot_count_as_new_decoder_output(self):
        output = self.root / "stale"
        output.mkdir()
        (output / "first.mp4").write_bytes(b"stale")
        with self.assertRaises(FileExistsError):
            run(self.csv, output, "method", self.root, decoder=self.root / "unused.py", config=self.root / "config.py")


if __name__ == "__main__":
    unittest.main()
