"""Guard temporal leakage, matched budgets, and the user's strict output limit."""
import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_qwen35_context as context


class ContextProtocolTests(unittest.TestCase):
    def row(self, start=100, end=124, indices=None):
        return dict(segment=1, matched_pair=1, start=start, end_exclusive=end,
                    source_indices=indices or [102, 108, 114, 120])

    def test_context_is_causal_and_scene_bounded(self):
        row = self.row()
        pair = context.sample_pair(row, 24, [0, 83])
        self.assertEqual(pair["image_budget"], 16)
        self.assertEqual(len(pair["context_indices"]), 4)
        self.assertTrue(all(83 <= i < 100 for i in pair["context_indices"]))
        self.assertTrue(set(row["source_indices"]) <= set(pair["context_target_indices"]))
        self.assertEqual(len(pair["context_indices"])+len(pair["context_target_indices"]),
                         len(pair["control_target_indices"]))

    def test_cut_at_start_resets_context(self):
        pair = context.sample_pair(self.row(), 24, [0, 100])
        self.assertEqual(pair["context_indices"], [])
        self.assertEqual(pair["context_target_indices"], pair["control_target_indices"])

    def test_internal_cut_disables_prior_context(self):
        pair = context.sample_pair(self.row(), 24, [0, 112])
        self.assertTrue(pair["target_contains_detected_cut"])
        self.assertEqual(pair["context_indices"], [])

    def test_single_frame_has_no_motion_evidence_or_fake_duplicates(self):
        pair = context.sample_pair(self.row(10, 11, [10]*4), 24, [0])
        self.assertEqual(pair["context_indices"], [])
        self.assertEqual(pair["context_target_indices"], [10])

    def test_short_interval_preserves_all_original_samples(self):
        pair = context.sample_pair(self.row(10, 15, [10, 11, 12, 13]), 24, [0])
        self.assertEqual(len(pair["context_indices"]), 1)
        self.assertEqual(pair["context_target_indices"], [10, 11, 12, 13])
        self.assertEqual(len(pair["control_target_indices"]), 5)

    def test_context_does_not_exceed_two_seconds(self):
        pair = context.sample_pair(self.row(), 24, [0])
        self.assertEqual(pair["context_indices"][0], 52)
        self.assertEqual(pair["context_indices"][-1], 99)

    def test_format_limits(self):
        self.assertTrue(context.policy(" ".join(["word"]*79), False)["format_passed"])
        for caption, truncated in [(" ".join(["word"]*80), False), ("", False),
                                   ("A dog.", True), ("Possibly a dog.", False),
                                   ("A. B. C. D. E. F. G.", False)]:
            self.assertFalse(context.policy(caption, truncated)["format_passed"])


if __name__ == "__main__":
    unittest.main()
