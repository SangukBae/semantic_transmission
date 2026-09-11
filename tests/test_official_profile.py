import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from semantic_transmission.artifacts import write_json
from semantic_transmission.continue_etri import main
from semantic_transmission.temporal import output_source_indices, segment_lengths, trim_prefix


class OfficialProfileTests(unittest.TestCase):
    def test_published_concatenation_retains_shared_endpoints(self):
        indices = [0, 1, 19, 239]
        generated = []
        for i, (a, b) in enumerate(zip(indices, indices[1:])):
            segment = ([None] * 17 if i else []) + list(range(a, b + 1))
            self.assertEqual(len(segment), segment_lengths(indices)[i])
            generated.extend(segment[trim_prefix(i, policy="official_release"):])
        self.assertEqual(output_source_indices(indices, "official_release"), generated)
        self.assertEqual(len(generated), 242)
        self.assertEqual(generated.count(1), 2)
        self.assertEqual(generated.count(19), 2)
        self.assertEqual(output_source_indices(indices), list(range(240)))

    def test_one_segment_has_exact_reference_length(self):
        self.assertEqual(output_source_indices([0, 239], "official_release"), list(range(240)))
        with self.assertRaises(ValueError):
            output_source_indices([0, 99], "unknown")

    def test_continuation_uses_only_new_profile_history_and_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            profile = repo / "configs/etri_official.json"
            write_json(profile, {"profile": "official_test"})
            path = repo / ".local/etri_continue.json"
            write_json(path, {"input_dir": "/source", "run_history": ["/old_hq"],
                             "run_history_by_profile": {"official_test": ["/new_official"]}})
            original = path.read_bytes()
            with patch("semantic_transmission.continue_etri.repository", return_value=repo), \
                 patch("semantic_transmission.continue_etri.research") as run, \
                 patch("sys.argv", ["continue", "--dry-run"]):
                main()
            args = run.call_args.args[0]
            self.assertIn(str(profile), args)
            self.assertIn("/new_official", args)
            self.assertNotIn("/old_hq", args)
            self.assertEqual(original, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
