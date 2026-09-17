import ast
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

from semantic_transmission.codec_transport import decoder_config_text
from semantic_transmission.packets import accounting
from semantic_transmission.temporal import (concatenate_segments, output_source_indices,
                                           resolve_concatenation_policy, unique_output_positions)
from semantic_transmission.transmission_accounting import packet_breakdown, channel_breakdown
from semantic_transmission.wire import pack

REPO = Path(__file__).resolve().parents[1]


class TransportLedgerTests(unittest.TestCase):
    def packet(self):
        header = {"video": dict(width=576, height=320, frames=9, fps=24),
                  "decoder": dict(policy="official_release", seed=42, steps=30),
                  "segments": [dict(path="clips/a/0.mp4", text='강아지 "둘"\n\\field', flow=1.25)],
                  "keyframes": [dict(index=0, average_power=0.123, complex_count=12,
                                     complex_offset=0, rate_bytes=2, rate_count=3)]}
        return header, pack(header, b"\x12\x30")

    def test_all_wire_bytes_accounted_including_unicode_escapes_crc_and_nibble_padding(self):
        header, packet = self.packet()
        ledger = packet_breakdown(packet)
        self.assertEqual(sum(ledger["components_bytes"].values()), len(packet))
        self.assertEqual(ledger["components_bytes"]["crc32"], 4)
        self.assertEqual(ledger["components_bytes"]["rate_index_payload"], 2)
        self.assertGreater(ledger["components_bytes"]["caption_json_values"], ledger["caption_utf8_bytes"])
        report = channel_breakdown(packet, 12, header["video"], accounting(len(packet)))
        bits = report["ldpc_bits"]
        self.assertEqual(bits["information"] + bits["padding"] + bits["parity"], bits["coded_total"])
        self.assertEqual(report["complex_channel_uses"]["total"], 12 + bits["coded_total"] // 4)
        self.assertAlmostEqual(report["cbr"]["total"], report["cbr"]["visual"] + report["cbr"]["digital"])
        self.assertFalse(report["paper_cbr_equivalence_claimed"])

    def test_missing_visual_values_or_wrong_padding_cannot_pass(self):
        header, packet = self.packet()
        report = accounting(len(packet))
        with self.assertRaisesRegex(ValueError, "visual channel"):
            channel_breakdown(packet, 11, header["video"], report)
        report["padding_bits"] += 1
        with self.assertRaisesRegex(ValueError, "padding_bits"):
            channel_breakdown(packet, 12, header["video"], report)


class ConcatenationTests(unittest.TestCase):
    def test_only_concatenation_changes_in_official_decoder_recipe(self):
        cfg = json.loads((REPO / "configs/lgvsc_variable.json").read_text())
        cfg["models"] = dict(stdit="a", vae="b", t5="c", vae2d="d")
        inputs = {"video": {k: cfg[k] for k in ("width", "height", "fps")},
                  "decoder": dict(seed=42, steps=30, policy="official_release")}
        before = {}; exec(decoder_config_text(cfg, REPO, inputs), before)
        inputs["decoder"]["concatenation_policy"] = "endpoint_exact"
        after = {}; exec(decoder_config_text(cfg, REPO, inputs), after)
        self.assertEqual(before.pop("concatenation_policy"), "official_release")
        self.assertEqual(after.pop("concatenation_policy"), "endpoint_exact")
        self.assertEqual(before, after)
        self.assertEqual(after["align"], 5)
        self.assertEqual(after["vae"]["micro_batch_size"], 4)
        self.assertTrue(after["model"]["enable_layernorm_kernel"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "tensor concatenation validation needs torch")
    def test_nonuniform_segments_remove_later_duplicates_without_changing_samples(self):
        import torch
        indices = [0, 1, 19, 239]
        segments = [torch.tensor(([999] * 17 if i else []) + list(range(a, b + 1))).reshape(1, -1, 1, 1)
                    for i, (a, b) in enumerate(zip(indices, indices[1:]))]
        original = [s.clone() for s in segments]
        official = concatenate_segments(segments, policy="official_release")
        exact = concatenate_segments(segments, policy="endpoint_exact")
        positions = unique_output_positions(indices)
        self.assertTrue(torch.equal(exact, official[:, positions]))
        self.assertEqual(exact.flatten().tolist(), list(range(240)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(segments, original)))
        with self.assertRaises(ValueError):
            resolve_concatenation_policy("official_release", "typo")

    def test_new_profiles_pin_stride_and_metric_contract(self):
        for name in ("etri_official", "lgvsc_variable", "lgvsc_webvid_select", "lgvsc_variable_endpoint_exact"):
            cfg = json.loads((REPO / f"configs/{name}.json").read_text())
            self.assertEqual(cfg["selection_stride"], 1)
            self.assertEqual(cfg["evaluation_profile"], "lgvsc_official_metrics_v1")
            self.assertEqual(cfg["decoder_policy"], "official_release")
        a = json.loads((REPO / "configs/lgvsc_variable.json").read_text())
        b = json.loads((REPO / "configs/lgvsc_variable_endpoint_exact.json").read_text())
        self.assertEqual({k for k in a if a[k] != b[k]}, {"profile", "concatenation_policy"})


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ("cv2", "numpy", "skimage")),
                     "pixel parity validation needs the evaluation environment")
class OfficialPixelMetricTests(unittest.TestCase):
    def test_pixels_match_upstream_file_functions_including_identical_frames(self):
        import cv2
        import numpy as np
        from skimage.metrics import structural_similarity
        from semantic_transmission.official_quality import pixel_scores
        tree = ast.parse((REPO / "06_evaluation/final_score.py").read_text())
        scope = dict(cv2=cv2, ssim=structural_similarity)
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in {"calculate_psnr", "calculate_ssim"}]
        exec(compile(ast.Module(body=functions, type_ignores=[]), "upstream_metrics", "exec"), scope)
        rng = np.random.default_rng(42)
        a = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as root:
            for b in (a.copy(), np.roll(a, 1, axis=1)):
                paths = [str(Path(root) / f"{i}.png") for i in range(2)]
                for p, rgb in zip(paths, (a, b)):
                    cv2.imwrite(p, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                scores = pixel_scores(a, b)
                self.assertEqual(scores["psnr_db"], scope["calculate_psnr"](*paths))
                self.assertEqual(scores["ssim"], scope["calculate_ssim"](*paths))


@unittest.skipUnless(os.environ.get("LGVSC_GPU_TESTS") == "1", "explicit GPU parity validation")
class OfficialGpuMetricTests(unittest.TestCase):
    def test_five_metrics_match_original_functions(self):
        import gc
        import cv2
        import clip
        import lpips
        import numpy as np
        import torch
        from PIL import Image
        import torchvision.transforms as T
        from DISTS_pytorch import DISTS
        from semantic_transmission.official_quality import OfficialMetrics
        from semantic_transmission.research_quality import read_frames
        run = REPO / "outputs/etri01_official_20260911_v2/01_person_walk"
        frames = read_frames(run / "receiver/reconstruction/sample_0000_frames")
        a = frames[[0, 120]]
        b = np.stack([a[0].copy(), np.roll(a[1], 5, axis=1)])
        _, rows = OfficialMetrics().evaluate(a, b)
        tree = ast.parse((REPO / "06_evaluation/final_score.py").read_text())
        scope = dict(torch=torch, Image=Image, lpips=lpips, T=T, device="cuda")
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in {"calculate_clip_similarity", "calculate_lpips", "calculate_dists"}]
        exec(compile(ast.Module(body=functions, type_ignores=[]), "upstream_metrics", "exec"), scope)
        with tempfile.TemporaryDirectory() as root:
            paths = []
            for i, (x, y) in enumerate(zip(a, b)):
                pair = [str(Path(root) / f"{i}_{j}.png") for j in range(2)]
                for p, rgb in zip(pair, (x, y)):
                    cv2.imwrite(p, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                paths.append(pair)
            for metric, key, function in [("lpips_vgg", "lpips_model", "calculate_lpips"),
                                           ("clip", "model", "calculate_clip_similarity"),
                                           ("dists", "dists_model", "calculate_dists")]:
                if metric == "lpips_vgg":
                    scope[key] = lpips.LPIPS(net="vgg").cuda().eval()
                elif metric == "clip":
                    scope[key], scope["preprocess"] = clip.load("ViT-B/32", device="cuda")
                    scope["cos"] = torch.nn.CosineSimilarity(dim=1)
                else:
                    scope[key] = DISTS().cuda().eval()
                for row, pair in zip(rows, paths):
                    self.assertAlmostEqual(row[metric], scope[function](*pair), delta=1e-6, msg=metric)
                del scope[key]
                gc.collect(); torch.cuda.empty_cache()
