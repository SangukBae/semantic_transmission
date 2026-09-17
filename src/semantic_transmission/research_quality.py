"""Common source-paired metrics for lossless reconstruction frames and MP4 output."""
import argparse
import csv
import json
import math
from pathlib import Path

from .artifacts import sha256, write_json
from .video_io import probe


def read_video(path):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"no video frames: {path}")
    return np.stack(frames)


def read_frames(directory):
    import numpy as np
    from PIL import Image
    paths = sorted(Path(directory).glob("*.png"))
    if not paths:
        raise ValueError(f"no PNG reconstruction frames: {directory}")
    return np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])


class Metrics:
    def __init__(self):
        import lpips
        self.lpips = lpips.LPIPS(net="alex").cuda().eval()

    def evaluate(self, source, reconstructed):
        import numpy as np
        import torch
        from skimage.metrics import structural_similarity
        if source.shape != reconstructed.shape:
            raise ValueError(f"source/reconstruction shape mismatch: {source.shape} != {reconstructed.shape}")
        if source.dtype != np.uint8 or reconstructed.dtype != np.uint8:
            raise ValueError("quality inputs must be decoded RGB uint8")
        rows = []
        for i, (a, b) in enumerate(zip(source, reconstructed)):
            mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
            rows.append({"frame": i, "psnr_db": 10 * math.log10(255**2 / max(mse, 1e-12)),
                "ssim": float(structural_similarity(a, b, channel_axis=-1, data_range=255,
                        gaussian_weights=True, sigma=1.5, use_sample_covariance=False))})
        with torch.inference_mode():
            for i in range(0, len(source), 8):
                a = torch.from_numpy(source[i:i+8].copy()).permute(0, 3, 1, 2).cuda().float() / 127.5 - 1
                b = torch.from_numpy(reconstructed[i:i+8].copy()).permute(0, 3, 1, 2).cuda().float() / 127.5 - 1
                for offset, value in enumerate(self.lpips(a, b).flatten().cpu().tolist()):
                    rows[i + offset]["lpips_alex"] = value
        summary = {key: float(np.mean([r[key] for r in rows])) for key in ("psnr_db", "ssim", "lpips_alex")}
        if not all(math.isfinite(v) for v in summary.values()):
            raise ValueError("nonfinite quality metric")
        summary.update(frames=len(source), shape=list(source.shape))
        return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["evaluate"])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, help="new directory for reevaluation; preserves old results")
    parser.add_argument("--evaluation-profile", choices=["legacy_hq_v1", "lgvsc_official_metrics_v1"])
    parser.add_argument("--compare-concatenation", action="store_true",
                        help="also score a view without repeated shared endpoints; no model rerun")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    cfg = json.loads((run / "run_config.json").read_text())
    destination = args.output_dir.resolve() if args.output_dir else run
    if args.output_dir:
        destination.mkdir(parents=True, exist_ok=False)
    elif (run / "quality.json").exists():
        parser.error("quality.json already exists; use --output-dir to preserve historical results")
    videos = list((run / "receiver/reconstruction").glob("*.mp4"))
    if len(videos) != 1:
        raise ValueError("evaluation requires exactly one reconstructed MP4")
    video = videos[0]
    info = probe(video)
    from .temporal import output_source_indices, resolve_concatenation_policy, unique_output_positions
    inputs = json.loads((run / "receiver/decoder_inputs.json").read_text())
    policy = inputs["decoder"].get("policy", "endpoint_exact")
    concatenation = resolve_concatenation_policy(policy, inputs["decoder"].get("concatenation_policy"))
    mapping = output_source_indices(inputs["indices"], concatenation)
    if info["frames"] != len(mapping) or any(info[k] != cfg[k] for k in ("fps", "width", "height")):
        raise ValueError(f"output temporal contract mismatch: {info}")
    reference = run / "data/normalized.mp4"
    source = read_video(reference)
    if len(source) != cfg["frames"] or max(mapping) != len(source) - 1:
        raise ValueError("normalized reference timeline differs from the profile")
    source = source[mapping]
    from .official_quality import OfficialMetrics, PROFILE, summarize
    evaluation = args.evaluation_profile or cfg.get("evaluation_profile", "legacy_hq_v1")
    if evaluation not in {PROFILE, "legacy_hq_v1"}:
        raise ValueError(f"unknown evaluation profile: {evaluation}")
    metrics = OfficialMetrics() if evaluation == PROFILE else Metrics()
    if args.compare_concatenation and evaluation != PROFILE:
        raise ValueError("concatenation comparison requires official metrics")
    # Old inputs may have moved disks; the run's recorded source hash remains
    # provenance, never substitute the normalized reference's hash for it.
    historical = run / "quality.json"
    if Path(cfg["input"]).is_file():
        source_hash = sha256(cfg["input"])
    elif historical.is_file():
        source_hash = json.loads(historical.read_text())["source_sha256"]
    else:
        raise FileNotFoundError(cfg["input"])
    if historical.is_file():
        old = json.loads(historical.read_text())
        if old["video_sha256"] != sha256(video) or old["source_sha256"] != source_hash:
            raise ValueError("historical video/source changed before reevaluation")
        if "reference_sha256" in old and old["reference_sha256"] != sha256(reference):
            raise ValueError("historical normalized reference changed before reevaluation")
    result = {"status": "PASSED", "video": info, "video_sha256": sha256(video),
              "source_sha256": source_hash, "source_file_verified_now": Path(cfg["input"]).is_file(),
              "reference_sha256": sha256(reference),
              "reference": str(reference), "output_source_indices": mapping,
              "decoder_policy": policy,
              "concatenation_policy": concatenation, "evaluation_profile": evaluation,
              "evaluation": ("frame_mean_PSNR_SSIM_gray_LPIPS_VGG_CLIP_ViTB32_cosine01_DISTS"
                             if evaluation == PROFILE else "frame_mean_RGB_uint8_SSIM_gaussian11_LPIPS_Alex")}
    if historical.is_file():
        result["reevaluated_from_quality_sha256"] = sha256(historical)
    for boundary in ("lossless_frames", "delivered_mp4"):
        values = (read_frames(video.with_suffix("").with_name(video.stem + "_frames"))
                  if boundary == "lossless_frames" else read_video(video))
        summary, rows = metrics.evaluate(source, values)
        result[boundary] = summary
        if evaluation == PROFILE:
            result.setdefault("evaluation_resources", {})[boundary] = metrics.resources
        if args.compare_concatenation:
            positions = unique_output_positions(inputs["indices"], concatenation)
            selected = [dict(rows[i], frame=j, generated_frame=i, source_frame=mapping[i])
                        for j, i in enumerate(positions)]
            comparison = result.setdefault("endpoint_exact_view", {
                "scope": "same generated pixels, later duplicate boundaries removed; no generation rerun or MP4 reencode",
                "kept_generated_indices": positions, "output_source_indices": [mapping[i] for i in positions]})
            comparison[boundary] = summarize(selected, (len(selected), *values.shape[1:]))
            with (destination / f"quality_endpoint_exact_{boundary}.csv").open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(selected[0])); writer.writeheader(); writer.writerows(selected)
        with (destination / f"quality_{boundary}.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        del values
    write_json(destination / "quality.json", result)


if __name__ == "__main__":
    main()
