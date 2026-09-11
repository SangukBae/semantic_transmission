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
    args = parser.parse_args()
    run = args.run_dir.resolve()
    cfg = json.loads((run / "run_config.json").read_text())
    video = next((run / "receiver/reconstruction").glob("*.mp4"))
    info = probe(video)
    if any(info[k] != cfg[k] for k in ("frames", "fps", "width", "height")):
        raise ValueError(f"output temporal contract mismatch: {info}")
    source = read_video(cfg["input"])
    metrics = Metrics()
    result = {"status": "PASSED", "video": info, "video_sha256": sha256(video),
              "source_sha256": sha256(cfg["input"]), "evaluation": "frame_mean_RGB_uint8_SSIM_gaussian11_LPIPS_Alex"}
    for boundary, values in (("lossless_frames", read_frames(video.with_suffix("").with_name(video.stem + "_frames"))),
                             ("delivered_mp4", read_video(video))):
        summary, rows = metrics.evaluate(source, values)
        result[boundary] = summary
        with (run / f"quality_{boundary}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(run / "quality.json", result)


if __name__ == "__main__":
    main()
