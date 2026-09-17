"""Model-independent, hash-checked evaluation with explicit physical timestamps."""
import hashlib
from pathlib import Path
import re

import numpy as np

from .artifacts import sha256
from .research_quality import read_video
from .video_io import probe


def sampled_video(path, count=12, width=224, height=128, sample_fps=None):
    import cv2
    info = probe(path)
    if sample_fps is None:
        indices = np.unique(np.rint(np.linspace(0, info["frames"] - 1, count)).astype(int))
    else:
        if not np.isfinite(sample_fps) or sample_fps <= 0 or sample_fps > info["fps"]:
            raise ValueError("sample FPS must be positive and no higher than source FPS")
        times = np.arange(int(np.floor((info["frames"] - 1) / info["fps"] * sample_fps)) + 1) / sample_fps
        indices = np.unique(np.floor(times * info["fps"] + .5).astype(int))
    cap = cv2.VideoCapture(str(path))
    frames = []
    wanted = set(indices.tolist())
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (width, height), interpolation=cv2.INTER_AREA))
        i += 1
    cap.release()
    if i != info["frames"] or len(frames) != len(indices):
        raise ValueError(f"incomplete decode: {path}: {i}/{info['frames']}")
    return np.stack(frames), indices, info


def load_pair(row, count=12, width=224, height=128, sample_fps=None):
    import cv2
    source_path = Path(row["source"])
    if sha256(source_path) != row["source_sha256"]:
        raise ValueError("source SHA256 changed")
    source, indices, source_info = sampled_video(source_path, count, width, height, sample_fps)
    rec_path = Path(row["reconstruction"])
    if rec_path.is_dir():
        from PIL import Image
        paths = list(rec_path.glob("*.png"))
        if not paths:
            raise ValueError("empty reconstruction directory")
        numbered = []
        for path in paths:
            match = re.search(r"(\d+)$", path.stem)
            if match is None:
                raise ValueError("reconstruction frames require numeric filename suffixes")
            numbered.append((int(match.group(1)), path))
        numbered.sort()
        numbers = [n for n, _ in numbered]
        if numbers[0] not in (0, 1) or numbers != list(range(numbers[0], numbers[0] + len(numbers))):
            raise ValueError("reconstruction frame numbers contain a gap or duplicate")
        paths = [path for _, path in numbered]
        digest = hashlib.sha256("".join(p.name + sha256(p) for p in paths).encode()).hexdigest()
        rec = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])
        fps = float(row["reconstruction_fps"])
    else:
        digest = sha256(rec_path)
        rec = read_video(rec_path)
        fps = probe(rec_path)["fps"]
    if digest != row["reconstruction_sha256"]:
        raise ValueError("reconstruction SHA256 changed")
    if fps <= 0:
        raise ValueError("invalid reconstruction FPS")
    # No visual-content matching or duration rescaling: delays remain in the score.
    times = indices / source_info["fps"]
    rec_indices = np.floor(times * fps + 0.5).astype(int)
    valid = rec_indices < len(rec)
    if valid.sum() < 2:
        raise ValueError("fewer than two overlapping physical timestamps")
    values = np.stack([cv2.resize(rec[j], (width, height), interpolation=cv2.INTER_AREA) for j in rec_indices[valid]])
    provenance = {"source_indices": indices[valid].tolist(), "reconstruction_indices": rec_indices[valid].tolist(),
                  "sample_coverage": float(valid.mean()), "missing_sample_count": int((~valid).sum()),
                  "max_timestamp_offset_s": float(np.max(np.abs(rec_indices[valid] / fps - times[valid]))),
                  "source_duration_s": source_info["frames"] / source_info["fps"],
                  "reconstruction_duration_s": len(rec) / fps,
                  "missing_source_tail_s": max(0.0, source_info["frames"] / source_info["fps"] - len(rec) / fps),
                  "extra_reconstruction_tail_s": max(0.0, len(rec) / fps - source_info["frames"] / source_info["fps"]),
                  "alignment": "nearest physical timestamp; no duplicate removal or content/time warping",
                  "requested_sample_fps": sample_fps,
                  "resize": [width, height], "reconstruction_sha256": digest}
    return source[valid], values, provenance
