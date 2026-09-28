"""Frame-exact semantic intervals and CPU validation before model loading.

The paper intervals are half-open [start, end). A one-frame interval is valid;
it must not be rounded to hundredths of a second or copied between H.264 GOPs.
"""
import json
import math
from pathlib import Path
import subprocess

from .artifacts import sha256, write_json
from .webvid5 import fingerprint


def sample_clip_frames(path, requested):
    """Match official end-clamping without fragile repeated OpenCV seeking.

    A one-frame H.264 clip can fail seeking back to zero after its first read.
    Upstream substitutes black frames on that error; decode once and reuse the
    actual last frame instead, so motion is never inferred from fake pixels.
    """
    import cv2
    from PIL import Image
    if not requested or any(type(i) is not int or i < 0 for i in requested):
        raise ValueError("requested clip frame indices must be nonnegative integers")
    capture = cv2.VideoCapture(str(path))
    decoded = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    if not decoded:
        raise ValueError(f"flow clip has no decoded frames: {path}")
    sampled = [min(i, len(decoded) - 1) for i in requested]
    return [decoded[i] for i in sampled], sampled


def validate_indices(indices, frame_count):
    if (len(indices) < 2 or any(type(i) is not int for i in indices)
            or indices != sorted(set(indices)) or indices[0] != 0
            or indices[-1] != frame_count - 1):
        raise ValueError("semantic intervals must cover ordered source endpoints")


def inspect_clip(path, frames, start, end, fps):
    """Check actual decoded pixels and every timestamp, not container estimates."""
    import cv2
    import numpy as np
    payload = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path)], text=True))
    times = [float(row["best_effort_timestamp_time"]) for row in payload.get("frames", [])]
    expected = end - start
    if len(times) != expected or any(not math.isfinite(t) or abs(t - i / fps) > 1e-6
                                     for i, t in enumerate(times)):
        raise ValueError(f"semantic clip frame count/PTS mismatch: {path}: {len(times)} != {expected}")
    cap = cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            ok, actual = cap.read()
            if not ok:
                break
            source = cv2.imread(str(frames / f"{start + count}.png"))
            if count >= expected or source is None or not np.array_equal(actual, source):
                raise ValueError(f"semantic clip pixel mismatch: {path}, source frame {start + count}")
            count += 1
    finally:
        cap.release()
    if count != expected:
        raise ValueError(f"semantic clip decoded frame count mismatch: {path}: {count} != {expected}")
    return {"start_frame": start, "end_frame_exclusive": end, "frames": count,
            "sha256": sha256(path), "pts_verified": True, "pixels_exact": True}


def prepare_frame_exact_clips(cfg, run):
    """Build missing clips, refuse stale/corrupt artifacts, and audit every clip."""
    run = Path(run)
    fps = cfg["fps"]
    if not isinstance(fps, (float, int)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("semantic clip fps must be positive")
    indices = json.loads((run / "keyframes.json").read_text())["indices"]
    validate_indices(indices, cfg["frames"])
    normalized = run / "data/normalized.mp4"
    frames = run / "data/frames/sample"
    source = {"normalized_sha256": sha256(normalized), "keyframes": indices,
              "fps": fps, "frames": cfg["frames"],
              "source_frame_sha256": [sha256(frames / f"{i}.png") for i in range(cfg["frames"])]}
    identity = fingerprint(source)
    audit_path = run / "semantic_clips_audit.json"
    prior = None
    if audit_path.exists():
        prior = json.loads(audit_path.read_text())
        checksum = prior.pop("checksum", None)
        if (checksum != fingerprint(prior) or prior.get("identity") != identity
                or prior.get("policy") != "frame_exact" or prior.get("version") != 1):
            raise ValueError("semantic clip audit identity/checksum mismatch")
    directory = run / "data/clips/sample"
    directory.mkdir(parents=True, exist_ok=True)
    paths, records = [], []
    for segment, (start, end) in enumerate(zip(indices, indices[1:])):
        path = directory / f"{segment:05d}.mp4"
        if not path.exists():
            temporary = directory / f".{segment:05d}.partial.mp4"
            try:
                subprocess.run([
                    "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(normalized),
                    "-vf", f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS",
                    "-an", "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p",
                    "-threads", "2", str(temporary)], check=True)
                inspect_clip(temporary, frames, start, end, fps)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        record = {"segment": segment, "path": str(path.relative_to(run)),
                  **inspect_clip(path, frames, start, end, fps)}
        paths.append(path)
        records.append(record)
    payload = {"version": 1, "policy": "frame_exact", "identity": identity,
               "source": source, "segments": len(records), "clips": records,
               "status": "PASSED", "interval_convention": "[start, end)"}
    if prior is not None and prior != payload:
        raise ValueError("semantic clip audit content mismatch")
    if prior is None:
        write_json(audit_path, dict(payload, checksum=fingerprint(payload)))
    return paths
