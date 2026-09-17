"""Deterministic interventions and renderer truth, independent of the metric code."""
import cv2
import numpy as np


def controls(frames):
    yield "identity", frames.copy(), {"kind": "control", "severity": 0}
    yield "brightness", np.clip(frames.astype(int) + 5, 0, 255).astype(np.uint8), {"kind": "control", "severity": 0}
    yield "blur", np.stack([cv2.GaussianBlur(f, (3, 3), 0.5) for f in frames]), {"kind": "control", "severity": 0}
    jpeg = []
    for f in frames:
        ok, data = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("JPEG control failed")
        jpeg.append(cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    yield "jpeg90", np.stack(jpeg), {"kind": "control", "severity": 0}


def temporal_errors(frames):
    n = len(frames)
    for kind in ("reverse", "freeze", "lag", "swap"):
        for severity in (0.25, 0.5, 1.0):
            mapping = np.arange(n)
            k = max(2, round(n * severity))
            if kind == "reverse":
                mapping[:k] = mapping[:k][::-1]
            elif kind == "freeze":
                mapping[:k] = 0
            elif kind == "lag":
                mapping = np.maximum(mapping - max(1, round(severity * n / 3)), 0)
            else:
                mapping[:k] = np.roll(mapping[:k], k // 2)
            rec = frames[mapping]
            changed = np.flatnonzero(np.any(rec != frames, axis=(1, 2, 3))).tolist()
            yield f"{kind}_{severity}", rec, {"kind": kind, "severity": severity,
                 "source_index_map": mapping.tolist(), "changed_frames": changed,
                 "observable_pixel_change": bool(changed)}


def scene(seed, count=12, kind=None, severity=0):
    rng = np.random.default_rng(seed)
    bg = rng.integers(28, 65, size=3).tolist()
    colors = [rng.integers(120, 250, size=3).tolist() for _ in range(3)]
    y = int(rng.integers(32, 90))
    radius = int(rng.integers(9, 16))
    reverse = bool(seed % 2)
    k = max(1, round(count * severity)) if kind else 0
    start = (count - k) // 2
    frames, records = [], []
    for t in range(count):
        f = np.full((128, 224, 3), bg, dtype=np.uint8)
        # Background texture and anti-aliased edges also exist in controls.
        for gx in range(0, 224, 28):
            cv2.line(f, (gx, 0), (gx, 127), tuple(v + 8 for v in bg), 1)
        x = round(25 + 165 * (t / (count - 1) if not reverse else 1 - t / (count - 1)))
        active = start <= t < start + k
        visible = not (kind == "object_omission" and active)
        distorted = kind == "object_shape" and active
        if visible:
            axes = (radius * 2, max(3, radius // 2)) if distorted else (radius, radius)
            cv2.ellipse(f, (x, y), axes, 0, 0, 360, colors[0], -1, cv2.LINE_AA)
        # A legitimate birth/death event, present in both source and controls.
        event_visible = count // 4 <= t < count * 3 // 4
        if event_visible:
            cv2.rectangle(f, (85, 98), (113, 120), colors[1], -1, cv2.LINE_AA)
        added = kind == "object_addition" and active
        if added:
            cv2.fillConvexPoly(f, np.array([[174, 93], [197, 122], [153, 122]]), colors[2], cv2.LINE_AA)
        frames.append(f)
        records.append({"frame": t, "moving_object": {"visible": visible, "center": [x, y], "distorted": distorted},
                        "event_object_visible": event_visible, "additional_object_visible": added})
    return np.stack(frames), records


def object_errors(seed, count=12):
    for kind in ("object_addition", "object_omission", "object_shape"):
        for severity in (0.25, 0.5, 1.0):
            frames, truth = scene(seed, count, kind, severity)
            yield f"{kind}_{severity}", frames, {"kind": kind, "severity": severity, "renderer_truth": truth,
                                                "observable_pixel_change": True}
