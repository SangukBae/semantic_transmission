"""Second, independently seeded intervention corpus. Contains no metric calls."""
import cv2
import numpy as np

FPS, HEIGHT, WIDTH, COUNT = 8., 192, 320, 32
SEVERITIES = (.125, .25, .5)


def nuisance_controls(frames):
    yield "identity", frames.copy()
    yield "brightness20", np.clip(frames.astype(np.int16) + 20, 0, 255).astype(np.uint8)
    yield "gamma075", np.rint(255 * (frames / 255.) ** .75).astype(np.uint8)
    out = []
    for frame in frames:
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            raise ValueError("JPEG encode failed")
        out.append(cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    yield "jpeg75", np.stack(out)


def temporal_variant(frames, kind, severity):
    n = len(frames)
    k = max(2, round(n * severity))
    start = (n - k) // 2
    index = np.arange(n)
    if kind == "reverse":
        index[start:start + k] = index[start:start + k][::-1]
    elif kind == "freeze":
        index[start:start + k] = index[start]
    elif kind == "lag":
        index = np.maximum(index - max(1, round(severity * n / 2)), 0)
    elif kind == "swap":
        index[start:start + k] = np.roll(index[start:start + k], k // 2)
    else:
        raise ValueError(kind)
    return frames[index], {"source_index_map": index.tolist(), "requested_interval": [start, start + k]}


def render_scene(seed, kind=None, severity=0., appearance=0, count=COUNT):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    base = rng.integers(25, 70, 3)
    colors = rng.integers(110, 225, (4, 3))
    radius = int(rng.integers(12, 22))
    phase = float(rng.uniform(-.25, .25))
    length = max(1, round(count * severity))
    begin = (count - length) // 2
    frames, labels, centers = [], [], []
    for t in range(count):
        # Appearance controls preserve masks, object color means and trajectories.
        bg = np.broadcast_to(base, (HEIGHT, WIDTH, 3)).astype(float).copy()
        if appearance:
            texture = 24 * np.sin(xx * .21 + seed) * np.cos(yy * .19)
        else:
            texture = 8 * np.sin(xx * .11) * np.cos(yy * .09)
        frame = np.clip(bg + texture[..., None], 0, 255).astype(np.uint8)
        ids = np.zeros((HEIGHT, WIDTH), np.uint8)
        active = kind is not None and begin <= t < begin + length
        mt = t
        if active and kind == "motion_reverse":
            mt = begin + length - 1 - (t - begin)
        if active and kind == "motion_freeze":
            mt = begin
        # A smooth direction reversal, plus a stationary birth/death object.
        progress = .5 - .5 * np.cos(2 * np.pi * mt / (count - 1) + phase)
        poses = [(int(48 + 214 * progress), 51), (int(263 - 208 * t / (count - 1)), 127), (160, 166), (278, 168)]
        visible = [not (active and kind == "object_omission"), True,
                   count // 4 <= t < 3 * count // 4, active and kind == "object_addition"]
        for obj, ((cx, cy), show) in enumerate(zip(poses, visible), 1):
            if not show:
                continue
            mask = np.zeros_like(ids)
            if obj == 1:
                axes = (radius * 2, max(4, radius // 2)) if active and kind == "object_shape" else (radius, radius)
                cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1, -1)
            elif obj == 2:
                cv2.rectangle(mask, (cx - radius, cy - 12), (cx + radius, cy + 12), 1, -1)
            elif obj == 3:
                cv2.fillConvexPoly(mask, np.array([[cx, cy - 15], [cx - 18, cy + 12], [cx + 18, cy + 12]]), 1)
            else:
                cv2.circle(mask, (cx, cy), 14, 1, -1)
            # Texture in object coordinates travels with the object.
            tex = (16 if appearance == 2 else 6) * np.sin((xx - cx) * (.65 if appearance == 2 else .35))
            pixels = np.clip(colors[obj - 1] + tex[..., None], 0, 255).astype(np.uint8)
            frame[mask > 0] = pixels[mask > 0]
            ids[mask > 0] = obj
        frames.append(frame)
        labels.append(ids)
        centers.append(poses)
    return np.stack(frames), {"masks": np.stack(labels), "centers": np.asarray(centers),
                            "active_interval": [begin, begin + length] if kind else None}


def variants(frames, seed=None):
    for name, rec in nuisance_controls(frames):
        yield name, rec, {"kind": "control", "severity": 0., "target": "control"}
    if seed is not None:
        for appearance in (1, 2):
            rec, _ = render_scene(seed, appearance=appearance)
            yield f"texture{appearance}", rec, {"kind": "control", "severity": 0., "target": "control"}
    for kind in ("reverse", "freeze", "lag", "swap"):
        for severity in SEVERITIES:
            rec, truth = temporal_variant(frames, kind, severity)
            yield f"{kind}_{severity}", rec, {"kind": kind, "severity": severity, "target": "motion", **truth}
    if seed is not None:
        for kind in ("motion_reverse", "motion_freeze", "object_addition", "object_omission", "object_shape"):
            for severity in SEVERITIES:
                rec, truth = render_scene(seed, kind, severity)
                yield f"{kind}_{severity}", rec, {"kind": kind, "severity": severity,
                        "target": "object" if kind.startswith("object") else "motion",
                        "active_interval": truth["active_interval"]}
