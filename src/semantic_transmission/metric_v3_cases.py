"""Independent simulator and annotation-only truth audit for ERE/STA.

No candidate metric is called here. Source event truth uses unsmoothed scene
kinematics, whereas the RGB candidate uses five-frame smoothed flow regions.
"""
import math

import cv2
import numpy as np

from .metric_v2_cases import nuisance_controls, temporal_variant

FPS, HEIGHT, WIDTH, COUNT = 8., 192, 320, 32
SEVERITIES = (.125, .25, .5)
RESTYLES = ("restyle_texture", "restyle_palette", "restyle_lighting", "restyle_camera", "restyle_instance", "restyle_gait")


def annotation_events(centers, visible, fps=FPS, shape=(HEIGHT, WIDTH)):
    """Ground-truth events from identity-preserving, unfiltered kinematics."""
    events = []
    h, w = shape
    for k in range(centers.shape[1]):
        candidates = []
        speed = np.linalg.norm(np.diff(centers[:, k], axis=0), axis=1)
        vectors = np.diff(centers[:, k], axis=0)
        for t in range(1, len(centers)):
            if visible[t, k] and not visible[t - 1, k]:
                vector = vectors[t] if t < len(vectors) and visible[t + 1, k] else np.zeros(2)
                candidates.append(("enter", t, centers[t, k], vector))
            elif visible[t - 1, k] and not visible[t, k]:
                vector = vectors[t - 2] if t >= 2 else np.zeros(2)
                candidates.append(("exit", t, centers[t - 1, k], vector))
            elif 0 < t < len(vectors) and visible[t - 1:t + 2, k].all():
                a, b = vectors[t - 1], vectors[t]
                na, nb = speed[t - 1], speed[t]
                kind = None
                if na < .5 <= nb:
                    kind = "start"
                elif nb < .5 <= na:
                    kind = "stop"
                elif min(na, nb) >= .5 and np.dot(a, b) < 0:
                    kind = "turn"
                if kind:
                    candidates.append((kind, t, centers[t, k], a if kind == "stop" else b))
        last = -math.inf
        for kind, t, center, vector in candidates:
            if t / fps - last < .375 - 1e-9:
                continue
            angle = math.atan2(float(vector[1]), float(vector[0])) % (2 * math.pi)
            events.append({"type": kind, "time_s": t / fps,
                           "direction": int(math.floor(angle / (math.pi / 4) + .5)) % 8,
                           "cell": np.clip(np.floor(center / [w, h] * 3), 0, 2).astype(int).tolist(),
                           "object_id": k + 1})
            last = t / fps
    return sorted(events, key=lambda e: (e["time_s"], e["object_id"], e["type"]))


def event_signature(events):
    return [(e["object_id"], e["type"], e["time_s"], e["direction"]) for e in events]


def truth_event_change(a, b):
    # Match using true object IDs, independently of candidate grid/track matching.
    from scipy.optimize import linear_sum_assignment
    cost = np.full((len(a), len(b)), 1e6)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            dd = abs(x["direction"] - y["direction"])
            dd = min(dd, 8 - dd)
            dt = abs(x["time_s"] - y["time_s"])
            if x["object_id"] == y["object_id"] and x["type"] == y["type"] and dt <= .25 + 1e-9 and dd <= 1:
                cost[i, j] = dt + .01 * dd
    matched = sum(cost[i, j] < 1e6 for i, j in zip(*linear_sum_assignment(cost)))
    return bool(matched != len(a) or matched != len(b))


def scene_state(seed):
    rng = np.random.default_rng(seed)
    start, turn, stop, restart = int(rng.integers(3, 6)), int(rng.integers(12, 15)), int(rng.integers(21, 24)), int(rng.integers(27, 29))
    velocity = float(rng.uniform(3.0, 4.5))
    origin = float(rng.uniform(90, 120))
    centers = np.zeros((COUNT, 4, 2), np.float32)
    visible = np.zeros((COUNT, 4), bool)
    x = origin
    for t in range(COUNT):
        if t:
            previous = t - 1
            x += velocity if start <= previous < turn or previous >= restart else -velocity if turn <= previous < stop else 0
        centers[t, 0] = [x, 55]
        centers[t, 1] = [232, 55]
        centers[t, 2] = [245 - 2.3 * t, 142]
        centers[t, 3] = [60, 142]
        visible[t, :3] = [True, 7 <= t < 26, True]
    return {"centers": centers, "visible": visible,
            "colors": rng.integers(110, 230, (4, 3), dtype=np.uint8),
            "radii": rng.integers(15, 20, 4), "background": rng.integers(25, 65, 3)}


def render_scene(seed, kind=None, severity=0., restyle=None):
    state = scene_state(seed)
    centers, visible = state["centers"].copy(), state["visible"].copy()
    length = max(2, round(COUNT * severity)); begin = (COUNT - length) // 2
    span = slice(begin, begin + length)
    if kind == "motion_reverse":
        centers[span, 0] = centers[span, 0][::-1]
    elif kind == "motion_freeze":
        centers[span, 0] = centers[begin, 0]
    elif kind == "object_omission":
        visible[span, 0] = False
    elif kind == "object_addition":
        visible[span, 3] = True
    if restyle == "restyle_gait":
        for k in (0, 2):
            moving = np.r_[False, np.linalg.norm(np.diff(centers[:, k], axis=0), axis=1) >= .5]
            centers[moving, k, 1] += .12 * np.sin(np.arange(COUNT)[moving] * .9)
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    frames, masks = [], []
    for t in range(COUNT):
        texture = (30 * np.sin(xx * .29 + seed) * np.cos(yy * .23) if restyle == "restyle_texture"
                   else 10 * np.sin(xx * .09) * np.cos(yy * .11))
        frame = np.clip(state["background"] + texture[..., None], 0, 255).astype(np.uint8)
        labels = np.zeros((HEIGHT, WIDTH), np.uint8)
        for k in range(4):
            if not visible[t, k]:
                continue
            cx, cy = centers[t, k]; r = state["radii"][k]
            if kind == "object_shape" and k == 0 and begin <= t < begin + length:
                inside = ((xx - cx) / (r * 1.65)) ** 2 + ((yy - cy) / (r * .7)) ** 2 <= 1
            elif k == 0 or k == 3:
                inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
            elif restyle == "restyle_instance":
                inside = ((np.abs(xx - cx) / r) ** 4 + (np.abs(yy - cy) / r) ** 4) <= 1
            else:
                inside = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
            detail = (22 * np.sin((xx - cx) * .7) * np.cos((yy - cy) * .6) if restyle == "restyle_texture"
                      else 8 * np.sin((xx - cx) * .3) * np.cos((yy - cy) * .3))
            pixels = np.clip(state["colors"][k].astype(float) + detail[..., None], 0, 255).astype(np.uint8)
            frame[inside] = pixels[inside]; labels[inside] = k + 1
        if restyle == "restyle_palette":
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            hsv[..., 0] = (hsv[..., 0].astype(int) + 65) % 180
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        if restyle == "restyle_lighting":
            frame = np.clip(frame.astype(float) * (.65 + .6 * xx / WIDTH)[..., None] + 25, 0, 255).astype(np.uint8)
        if restyle == "restyle_camera":
            angle = 1.5 * math.sin(t * .45)
            matrix = cv2.getRotationMatrix2D((WIDTH / 2, HEIGHT / 2), angle, 1)
            matrix[:, 2] += [2 * math.sin(t * .8), 2 * math.cos(t * .6)]
            frame = cv2.warpAffine(frame, matrix, (WIDTH, HEIGHT), borderMode=cv2.BORDER_REFLECT_101)
            labels = cv2.warpAffine(labels, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_NEAREST)
        frames.append(frame); masks.append(labels)
    truth = {"centers": centers, "visible": visible, "masks": np.stack(masks)}
    return np.stack(frames), truth


def public_truth(labels):
    ids = np.unique(labels); ids = ids[(ids > 0) & (ids < 255)]
    centers = np.zeros((len(labels), len(ids), 2), np.float32)
    visible = np.zeros((len(labels), len(ids)), bool)
    for t, label in enumerate(labels):
        for k, identity in enumerate(ids):
            y, x = np.nonzero(label == identity)
            if len(x):
                visible[t, k] = True; centers[t, k] = [x.mean(), y.mean()]
    return {"centers": centers, "visible": visible, "masks": labels}


def variants(frames, truth, seed=None):
    for name, rec in nuisance_controls(frames):
        yield name, rec, truth, {"target": "control", "kind": "control", "severity": 0., "family": "pixel"}
    if seed is not None:
        for name in RESTYLES:
            rec, rt = render_scene(seed, restyle=name)
            yield name, rec, rt, {"target": "control", "kind": "control", "severity": 0., "family": "rerender"}
    else:
        hsv = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2HSV) for f in frames])
        hsv[..., 0] = (hsv[..., 0].astype(int) + 65) % 180
        values = {"hue_rotate": np.stack([cv2.cvtColor(f, cv2.COLOR_HSV2RGB) for f in hsv]),
                  "grayscale": np.stack([cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB) for f in frames]),
                  "downscale_half_upsample": np.stack([cv2.resize(cv2.resize(f, (WIDTH // 2, HEIGHT // 2), interpolation=cv2.INTER_AREA), (WIDTH, HEIGHT)) for f in frames])}
        jpeg = []
        for f in frames:
            ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 20])
            if not ok:
                raise ValueError("JPEG encoding failed")
            jpeg.append(cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        values["jpeg20"] = np.stack(jpeg)
        for name, rec in values.items():
            yield name, rec, truth, {"target": "control", "kind": "control", "severity": 0., "family": "appearance_proxy"}
    for kind in ("reverse", "freeze", "lag", "swap"):
        for severity in SEVERITIES:
            rec, index = temporal_variant(frames, kind, severity)
            indices = index["source_index_map"]
            rt = {k: v[indices] for k, v in truth.items()}
            yield f"{kind}_{severity}", rec, rt, {"target": "motion", "kind": kind, "severity": severity, "family": "pixel", **index}
    if seed is not None:
        for kind in ("motion_reverse", "motion_freeze", "object_addition", "object_omission", "object_shape"):
            for severity in SEVERITIES:
                rec, rt = render_scene(seed, kind=kind, severity=severity)
                yield f"{kind}_{severity}", rec, rt, {"target": "object" if kind.startswith("object_") else "motion",
                      "kind": kind, "severity": severity, "family": "rerender"}


def audit_truth(source, reconstructed, kind, variant):
    a = annotation_events(source["centers"], source["visible"])
    b = annotation_events(reconstructed["centers"], reconstructed["visible"])
    changed = truth_event_change(a, b)
    details = {"reference_events": a, "reconstruction_events": b, "event_changed": changed}
    if kind == "control":
        accepted = event_signature(a) == event_signature(b) and np.array_equal(source["visible"], reconstructed["visible"])
        if variant == "restyle_instance":
            overlaps, ratios = [], []
            for ma, mb in zip(source["masks"], reconstructed["masks"]):
                for k in np.unique(ma):
                    if not k:
                        continue
                    x, y = ma == k, mb == k
                    overlaps.append(np.logical_and(x, y).sum() / max(1, np.logical_or(x, y).sum()))
                    def aspect(mask):
                        yy, xx = np.nonzero(mask)
                        return (np.ptp(xx) + 1) / (np.ptp(yy) + 1)
                    ratios.append(abs(aspect(x) / aspect(y) - 1))
            details.update(min_mask_iou=float(min(overlaps)), max_aspect_change=float(max(ratios)))
            accepted = accepted and min(overlaps) >= .7 and max(ratios) <= .1
        reason = "control_invariant" if accepted else "control_changed_events_existence_or_shape_bounds"
    elif kind == "object_shape":
        ratios = []
        for ma, mb in zip(source["masks"], reconstructed["masks"]):
            ya, xa = np.nonzero(ma == 1); yb, xb = np.nonzero(mb == 1)
            if len(xa) and len(xb):
                ratios.append(((np.ptp(xb) + 1) / (np.ptp(yb) + 1)) / ((np.ptp(xa) + 1) / (np.ptp(ya) + 1)))
        accepted = bool(ratios and max(ratios) >= 2)
        details["maximum_aspect_ratio_factor"] = float(max(ratios)) if ratios else None
        reason = "shape_changed_at_least_twofold" if accepted else "shape_change_below_bound"
    else:
        accepted = changed
        reason = "event_changed" if accepted else "no_verified_event_change"
    return {"accepted": bool(accepted), "reason": reason, **details}
