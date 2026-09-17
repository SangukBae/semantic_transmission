"""MTE research candidate, evaluated on an unwarped physical time axis.

The scorer receives only RGB frames/flow, never intervention or renderer labels.
Directional bins retain opposing motion; the original vector-only proposal is
reported as an ablation. Camera motion is reported separately, not discarded.
"""
from collections import OrderedDict
import hashlib
import math

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def check_video(x):
    if x.ndim != 4 or x.shape[-1] != 3 or x.dtype != np.uint8 or len(x) < 3:
        raise ValueError("need >=3 RGB uint8 frames [T,H,W,3]")


class FlowExtractor:
    def __init__(self, backend="raft_small", device="cuda", cache_size=256):
        self.backend, self.device, self.cache_size = backend, device, cache_size
        self.cache = OrderedDict()
        if backend == "raft_small":
            import torch
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
            self.torch = torch
            self.weights = Raft_Small_Weights.C_T_V2
            self.model = raft_small(weights=self.weights, progress=False).to(device).eval()
        elif backend != "farneback":
            raise ValueError(backend)

    def __call__(self, frames):
        check_video(frames)
        keys = [hashlib.sha256(a.tobytes() + b.tobytes()).hexdigest() + str(a.shape)
                for a, b in zip(frames, frames[1:])]
        unique = list(dict.fromkeys(k for k in keys if k not in self.cache))
        lookup = dict(zip(keys, zip(frames, frames[1:])))
        for start in range(0, len(unique), 8):
            batch = unique[start:start + 8]
            if self.backend == "farneback":
                flows = [cv2.calcOpticalFlowFarneback(cv2.cvtColor(lookup[k][0], cv2.COLOR_RGB2GRAY),
                         cv2.cvtColor(lookup[k][1], cv2.COLOR_RGB2GRAY), None,
                         .5, 3, 15, 3, 5, 1.2, 0) for k in batch]
            else:
                torch = self.torch
                a, b = (torch.from_numpy(np.stack([lookup[k][i] for k in batch])).to(self.device)
                        .permute(0, 3, 1, 2).float() / 127.5 - 1 for i in (0, 1))
                h, w = a.shape[-2:]
                ph, pw = max(0, 128 - h) + (-max(h, 128)) % 8, max(0, 128 - w) + (-max(w, 128)) % 8
                a, b = (torch.nn.functional.pad(x, (0, pw, 0, ph), mode="replicate") for x in (a, b))
                with torch.inference_mode():
                    flows = self.model(a, b, num_flow_updates=12)[-1][:, :, :h, :w].permute(0, 2, 3, 1).cpu().numpy()
            for k, flow in zip(batch, flows):
                if not np.isfinite(flow).all():
                    raise ValueError("nonfinite optical flow")
                self.cache[k] = flow
        result = np.stack([self.cache[k] for k in keys])
        for k in keys:
            self.cache.move_to_end(k)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return result


def describe_flow(flows, fps=8., activity_px=.25):
    if flows.ndim != 4 or flows.shape[-1] != 2 or not np.isfinite(flows).all() or fps <= 0:
        raise ValueError("finite [T,H,W,2] flow and positive FPS required")
    _, h, w, _ = flows.shape
    yy, xx = np.mgrid[:h, :w]
    grid = np.stack((xx, yy), -1).astype(np.float32)
    points = grid[::8, ::8].reshape(-1, 2)
    result = {k: [] for k in ("hist", "vector", "centroid", "activity", "camera", "camera_fit")}
    for flow in flows:
        targets = points + flow[::8, ::8].reshape(-1, 2)
        cv2.setRNGSeed(913)
        affine, inliers = cv2.estimateAffinePartial2D(points, targets, method=cv2.RANSAC,
                                ransacReprojThreshold=1., maxIters=1000, confidence=.99)
        fit = float(inliers.mean()) if inliers is not None else 0.
        valid = affine is not None and np.isfinite(affine).all() and fit >= .5
        camera = grid @ affine[:, :2].T + affine[:, 2] - grid if valid else np.zeros_like(flow)
        residual = flow - camera
        mag = np.linalg.norm(residual, axis=-1)
        weight = np.where(mag >= activity_px, mag / (mag + 1), 0)
        # Eight nonnegative directional bins avoid opposite-vector cancellation.
        direction = ((np.arctan2(residual[..., 1], residual[..., 0]) + 2 * np.pi) % (2 * np.pi)) / (np.pi / 4)
        low = np.floor(direction).astype(int) % 8
        frac = direction - np.floor(direction)
        hist, vectors = [], []
        for ys in np.array_split(np.arange(h), 3):
            for xs in np.array_split(np.arange(w), 3):
                iy = np.ix_(ys, xs)
                ww, mm, ll, ff = weight[iy], mag[iy], low[iy], frac[iy]
                hist.extend([float(np.mean(ww * mm * ((ll == k) * (1 - ff) + ((ll + 1) % 8 == k) * ff)))
                             for k in range(8)])
                vectors.extend((residual[iy] * ww[..., None]).mean((0, 1)).tolist())
        centroid = ((grid * weight[..., None]).sum((0, 1)) / max(weight.sum(), 1e-12) / [w, h])
        result["hist"].append(np.asarray(hist) * fps / math.hypot(w, h))
        result["vector"].append(np.asarray(vectors) * fps / math.hypot(w, h))
        result["centroid"].append(centroid)
        result["activity"].append(float((mag * weight).mean()))
        # Five anchors retain camera rotation/scale, whose mean translation is zero.
        anchors = camera[[0, 0, h - 1, h - 1, h // 2], [0, w - 1, 0, w - 1, w // 2]]
        result["camera"].append((anchors * fps / [w, h]).reshape(-1) / math.sqrt(5))
        result["camera_fit"].append(fit if valid else 0.)
    return {k: np.asarray(v) for k, v in result.items()}


def _relative(a, b):
    return np.clip(np.linalg.norm(a - b, axis=-1) /
                   np.maximum(np.linalg.norm(a, axis=-1) + np.linalg.norm(b, axis=-1), 1e-12), 0, 1)


def motion_events(desc, fps, activity_threshold=.08):
    """Observable starts/stops/turns, with types and explicit timestamps."""
    active = desc["activity"] >= activity_threshold
    # Hysteresis: require two consecutive active/inactive transitions.
    events = []
    for i in range(2, len(active) - 1):
        before, after = active[i - 2:i], active[i:i + 2]
        kind = None
        if not before.any() and after.all():
            kind = "start"
        elif before.all() and not after.any():
            kind = "stop"
        elif before.all() and after.all():
            a = desc["vector"][i - 2:i].mean(0)
            b = desc["vector"][i:i + 2].mean(0)
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            if norm > 1e-10 and np.dot(a, b) / norm < -.25:
                kind = "turn"
        if kind and (not events or (i / fps - events[-1]["time_s"]) >= .375):
            events.append({"time_s": i / fps, "type": kind})
    return events


def match_events(a, b, tolerance_s=.25):
    cost = np.full((len(a), len(b)), 1e6)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            delta = abs(x["time_s"] - y["time_s"])
            if x["type"] == y["type"] and delta <= tolerance_s + 1e-9:
                cost[i, j] = delta
    pairs = [(int(i), int(j)) for i, j in zip(*linear_sum_assignment(cost)) if cost[i, j] < 1e6]
    return {"event_miss_rate": 1 - len(pairs) / len(a) if a else None,
            "event_extra_rate": 1 - len(pairs) / len(b) if b else None,
            "reference_events": a, "reconstruction_events": b, "event_pairs": pairs}


def score_motion(a, b, fps=8., window_s=.5, tail_fraction=.2):
    if fps <= 0 or not 0 < tail_fraction <= 1 or len(a["hist"]) != len(b["hist"]):
        raise ValueError("equal timelines and valid aggregation parameters required")
    size = max(1, round(fps * window_s))
    windows, legacy, camera = [], [], []
    eligible = 0
    for start in range(max(1, len(a["hist"]) - size + 1)):
        end = min(len(a["hist"]), start + size)
        active = max(a["activity"][start:end].mean(), b["activity"][start:end].mean()) >= .08
        va, vb = (x["hist"][start:end].mean(0) for x in (a, b))
        err = float(_relative(va, vb))
        ca, cb = (x["centroid"][start:end].mean(0) for x in (a, b))
        if min(a["activity"][start:end].mean(), b["activity"][start:end].mean()) >= .08:
            err = max(err, min(1., float(np.linalg.norm(ca - cb)) / .15))
        windows.append({"start_s": start / fps, "end_s": (end + 1) / fps,
                        "error": err if active else None})
        eligible += int(active)
        if active:
            legacy.append(float(_relative(a["vector"][start:end].mean(0), b["vector"][start:end].mean(0))))
        camera.append(float(np.linalg.norm(a["camera"][start:end].mean(0) - b["camera"][start:end].mean(0))))
    errors = [w["error"] for w in windows if w["error"] is not None]
    tail = lambda xs: float(np.mean(sorted(xs)[-max(1, math.ceil(len(xs) * tail_fraction)):])) if xs else None
    out = {"mte_tail": tail(errors), "mte_mean_ablation": float(np.mean(errors)) if errors else None,
           "mte_vector_ablation": tail(legacy), "mte_max": max(errors) if errors else None,
           "mte_window_coverage": eligible / len(windows), "mte_windows": windows,
           "camera_displacement_error": tail(camera),
           "camera_fit_coverage": float(np.mean((a["camera_fit"] >= .5) & (b["camera_fit"] >= .5)))}
    out.update(match_events(motion_events(a, fps), motion_events(b, fps)))
    return out


class MotionMetric:
    def __init__(self, backend="raft_small", device="cuda", fps=8.):
        self.extract = FlowExtractor(backend, device)
        self.fps = fps

    def evaluate(self, a, b):
        check_video(a)
        check_video(b)
        if a.shape != b.shape:
            raise ValueError("MTE requires equal frames on a shared physical timeline")
        fa, fb = self.extract(a), self.extract(b)
        result = score_motion(describe_flow(fa, self.fps), describe_flow(fb, self.fps), self.fps)
        result["tof_" + self.extract.backend] = float(np.linalg.norm(fa - fb, axis=-1).mean())
        return result
