"""ERE/SER from independent RGB presence and residual-motion channels.

SAM2/DINO tracks retain stationary objects. RAFT observes their motion, never
decides their existence. Annotations are used only by external audit code.
"""
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .motion_metric import FlowExtractor, check_video

PARAMETERS = {
    "activity_px": .25, "minimum_area_fraction": .003,
    "maximum_gap_frames": 2, "maximum_distance_diagonal": .1,
    "smooth_frames": 5, "minimum_speed_px_per_frame": .5,
    "speed_bin_edges_px_per_frame": [1.5, 4.],
    "minimum_event_separation_s": .375, "matching_tolerance_s": .25,
    "turn_angle_degrees": 90., "turn_context_frames": 2,
    "state_persistence_frames": 2,
}


def descriptor(kind, time_s, center, velocity, shape, track_id=None):
    h, w = shape
    speed = float(np.linalg.norm(velocity))
    angle = math.atan2(float(velocity[1]), float(velocity[0]))
    return {"type": kind, "time_s": float(time_s),
            "direction": int(math.floor((angle % (2 * math.pi)) / (math.pi / 4) + .5)) % 8,
            "speed_bin": int(np.searchsorted(PARAMETERS["speed_bin_edges_px_per_frame"], speed)),
            "cell": np.clip(np.floor(np.asarray(center) / [w, h] * 3), 0, 2).astype(int).tolist(),
            "center": (np.asarray(center) / [w, h]).tolist(), "track_id": track_id}


def event_pairs(a, b, tolerance_s=.25, direction_tolerance=1):
    cost = np.full((len(a), len(b)), 1e6)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            dt = abs(x["time_s"] - y["time_s"])
            dd = abs(x["direction"] - y["direction"])
            dd = min(dd, 8 - dd)
            cell = max(abs(p - q) for p, q in zip(x["cell"], y["cell"]))
            if x["type"] == y["type"] and dt <= tolerance_s + 1e-9 and dd <= direction_tolerance and cell <= 1:
                cost[i, j] = dt + .01 * dd + .001 * cell
    return [(int(i), int(j)) for i, j in zip(*linear_sum_assignment(cost)) if cost[i, j] < 1e6]


def score_events(a, b):
    pairs = event_pairs(a, b)
    ma, mb = {i for i, _ in pairs}, {j for _, j in pairs}
    return {"ere": 1 - len(pairs) / len(a) if len(a) >= 2 else None,
            "ser": 1 - len(pairs) / len(b) if b else None,
            "reference_event_count": len(a), "reconstruction_event_count": len(b),
            "matched_event_count": len(pairs), "event_pairs": pairs,
            "reference_events": a, "reconstruction_events": b,
            "error_intervals": [{"side": side, "time_s": e["time_s"], "cell": e["cell"], "type": e["type"]}
                                for side, events, matched in (("reference", a, ma), ("reconstruction", b, mb))
                                for k, e in enumerate(events) if k not in matched],
            "ere_reason": "fewer_than_two_reference_events" if len(a) < 2 else "predicted_motion_events"}


def residual_regions(flows):
    if flows.ndim != 4 or flows.shape[-1] != 2 or not np.isfinite(flows).all():
        raise ValueError("finite [T,H,W,2] optical flow required")
    _, h, w, _ = flows.shape
    yy, xx = np.mgrid[:h, :w]
    grid = np.stack((xx, yy), -1).astype(np.float32)
    points = grid[::8, ::8].reshape(-1, 2)
    frames, fits = [], []
    for flow in flows:
        cv2.setRNGSeed(913)
        affine, inliers = cv2.estimateAffinePartial2D(points, points + flow[::8, ::8].reshape(-1, 2),
            method=cv2.RANSAC, ransacReprojThreshold=1., maxIters=1000, confidence=.99)
        fit = float(inliers.mean()) if inliers is not None else 0.
        valid = affine is not None and np.isfinite(affine).all() and fit >= .5
        camera = grid @ affine[:, :2].T + affine[:, 2] - grid if valid else np.zeros_like(flow)
        residual = flow - camera
        mask = (np.linalg.norm(residual, axis=-1) >= PARAMETERS["activity_px"]).astype(np.uint8)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
        regions = []
        for k in range(1, count):
            if stats[k, cv2.CC_STAT_AREA] < h * w * PARAMETERS["minimum_area_fraction"]:
                continue
            region = labels == k
            regions.append({"mask": region, "center": centers[k], "velocity": residual[region].mean(0)})
        frames.append(regions)
        fits.append(fit if valid else 0.)
    return frames, fits


def link_regions(frames, shape):
    tracks = []
    potential_links, resumed_links = 0, 0
    limit = math.hypot(*shape) * PARAMETERS["maximum_distance_diagonal"]
    for t, regions in enumerate(frames):
        available = [k for k, tr in enumerate(tracks) if t - max(tr) <= PARAMETERS["maximum_gap_frames"] + 1]
        cost = np.full((len(available), len(regions)), 1e6)
        for i, k in enumerate(available):
            last_t = max(tracks[k]); last = tracks[k][last_t]
            predicted = last["center"] + last["velocity"] * (t - last_t)
            for j, region in enumerate(regions):
                distance = float(np.linalg.norm(predicted - region["center"]))
                overlap = np.logical_and(last["mask"], region["mask"]).sum() / max(1, np.logical_or(last["mask"], region["mask"]).sum())
                if distance <= limit:
                    cost[i, j] = 1 - overlap + distance / limit
        paired = [(i, j) for i, j in zip(*linear_sum_assignment(cost)) if cost[i, j] < 1e6]
        for i, j in paired:
            k = available[i]
            resumed_links += int(t - max(tracks[k]) > 1)
            potential_links += 1
            tracks[k][t] = regions[j]
        used = {j for _, j in paired}
        tracks.extend({t: r} for j, r in enumerate(regions) if j not in used)
    return tracks, {"resumed_gap_links": resumed_links, "links": potential_links,
                    "track_fragmentation_rate": None,
                    "fragmentation_reason": "true identity required; measured separately against audit annotations"}


def track_events(tracks, frame_count, shape, fps=8.):
    events = []
    speed_min = PARAMETERS["minimum_speed_px_per_frame"]
    for k, tr in enumerate(tracks):
        observed = sorted(tr)
        if len(observed) < 2:
            continue
        times = np.arange(observed[0], observed[-1] + 1)
        centers = np.stack([tr[t]["center"] for t in observed])
        points = np.stack([np.interp(times, observed, centers[:, d]) for d in (0, 1)], 1)
        smooth = np.stack([points[max(0, t - 2):min(len(points), t + 3)].mean(0) for t in range(len(points))])
        velocity = np.diff(smooth, axis=0)
        candidates = []
        if times[0] > 0:
            candidates.append(("enter", times[0], smooth[0], velocity[0]))
        for i in range(1, len(velocity)):
            before = velocity[max(0, i - 2):i].mean(0)
            after = velocity[i:min(len(velocity), i + 2)].mean(0)
            na, nb = np.linalg.norm(before), np.linalg.norm(after)
            kind = None
            if na < speed_min <= nb:
                kind = "start"
            elif nb < speed_min <= na:
                kind = "stop"
            elif min(na, nb) >= speed_min and np.dot(before, after) < 0:
                kind = "turn"
            if kind:
                candidates.append((kind, times[i], smooth[i], before if kind == "stop" else after))
        if times[-1] < frame_count - 2:
            candidates.append(("exit", times[-1] + 1, smooth[-1], velocity[-1]))
        last = -math.inf
        for kind, t, center, vector in sorted(candidates, key=lambda c: c[1]):
            if t / fps - last >= PARAMETERS["minimum_event_separation_s"] - 1e-9:
                events.append(descriptor(kind, t / fps, center, vector, shape, k))
                last = t / fps
    return sorted(events, key=lambda e: (e["time_s"], e["type"], e["track_id"]))


def residual_flow(flows):
    """Same camera fit as MTE, retaining the residual field for presence masks."""
    if flows.ndim != 4 or flows.shape[-1] != 2 or not np.isfinite(flows).all():
        raise ValueError("finite [T,H,W,2] optical flow required")
    _, h, w, _ = flows.shape
    yy, xx = np.mgrid[:h, :w]
    grid = np.stack((xx, yy), -1).astype(np.float32)
    points = grid[::8, ::8].reshape(-1, 2)
    residuals, cameras, fits = [], [], []
    for flow in flows:
        cv2.setRNGSeed(913)
        matrix, inliers = cv2.estimateAffinePartial2D(points, points + flow[::8, ::8].reshape(-1, 2),
            method=cv2.RANSAC, ransacReprojThreshold=1., maxIters=1000, confidence=.99)
        fit = float(inliers.mean()) if inliers is not None else 0.
        valid = matrix is not None and np.isfinite(matrix).all() and fit >= .5
        camera = grid @ matrix[:, :2].T + matrix[:, 2] - grid if valid else np.zeros_like(flow)
        residuals.append(flow - camera); cameras.append(camera); fits.append(fit if valid else 0.)
    return np.stack(residuals), np.stack(cameras), fits


def presence_tracks(detections, residuals, cameras):
    from .object_metric import associate_frames
    tracks = associate_frames(detections)
    for track in tracks:
        for t, detection in track.items():
            yy, xx = np.nonzero(detection['mask'])
            detection['center'] = np.array([xx.mean(), yy.mean()])
            detection['velocity'] = residuals[t][detection['mask']].mean(0) if t < len(residuals) else None
            detection['camera_velocity'] = cameras[t][detection['mask']].mean(0) if t < len(cameras) else None
    return tracks


def compact_tracks(tracks):
    """Numeric tracks for development diagnostics; no masks or labels needed."""
    return [{int(t): {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in observation.items() if k in ('center', 'velocity', 'camera_velocity')}
             for t, observation in track.items()} for track in tracks]


def presence_events(tracks, frame_count, shape, fps=8., parameters=None):
    cfg = {**PARAMETERS, **(parameters or {})}
    width = int(cfg['smooth_frames'])
    if width < 1 or width % 2 != 1:
        raise ValueError('smoothing width must be positive and odd')
    half, context = width // 2, int(cfg['turn_context_frames'])
    if context < 1 or fps <= 0:
        raise ValueError('positive context and FPS required')
    minimum, separation = cfg['minimum_speed_px_per_frame'], cfg['minimum_event_separation_s'] * fps
    events = []
    for identity, source in enumerate(tracks):
        track = {int(t): observation for t, observation in source.items()}
        observed = sorted(track)
        if not observed:
            continue
        times = np.arange(observed[0], observed[-1] + 1)
        xy = np.asarray([track[t]['center'] for t in observed])
        points = np.stack([np.interp(times, observed, xy[:, k]) for k in (0, 1)], -1)
        available = [t for t in observed if track[t].get('velocity') is not None]
        flow = np.zeros((max(0, len(times) - 1), 2))
        camera = np.zeros_like(flow)
        if available and len(flow):
            v = np.asarray([track[t]['velocity'] for t in available])
            c = np.asarray([track[t].get('camera_velocity', [0., 0.]) for t in available])
            flow = np.stack([np.interp(times[:-1], available, v[:, k]) for k in (0, 1)], -1)
            camera = np.stack([np.interp(times[:-1], available, c[:, k]) for k in (0, 1)], -1)
        stable = points - np.vstack((np.zeros((1, 2)), np.cumsum(camera, axis=0)))
        smooth = np.stack([stable[max(0, i - half):min(len(stable), i + half + 1)].mean(0) for i in range(len(stable))])
        velocity = np.diff(smooth, axis=0)
        active = np.linalg.norm(flow, axis=1) >= minimum
        candidates = []
        def add(kind, frame, index, vector, priority):
            candidates.append((kind, int(frame), points[index], np.asarray(vector), priority))
        # A stationary birth/death has no direction; use the fixed zero-speed
        # descriptor convention also used by the independent annotation audit.
        first = flow[0] if len(flow) and active[0] else np.zeros(2)
        last = flow[-1] if len(flow) and active[-1] else np.zeros(2)
        if times[0] > 0:
            add('enter', times[0], 0, first, 1000.)
        if times[-1] < frame_count - 1:
            add('exit', times[-1] + 1, len(points) - 1, last, 1000.)
        persistence = int(cfg['state_persistence_frames'])
        for i in range(1, len(flow)):
            before_active = active[max(0, i - persistence):i]
            after_active = active[i:i + persistence]
            if len(before_active) < persistence or len(after_active) < persistence:
                continue
            if not before_active.any() and after_active.all():
                add('start', times[i], i, flow[i:i + persistence].mean(0), 100.)
            if before_active.all() and not after_active.any():
                add('stop', times[i], i, flow[max(0, i - persistence):i].mean(0), 100.)
        for i in range(1, len(velocity)):
            left, right = max(0, i - context), min(len(velocity), i + context)
            if not active[left:i].any() or not active[i:right].any():
                continue
            a, b = velocity[left:i].mean(0), velocity[i:right].mean(0)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if min(na, nb) < minimum:
                continue
            angle = math.degrees(math.acos(float(np.clip(np.dot(a, b) / (na * nb), -1, 1))))
            if angle > cfg['turn_angle_degrees']:
                add('turn', times[i], i, b, 200. + angle)
        chosen = []
        for candidate in sorted(candidates, key=lambda x: (-x[4], x[1])):
            if all(abs(candidate[1] - prior[1]) >= separation - 1e-9 for prior in chosen):
                chosen.append(candidate)
        for kind, t, center, vector, _ in sorted(chosen, key=lambda x: x[1]):
            events.append(descriptor(kind, t / fps, center, vector, shape, identity))
    return sorted(events, key=lambda e: (e['time_s'], e['track_id'], e['type']))


class EventMetric:
    def __init__(self, device="cuda", backend="raft_small", fps=8., model_root='.local/metric_v2_models',
                 cache_root='outputs/metric_v31_presence_cache', parameters=None):
        from .object_metric import ObjectExtractor
        self.flow = FlowExtractor(backend=backend, device=device)
        self.presence = ObjectExtractor(Path(model_root), Path(cache_root), device=device)
        self.fps, self.parameters = fps, parameters

    def observe(self, frames):
        check_video(frames)
        residuals, cameras, fits = residual_flow(self.flow(frames))
        tracks = presence_tracks(self.presence(frames), residuals, cameras)
        return {'tracks': tracks, 'track_count': len(tracks), 'frame_count': len(frames),
                'shape': list(frames.shape[1:3]), 'camera_fit_coverage': float(np.mean(np.asarray(fits) >= .5)),
                'track_fragmentation_rate': None, 'fragmentation_reason': 'requires independent identity audit',
                'presence_channel': 'SAM2.1 Hiera-tiny + DINOv2 ViT-S/14, existing association rules',
                'motion_channel': 'mask-mean RAFT residual for activity; stabilized smoothed centers for turns'}

    def extract(self, frames):
        observations = self.observe(frames)
        return {**observations, 'events': presence_events(observations['tracks'], len(frames), frames.shape[1:3],
                                                        self.fps, self.parameters)}

    def evaluate(self, a, b):
        if a.shape != b.shape:
            raise ValueError("ERE requires equal RGB shape and shared unwarped timestamps")
        x, y = self.extract(a), self.extract(b)
        result = score_events(x["events"], y["events"])
        result["extraction"] = {side: {k: v for k, v in d.items() if k not in ("events", "tracks")}
                                for side, d in (("reference", x), ("reconstruction", y))}
        return result
