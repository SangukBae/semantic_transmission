"""OTF candidate using independently discovered SAM2 masks and DINOv2 features.

No text, reference annotation, corruption label or oracle mask enters extraction.
Cross-video association is one-to-one over entire predicted tracks. The F1 uses
shared matched object-frame counts, unlike a harmonic mean of incompatible area
weighted omission/addition fractions. Outputs describe predicted objects only.
"""
import hashlib
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .motion_metric import check_video


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.


def center(mask):
    y, x = np.where(mask)
    return np.array([x.mean() / mask.shape[1], y.mean() / mask.shape[0]]) if len(x) else np.zeros(2)


def shape_iou(a, b, shift_fraction=.03):
    # Alignment is bounded and used for shape comparison only; temporal alignment
    # and MTE retain all motion/delay. No maximum over unbounded translations.
    delta = (center(a) - center(b)) * [a.shape[1], a.shape[0]]
    cap = np.array([a.shape[1], a.shape[0]]) * shift_fraction
    dx, dy = np.clip(np.rint(delta), -cap, cap)
    moved = cv2.warpAffine(b.astype(np.uint8), np.float32([[1, 0, dx], [0, 1, dy]]),
                           (a.shape[1], a.shape[0]), flags=cv2.INTER_NEAREST).astype(bool)
    return max(iou(a, b), iou(a, moved))


def associate_frames(detections, max_gap=2):
    tracks = []
    for t, dets in enumerate(detections):
        active = [k for k, tr in enumerate(tracks) if t - max(tr) <= max_gap + 1]
        cost = np.full((len(active), len(dets)), 1e6)
        for i, k in enumerate(active):
            prev = tracks[k][max(tracks[k])]
            for j, det in enumerate(dets):
                sim = float(np.clip(np.dot(prev["feature"], det["feature"]), -1, 1))
                overlap = iou(prev["mask"], det["mask"])
                dist = float(np.linalg.norm(center(prev["mask"]) - center(det["mask"])))
                if overlap >= .1 or (dist <= .15 and sim >= .6):
                    cost[i, j] = .5 * (1 - sim) + .5 * (1 - overlap)
        cost[cost > .65] = 1e6
        matched = set()
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] <= .65:
                tracks[active[i]][t] = dets[j]
                matched.add(j)
        for j, det in enumerate(dets):
            if j not in matched:
                tracks.append({t: det})
    return tracks


def _intervals(times, fps):
    times = sorted(times)
    out = []
    for t in times:
        if out and out[-1][1] == t:
            out[-1][1] = t + 1
        else:
            out.append([t, t + 1])
    return [{"start_s": a / fps, "end_s": b / fps, "frames": b - a} for a, b in out]


def score_tracks(a, b, fps=8., frames=None):
    if fps <= 0:
        raise ValueError("positive FPS required")
    if not a:
        return {"otf_f1": None, "otf_error": None, "otf_distortion_error": None, "oor": None, "hor": None, "odr": None,
                "otf_reason": "no_reference_tracks", "reference_tracks": 0, "reconstruction_tracks": len(b)}
    cost = np.full((len(a), len(b)), 1e6)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            common = sorted(set(x) & set(y))
            if not common:
                continue
            sim = np.mean([np.dot(x[t]["feature"], y[t]["feature"]) for t in common])
            overlap = np.mean([shape_iou(x[t]["mask"], y[t]["mask"]) for t in common])
            distance = np.mean([np.linalg.norm(center(x[t]["mask"]) - center(y[t]["mask"])) for t in common])
            if overlap >= .1 or (sim >= .65 and distance <= .2):
                cost[i, j] = .5 * (1 - sim) + .5 * (1 - overlap)
    cost[cost > .65] = 1e6
    pairs = [(int(i), int(j)) for i, j in zip(*linear_sum_assignment(cost)) if cost[i, j] <= .65]
    amap, bmap = dict(pairs), {j: i for i, j in pairs}
    matched = sum(len(set(a[i]) & set(b[j])) for i, j in pairs)
    source_count, rec_count = sum(map(len, a)), sum(map(len, b))
    missing, added, distorted = [], [], []
    distortion_count = 0
    for i, tr in enumerate(a):
        absent = set(tr) - set(b[amap[i]]) if i in amap else set(tr)
        missing.extend({"track": i, **x} for x in _intervals(absent, fps))
    for j, tr in enumerate(b):
        extra = set(tr) - set(a[bmap[j]]) if j in bmap else set(tr)
        added.extend({"track": j, **x} for x in _intervals(extra, fps))
    for i, j in pairs:
        bad = []
        for t in sorted(set(a[i]) & set(b[j])):
            x, y = a[i][t], b[j][t]
            if np.dot(x["feature"], y["feature"]) < .65 or shape_iou(x["mask"], y["mask"]) < .5:
                bad.append(t)
        distortion_count += len(bad)
        distorted.extend({"reference_track": i, "reconstruction_track": j, **x} for x in _intervals(bad, fps))
    f1 = 2 * matched / (source_count + rec_count)
    return {"otf_f1": f1, "otf_error": 1 - f1, "oor": 1 - matched / source_count,
            "hor": 1 - matched / rec_count if rec_count else None,
            "odr": distortion_count / matched if matched else None,
            "otf_distortion_error": max(1 - f1, distortion_count / matched if matched else 0),
            "reference_tracks": len(a), "reconstruction_tracks": len(b),
            "matched_object_frames": matched, "reference_object_frames": source_count,
            "reconstruction_object_frames": rec_count, "track_pairs": pairs,
            "omission_intervals": missing, "addition_intervals": added, "distortion_intervals": distorted,
            "long_omission_events": sum(x["frames"] / fps >= .5 for x in missing),
            "long_addition_events": sum(x["frames"] / fps >= .5 for x in added),
            "otf_reason": "predicted_masks_not_semantic_ground_truth"}


class ObjectExtractor:
    def __init__(self, model_root, cache_root, device="cuda"):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        self.torch, self.device = torch, device
        root = Path(model_root)
        sam = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", str(root / "sam2.1_hiera_tiny.pt"),
                         device=device, apply_postprocessing=False)
        self.sam = SAM2AutomaticMaskGenerator(sam, points_per_side=12, points_per_batch=64,
                        pred_iou_thresh=.8, stability_score_thresh=.9, crop_n_layers=0,
                        min_mask_region_area=0, box_nms_thresh=.7)
        self.dino = torch.hub.load(str(root / "dinov2"), "dinov2_vits14", source="local", pretrained=False)
        self.dino.load_state_dict(torch.load(root / "dinov2_vits14_pretrain.pth", map_location="cpu", weights_only=True))
        self.dino = self.dino.to(device).eval()
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        # Cache identity includes weights + extractor source, preventing stale reuse.
        from .artifacts import sha256
        self.signature = hashlib.sha256((sha256(Path(__file__)) + sha256(root / "sam2.1_hiera_tiny.pt") +
                         sha256(root / "dinov2_vits14_pretrain.pth")).encode()).hexdigest()

    def frame(self, frame):
        torch = self.torch
        key = hashlib.sha256(frame.tobytes() + str(frame.shape).encode() + self.signature.encode()).hexdigest()
        cache = self.cache_root / (key + ".npz")
        if cache.exists():
            with np.load(cache, allow_pickle=False) as f:
                return [{"mask": m.astype(bool), "feature": v} for m, v in zip(f["masks"], f["features"])]
        with torch.inference_mode():
            annotations = self.sam.generate(frame)
            # Largest non-background masks first; suppress nested duplicate parts.
            annotations = sorted([x for x in annotations if .002 <= x["area"] / np.prod(frame.shape[:2]) <= .65],
                                 key=lambda x: -x["area"])
            masks = []
            for x in annotations:
                m = x["segmentation"]
                if not any(np.logical_and(m, prev).sum() / max(1, m.sum()) > .85 for prev in masks):
                    masks.append(m)
                if len(masks) == 20:
                    break
            x = torch.from_numpy(cv2.resize(frame, (392, 224))).to(self.device).permute(2, 0, 1)[None].float() / 255
            x = (x - x.new_tensor([.485, .456, .406])[None, :, None, None]) / x.new_tensor([.229, .224, .225])[None, :, None, None]
            feature = self.dino.forward_features(x)["x_norm_patchtokens"][0].reshape(16, 28, -1).float().cpu().numpy()
        vectors = []
        for mask in masks:
            weight = cv2.resize(mask.astype(np.float32), (28, 16), interpolation=cv2.INTER_AREA)
            value = (feature * weight[..., None]).sum((0, 1)) / max(weight.sum(), 1e-12)
            value /= max(np.linalg.norm(value), 1e-12)
            vectors.append(value)
        mm = np.stack(masks).astype(np.uint8) if masks else np.empty((0, *frame.shape[:2]), np.uint8)
        ff = np.stack(vectors) if vectors else np.empty((0, 384), np.float32)
        if not np.isfinite(ff).all():
            raise ValueError("nonfinite DINO features")
        np.savez_compressed(cache, masks=mm, features=ff)
        return [{"mask": m.astype(bool), "feature": v} for m, v in zip(mm, ff)]

    def __call__(self, frames):
        check_video(frames)
        return [self.frame(f) for f in frames]


class ObjectMetric:
    def __init__(self, model_root, cache_root, device="cuda", fps=8.):
        self.extract = ObjectExtractor(model_root, cache_root, device)
        self.fps = fps

    def evaluate(self, a, b):
        check_video(a)
        check_video(b)
        if a.shape != b.shape:
            raise ValueError("OTF requires equal frames on a shared physical timeline")
        da, db = self.extract(a), self.extract(b)
        result = score_tracks(associate_frames(da), associate_frames(db), self.fps, len(a))
        result["reference_detection_frame_coverage"] = sum(bool(x) for x in da) / len(da)
        result["reconstruction_detection_frame_coverage"] = sum(bool(x) for x in db) / len(db)
        return result
