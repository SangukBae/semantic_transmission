"""Image-only research candidates. No corruption labels enter these functions.

RTE: source-relative CLIP transition discrepancy; LSSD: directed local CLIP
token support discrepancy. Neither score is a certified semantic error label.
"""
from collections import OrderedDict
import hashlib
import math

import numpy as np


def transition_error(a, b, epsilon=1e-4):
    if a.shape != b.shape or len(a) < 2:
        raise ValueError("transition features require equal sequences with >=2 frames")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite transition features")
    da, db = np.diff(a, axis=0), np.diff(b, axis=0)
    activity = np.linalg.norm(da, axis=-1) + np.linalg.norm(db, axis=-1)
    valid = activity > epsilon
    numerator = np.linalg.norm(da - db, axis=-1)
    per_transition = [float(min(n / d, 1)) if v else None for n, d, v in zip(numerator, activity, valid)]
    return {"rte": float(np.clip(numerator[valid].sum() / activity[valid].sum(), 0, 1)) if valid.any() else None,
            "transition_coverage": float(valid.mean()), "rte_transitions": per_transition,
            "reference_feature_activity": float(np.linalg.norm(da, axis=-1).mean())}


def local_support(a, b, radius=1, upper_fraction=0.25):
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("local features require equal [time,patch,feature] arrays")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite local features")
    grid = math.isqrt(a.shape[1])
    if grid * grid != a.shape[1]:
        raise ValueError("patches must form a square grid")
    y, x = np.divmod(np.arange(a.shape[1]), grid)
    allowed = np.maximum(abs(y[:, None] - y), abs(x[:, None] - x)) <= radius
    distances = np.clip((1 - np.einsum("tpc,tqc->tpq", a, b)) / 2, 0, 1)
    distances[:, ~allowed] = np.inf
    missing = distances.min(axis=2)
    extra = distances.min(axis=1)
    count = max(1, math.ceil(a.shape[1] * upper_fraction))
    # Directed upper tails retain local discrepancies that a whole-frame mean hides.
    m = np.sort(missing, axis=1)[:, -count:].mean(axis=1)
    e = np.sort(extra, axis=1)[:, -count:].mean(axis=1)
    return {"lssd": float(np.maximum(m, e).mean()),
            "reference_support_gap": float(m.mean()), "reconstruction_support_gap": float(e.mean()),
            "lssd_frames": np.maximum(m, e).tolist()}


class AutomaticMetrics:
    def __init__(self, device="cuda", cache_size=2048):
        import clip
        import lpips
        import torch
        self.torch, self.device = torch, device
        self.clip, _ = clip.load("ViT-B/32", device=device, jit=False)
        self.clip.eval()
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.cache = OrderedDict()
        self.cache_size = cache_size

    def features(self, frames):
        import cv2
        torch = self.torch
        keys = [hashlib.sha256(f.tobytes()).hexdigest() + str(f.shape) for f in frames]
        missing = list(dict.fromkeys(k for k in keys if k not in self.cache))
        lookup = dict(zip(keys, frames))
        for start in range(0, len(missing), 24):
            batch_keys = missing[start:start + 24]
            images = []
            for key in batch_keys:
                f = lookup[key]
                h, w = f.shape[:2]
                scale = 224 / max(h, w)
                nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
                canvas = np.full((224, 224, 3), 127, dtype=np.uint8)
                top, left = (224 - nh) // 2, (224 - nw) // 2
                canvas[top:top + nh, left:left + nw] = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA)
                images.append(canvas)
            x = torch.tensor(np.stack(images), device=self.device).permute(0, 3, 1, 2).float() / 255
            mean = x.new_tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
            std = x.new_tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
            with torch.inference_mode():
                v = self.clip.visual
                x = v.conv1(((x - mean) / std).to(v.conv1.weight.dtype)).flatten(2).permute(0, 2, 1)
                cls = v.class_embedding.to(x.dtype)[None, None, :].expand(len(x), 1, -1)
                x = torch.cat([cls, x], dim=1) + v.positional_embedding.to(x.dtype)
                x = v.transformer(v.ln_pre(x).permute(1, 0, 2)).permute(1, 0, 2)
                x = (v.ln_post(x) @ v.proj).float()
                x = (x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
            for key, feature in zip(batch_keys, x):
                self.cache[key] = feature
        result = np.stack([self.cache[k] for k in keys])
        for k in keys:
            self.cache.move_to_end(k)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return result[:, 0], result[:, 1:]

    def evaluate(self, source, reconstructed):
        from skimage.metrics import structural_similarity
        torch = self.torch
        if source.dtype != np.uint8 or reconstructed.dtype != np.uint8 or source.shape != reconstructed.shape:
            raise ValueError("metrics require equal RGB uint8 [time,height,width,3] arrays")
        if source.ndim != 4 or source.shape[-1] != 3 or len(source) < 2:
            raise ValueError("at least two RGB frames are required")
        ga, pa = self.features(source)
        gb, pb = self.features(reconstructed)
        values = transition_error(ga, gb)
        values.update(local_support(pa, pb))
        values["clip_cosine"] = float(np.clip(np.einsum("tc,tc->t", ga, gb), -1, 1).mean())
        mse = ((source.astype(np.float64) - reconstructed) ** 2).mean(axis=(1, 2, 3))
        # Capped exact-match PSNR keeps strict JSON and aggregate statistics finite.
        values["psnr_db"] = float((10 * np.log10(255 ** 2 / np.maximum(mse, 1e-12))).mean())
        values["ssim"] = float(np.mean([structural_similarity(a, b, channel_axis=-1, data_range=255,
             gaussian_weights=True, sigma=1.5, use_sample_covariance=False) for a, b in zip(source, reconstructed)]))
        perceptual = []
        with torch.inference_mode():
            for start in range(0, len(source), 16):
                a = torch.tensor(source[start:start + 16].copy(), device=self.device).permute(0, 3, 1, 2).float() / 127.5 - 1
                b = torch.tensor(reconstructed[start:start + 16].copy(), device=self.device).permute(0, 3, 1, 2).float() / 127.5 - 1
                perceptual.extend(self.lpips(a, b).flatten().cpu().tolist())
        values["lpips_alex"] = float(np.mean(perceptual))
        da = np.diff(source.astype(np.float32) / 255, axis=0)
        db = np.diff(reconstructed.astype(np.float32) / 255, axis=0)
        values["temporal_pixel_mae"] = float(np.abs(da - db).mean())
        return values
