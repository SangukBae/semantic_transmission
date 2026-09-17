"""TecoGAN-style tLP/tOF on the explicitly sampled evaluation timeline.

Uses current LPIPS-Alex v0.1 and all pixels/samples (no TecoGAN border crop or
first/last-frame exclusion). This is a documented adaptation, not a reproduction
of published TecoGAN numbers. PSNR from the sequence-wide MSE is also retained.
"""
import hashlib

import cv2
import numpy as np


class TemporalBaselines:
    def __init__(self, device="cpu", lpips_model=None):
        import torch
        import lpips
        self.torch, self.device = torch, device
        self.lpips = lpips_model if lpips_model is not None else lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.cache = {}

    def transitions(self, frames):
        key = hashlib.sha256(frames.tobytes()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        torch = self.torch
        flow = []
        grey = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
        for a, b in zip(grey, grey[1:]):
            flow.append(cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0))
        with torch.inference_mode():
            x = torch.tensor(frames.copy(), device=self.device).permute(0, 3, 1, 2).float() / 127.5 - 1
            lp = self.lpips(x[:-1], x[1:]).flatten().cpu().numpy()
        value = (np.stack(flow), lp)
        # Per-source cache only; avoid retaining optical-flow arrays for the corpus.
        self.cache[key] = value
        return value

    def evaluate(self, a, b):
        if a.shape != b.shape or len(a) < 2 or a.dtype != np.uint8 or b.dtype != np.uint8:
            raise ValueError("equal RGB uint8 sequences required")
        fa, la = self.transitions(a)
        fb, lb = self.transitions(b)
        mse = float(np.mean((a.astype(np.float64) - b) ** 2))
        return {"tlp_alex": float(np.abs(la - lb).mean()),
                "tof_farneback": float(np.linalg.norm(fa - fb, axis=-1).mean()),
                "psnr_global_db": float(10 * np.log10(255 ** 2 / max(mse, 1e-12)))}
