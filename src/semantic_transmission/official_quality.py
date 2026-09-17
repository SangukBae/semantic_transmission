"""The five metrics in the pinned LGVSC final_score.py, one GPU model at a time."""
import gc
import math

PROFILE = "lgvsc_official_metrics_v1"
KEYS = ("psnr_db", "ssim", "lpips_vgg", "clip", "dists")


def summarize(rows, shape):
    import numpy as np
    if not rows:
        raise ValueError("cannot summarize zero frames")
    result = {key: float(np.mean([row[key] for row in rows])) for key in KEYS}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("nonfinite quality metric")
    return dict(result, frames=len(rows), shape=list(shape))


def pixel_scores(a, b):
    import cv2
    from skimage.metrics import structural_similarity
    # The release reads extracted PNGs with IMREAD_GRAYSCALE. PNG's grayscale
    # conversion can round differently from cvtColor; preserve that exact path.
    gray = []
    for rgb in (a, b):
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("PNG encoding failed during grayscale evaluation")
        gray.append(cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE))
    return {"psnr_db": float(cv2.PSNR(a, b)),
            "ssim": float(structural_similarity(gray[0], gray[1]))}


class OfficialMetrics:
    """Batch size one and sequential model lifetimes bound evaluation VRAM."""

    def __init__(self, device="cuda"):
        self.device = device

    def evaluate(self, source, reconstructed):
        import numpy as np
        import torch
        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        if source.shape != reconstructed.shape or source.ndim != 4 or source.shape[-1] != 3:
            raise ValueError("quality inputs must be matching T,H,W,3 arrays")
        if source.dtype != np.uint8 or reconstructed.dtype != np.uint8 or not len(source):
            raise ValueError("quality inputs must be nonempty decoded RGB uint8")
        if str(self.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        rows = [dict(frame=i, **pixel_scores(a, b))
                for i, (a, b) in enumerate(zip(source, reconstructed))]
        for metric in ("lpips_vgg", "clip", "dists"):
            print(f"Evaluating {metric}: {len(rows)} frames", flush=True)
            if metric == "lpips_vgg":
                import lpips
                model = lpips.LPIPS(net="vgg").to(self.device).eval()
            elif metric == "clip":
                import clip
                model, preprocess = clip.load("ViT-B/32", device=self.device)
                model.eval()
            else:
                from DISTS_pytorch import DISTS
                model = DISTS().to(self.device).eval()
            try:
                with torch.inference_mode():
                    for row, a, b in zip(rows, source, reconstructed):
                        if metric == "clip":
                            x = preprocess(Image.fromarray(a)).unsqueeze(0).to(self.device)
                            y = preprocess(Image.fromarray(b)).unsqueeze(0).to(self.device)
                            similarity = torch.nn.functional.cosine_similarity(
                                model.encode_image(x), model.encode_image(y), dim=1).item()
                            value = (similarity + 1) / 2
                        elif metric == "lpips_vgg":
                            x, y = (lpips.im2tensor(v).to(self.device) for v in (a, b))
                            value = model(x, y).item()
                        else:
                            x, y = (to_tensor(Image.fromarray(v)).unsqueeze(0).to(self.device)
                                    for v in (a, b))
                            value = model(x, y).item()
                        row[metric] = float(value)
            finally:
                del model
                gc.collect()
                if str(self.device).startswith("cuda"):
                    torch.cuda.empty_cache()
        self.resources = {"device": str(self.device), "batch_size": 1,
                          "model_loading": "sequential",
                          "peak_allocated_bytes": (torch.cuda.max_memory_allocated()
                              if str(self.device).startswith("cuda") else None)}
        return summarize(rows, source.shape), rows
