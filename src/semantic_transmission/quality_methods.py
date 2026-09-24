"""Training-free quality prototypes; receiver functions never read source video."""
import bisect

import cv2
import numpy as np


def as_u8(value):
    return np.clip(np.rint(value), 0, 255).astype(np.uint8)


def warp(values, flow):
    h, w = flow.shape[:2]
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(values, x + flow[..., 0], y + flow[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def flow_between(a, b, width=144):
    h, w = a.shape[:2]
    small = (min(width, w), max(8, round(h * min(width, w) / w)))
    x, y = [cv2.cvtColor(cv2.resize(v, small), cv2.COLOR_RGB2GRAY) for v in (a, b)]
    flow = cv2.calcOpticalFlowFarneback(x, y, None, .5, 3, 15, 3, 5, 1.2, 0)
    flow = cv2.resize(flow, (w, h))
    flow[..., 0] *= w / small[0]
    flow[..., 1] *= h / small[1]
    return flow


def anchor_correction(generated, received_keys, strength, progress=lambda *_: None):
    """Propagate bounded low-frequency anchor residuals with FB confidence.

    This is a prototype, not a claim to recover unseen events. Correspondences
    are estimated only from generated frames; noisy receiver anchors can hurt.
    """
    if not 0 <= strength <= 1:
        raise ValueError("correction strength must lie in [0,1]")
    indices = sorted(received_keys)
    if not indices or indices[0] != 0 or indices[-1] != len(generated) - 1:
        raise ValueError("receiver anchors must cover both video endpoints")
    if any(received_keys[i].shape != generated[i].shape for i in indices):
        raise ValueError("receiver anchor geometry mismatch")
    if strength == 0:
        progress(len(generated), len(generated))
        return generated.copy()
    residuals = {i: np.clip(cv2.GaussianBlur(received_keys[i].astype(np.float32) -
                         generated[i].astype(np.float32), (0, 0), 2), -32, 32) for i in indices}
    output = np.empty_like(generated)
    for t, frame in enumerate(generated):
        if t in residuals:
            adjustment = residuals[t]
        else:
            p = bisect.bisect_right(indices, t)
            left, right = indices[p-1:p+1]
            adjustment = np.zeros_like(frame, dtype=np.float32)
            # No division by confidence: unreliable/occluded pixels get less correction.
            for anchor, temporal_weight in ((left, (right-t)/(right-left)),
                                             (right, (t-left)/(right-left))):
                backward = flow_between(frame, generated[anchor])
                forward = flow_between(generated[anchor], frame)
                error = np.linalg.norm(backward + warp(forward, backward), axis=-1)
                photometric = np.mean(np.abs(frame.astype(np.float32) -
                                       warp(generated[anchor].astype(np.float32), backward)), axis=-1)
                valid = warp(np.ones(frame.shape[:2], np.float32), backward) > .99
                confidence = np.exp(-error / 2) * np.exp(-photometric / 20) * valid
                adjustment += temporal_weight * confidence[..., None] * warp(residuals[anchor], backward)
        output[t] = as_u8(frame.astype(np.float32) + strength * adjustment)
        progress(t+1, len(generated))
    return output


def low_resolution_constraint(generated, received_lowres, strength=.5, progress=lambda *_: None):
    """Soft low-frequency correction from decoded, transmitted low-res video."""
    if len(generated) != len(received_lowres) or not 0 <= strength <= 1:
        raise ValueError("low-res constraint length/strength mismatch")
    output = np.empty_like(generated)
    for i, (frame, low) in enumerate(zip(generated, received_lowres)):
        reduced = cv2.resize(frame.astype(np.float32), (low.shape[1], low.shape[0]), interpolation=cv2.INTER_AREA)
        delta = cv2.resize(low.astype(np.float32) - reduced, (frame.shape[1], frame.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        output[i] = as_u8(frame.astype(np.float32) + strength * delta)
        progress(i+1, len(generated))
    return output


def rank_insertions(source, original, limit, adaptive=True):
    """Sender-only greedy interpolation-error ranking or widest-gap midpoints.

    Existing semantic boundaries are retained so automatic captions remain valid.
    Ranking does not use reconstructed test outputs or source-based receiver choices.
    """
    indices = list(original)
    if indices != sorted(set(indices)) or indices[0] != 0 or indices[-1] != len(source)-1:
        raise ValueError("invalid original keyframe coverage")
    frames = np.stack([cv2.resize(f, (72, 40)).astype(np.float32) for f in source])
    result = []
    for _ in range(min(limit, len(source)-len(indices))):
        candidates = []
        for a, b in zip(indices, indices[1:]):
            if b-a < 2:
                continue
            if adaptive:
                for t in range(a+1, b):
                    weight = (t-a)/(b-a)
                    estimate = frames[a]*(1-weight) + frames[b]*weight
                    candidates.append((float(np.mean(np.abs(frames[t]-estimate))), -t, t))
            else:
                t = (a+b)//2
                candidates.append((b-a, -t, t))
        if not candidates:
            break
        selected = max(candidates)[2]
        result.append(selected)
        bisect.insort(indices, selected)
    return result


def temporal_error(source, output):
    """Reference-paired frame-difference error; not a hallucination detector."""
    if len(source) < 2:
        return 0.0
    a = np.diff(source.astype(np.float32), axis=0)
    b = np.diff(output.astype(np.float32), axis=0)
    return float(np.mean(np.abs(a-b))/255)


def choose_strength(rows):
    """Tune on development only, including no-op and explicit quality guards."""
    base = rows["0.0"]["interior"]
    if base is None or "lpips_vgg" not in base:
        return 0.0
    candidates = [(base["lpips_vgg"], 0.0)]
    for key, metrics in rows.items():
        interior = metrics["interior"]
        if interior and interior["psnr_db"] >= base["psnr_db"]-.1 and (
            interior["lpips_vgg"] <= base["lpips_vgg"]-.002
            and metrics["temporal_error"] <= rows["0.0"]["temporal_error"]+.002
        ):
            candidates.append((interior["lpips_vgg"], float(key)))
    return min(candidates)[1]
