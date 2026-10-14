"""Experimental received-only key preservation after guarded RGB refinement.

No source pixels, metric scores or per-video tuning enter this operator. Exact
key replacement is an experimental policy, not a claim of temporal continuity.
"""
import cv2
import numpy as np

from .sparse_trajectory import gray
from .trajectory_receiver import grid, in_bounds, remap, warp_high

ANCHOR = dict(radius=4, cycle_scale=1.5, photometric_scale=18.)


def preserve_keys(frames, keys, progress=None):
    """Restore received RGB at key times; fade aligned corrections nearby."""
    if len(frames) < 2 or len(keys) != 2 or any(k.shape != frames[0].shape for k in keys):
        raise ValueError('two received keys and a matching full segment required')
    output = frames.copy()
    weights = []
    engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    for t in range(len(frames)):
        distances = (t, len(frames)-1-t)
        endpoint = int(distances[1] < distances[0])
        distance = distances[endpoint]
        if distance == 0:
            output[t] = keys[endpoint]
            weights.append(1.)
        elif distance < ANCHOR['radius']:
            target, reference = gray(frames[t]), gray(keys[endpoint])
            backward = engine.calc(target, reference, None)
            forward = engine.calc(reference, target, None)
            coordinates = grid(*target.shape) + backward
            cycle = np.linalg.norm(backward + remap(forward, coordinates), axis=2)
            photo = np.abs(target.astype(float)-remap(reference, coordinates))
            confidence = (np.exp(-cycle/ANCHOR['cycle_scale'])
                          * np.exp(-(photo/ANCHOR['photometric_scale'])**2)
                          * in_bounds(coordinates))
            alpha = cv2.resize(confidence, (frames.shape[2], frames.shape[1]))[..., None]
            alpha *= 1-distance/ANCHOR['radius']
            aligned = warp_high(keys[endpoint], coordinates).astype(np.float32)
            output[t] = np.rint(frames[t]*(1-alpha)+aligned*alpha).clip(0, 255).astype(np.uint8)
            weights.append(float(alpha.mean()))
        else:
            weights.append(0.)
        if progress:
            progress(t+1, len(frames))
    if not all(np.array_equal(output[i], k) for i, k in zip((0, -1), keys)):
        raise ValueError('received key preservation failed')
    return output, dict(parameters=ANCHOR, received_endpoints_exact=True,
                        per_frame_weight=weights, receiver_reads_source=False)
