"""Received-key context correction that retains generated intra-window motion.

The old bridge replaces every interior frame with endpoint interpolation. This
module transports the generated video's deviation from its own endpoint bridge
into the corrected geometry. Original/evaluation pixels never enter inference.
"""
import cv2
import numpy as np

from .receiver_key_context import PARAMETERS as CONTEXT, flows, interpolate_pair, radius
from .sparse_trajectory import gray
from .trajectory_receiver import grid, in_bounds, remap, warp_high

PARAMETERS = dict(cycle_scale=1.5, appearance_scale=48., confidence_blur=1.)


def transfer_residual(corrected, bridge, actual, strength=1., align=True, params=PARAMETERS):
    """Add temporal detail after registering the two interpolation geometries."""
    if not 0 <= strength <= 1:
        raise ValueError('residual strength must be in [0, 1]')
    residual = actual.astype(np.float32)-bridge.astype(np.float32)
    if align and np.any(residual):
        backward, forward = flows(corrected, bridge)
        g = grid(*backward.shape[:2]); coordinates = g+backward
        cycle = np.linalg.norm(backward+remap(forward, coordinates), axis=2)
        photo = np.abs(gray(corrected).astype(float)-remap(gray(bridge), coordinates))
        confidence = np.exp(-cycle/params['cycle_scale']-(photo/params['appearance_scale'])**2)
        confidence *= in_bounds(coordinates)
        confidence = cv2.GaussianBlur(confidence, (0, 0), params['confidence_blur'])
        residual = warp_high(residual, coordinates)
        weight = cv2.resize(confidence, (actual.shape[1], actual.shape[0]))[..., None]
    else:
        weight = np.ones((*actual.shape[:2], 1), np.float32)
    result = corrected.astype(np.float32)+strength*weight*residual
    return np.rint(result).clip(0,255).astype(np.uint8), dict(
        mean_confidence=float(weight.mean()), mean_abs_residual=float(np.abs(residual).mean()))


def preserve_motion_context(frames, keys, progress=None, strength=1., align=True):
    if frames.dtype!=np.uint8 or len(frames)<2 or len(keys)!=2:
        raise ValueError('uint8 segment and two received keys required')
    if any(k.shape!=frames[0].shape or k.dtype!=np.uint8 for k in keys):
        raise ValueError('matching received RGB required')
    if not 0 <= strength <= 1:
        raise ValueError('residual strength must be in [0, 1]')
    output=frames.copy();audit=[]
    for endpoint in (0,1):
        sequence=frames if endpoint==0 else frames[::-1]
        r,decision=radius(sequence,keys[endpoint]);rows=[]
        if r:
            corrected,_=interpolate_pair(keys[endpoint],sequence[r],r)
            bridge,_=interpolate_pair(sequence[0],sequence[r],r)
            for t in range(1,r):
                corrected[t],row=transfer_residual(corrected[t],bridge[t],sequence[t],strength,align)
                rows.append(dict(offset=t,**row))
            if endpoint==0:output[:r+1]=corrected
            else:output[-r-1:]=corrected[::-1]
        output[0 if endpoint==0 else -1]=keys[endpoint]
        audit.append(dict(endpoint=endpoint,**decision,detail=rows))
        if progress:progress(endpoint+1,2)
    return output,dict(policy='received_key_motion_residual_v1',context_parameters=CONTEXT,
        parameters=PARAMETERS,strength=strength,align=align,endpoints=audit,
        received_keys_exact=True,receiver_reads_source=False)
