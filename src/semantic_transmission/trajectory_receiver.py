"""Received-only BSU output refinement with trajectory and temporal ablations.

No source images, learned weights, scene-specific parameters or original-metric
selection enter this module. Uncertain correspondence preserves BSU output.
"""
import cv2
import numpy as np

from .sparse_trajectory import SIZE, gray

PARAMETERS = dict(reference_strength=.65, temporal_strength=.85, temporal_stride=4,
                  spatial_radius=48., photometric_scale=18., cycle_scale=1.5,
                  scene_cut_mad=45., max_displacement=72., endpoint_ramp=4)


def remap(image, coordinates):
    return cv2.remap(image, coordinates[..., 0].astype(np.float32), coordinates[..., 1].astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def grid(height, width):
    y, x = np.mgrid[:height, :width]
    return np.stack([x, y], axis=-1).astype(np.float32)


def warp_high(image, coordinates):
    h, w = image.shape[:2]
    displacement = coordinates-grid(SIZE[1],SIZE[0])
    up = grid(h,w)+cv2.resize(displacement,(w,h))*np.array([w/SIZE[0],h/SIZE[1]],np.float32)
    return remap(image, up)


def in_bounds(coordinates):
    return ((coordinates[..., 0] >= 0) & (coordinates[..., 0] <= SIZE[0]-1)
            & (coordinates[..., 1] >= 0) & (coordinates[..., 1] <= SIZE[1]-1))


def backward_map(target, reference, params=PARAMETERS):
    """Fit camera similarity plus spatially local point residuals, on RX tracks."""
    g = grid(SIZE[1], SIZE[0])
    if len(target) < 3:
        return g, np.zeros(g.shape[:2], np.float32)
    matrix, inliers = cv2.estimateAffinePartial2D(target, reference, method=cv2.RANSAC,
                                                 ransacReprojThreshold=2., maxIters=500)
    if matrix is None or not np.isfinite(matrix).all():
        return g, np.zeros(g.shape[:2], np.float32)
    singular = np.linalg.svd(matrix[:, :2], compute_uv=False)
    if singular.min() < .65 or singular.max() > 1.5:
        return g, np.zeros(g.shape[:2], np.float32)
    mapped = g @ matrix[:, :2].T + matrix[:, 2]
    residual = reference - (target @ matrix[:, :2].T + matrix[:, 2])
    weight = np.exp(-np.sum((g[:, :, None]-target[None, None])**2, axis=-1)/(2*params['spatial_radius']**2))
    correction = (weight[..., None]*residual[None, None]).sum(axis=2)/(weight.sum(axis=2)[..., None]+.25)
    mapped += correction
    coverage = np.clip(weight.sum(axis=2), 0, 1)
    coverage *= in_bounds(mapped) & (np.linalg.norm(mapped-g, axis=2) < params['max_displacement'])
    return mapped.astype(np.float32), coverage.astype(np.float32)


def refine_references(generated, keys, trajectory, progress=None, params=PARAMETERS):
    if len(keys) != 2 or keys[0].shape != generated[0].shape or keys[1].shape != generated[0].shape:
        raise ValueError('need two matching received endpoint images')
    if trajectory is None:
        return generated.copy(), dict(bypassed=True, reason='no paid trajectory', mean_reference_weight=0.)
    if trajectory['frames'] != len(generated):
        raise ValueError('trajectory/generation timeline mismatch')
    output = generated.copy(); weights = []
    for t in range(len(generated)):
        candidates, confidence = [], []
        for endpoint in (0, 1):
            origin_t = 0 if endpoint == 0 else len(generated)-1
            keep = ((trajectory['origins'] == endpoint) & trajectory['valid'][t]
                    & trajectory['valid'][origin_t])
            coords, conf = backward_map(trajectory['xy'][t, keep], trajectory['xy'][origin_t, keep], params)
            candidates.append(warp_high(keys[endpoint], coords).astype(np.float32))
            confidence.append(conf)
        a, b = candidates
        difference = np.mean(np.abs(cv2.resize(a, SIZE)-cv2.resize(b, SIZE)), axis=2)
        agreement = np.exp(-(difference/params['photometric_scale'])**2)
        alpha = params['reference_strength']*np.minimum(*confidence)*agreement
        # Generated endpoints retain their exact pixels, with a smooth edge ramp.
        ramp = min(1., t/params['endpoint_ramp'], (len(generated)-1-t)/params['endpoint_ramp'])
        alpha = cv2.GaussianBlur(alpha, (0,0), 2)*ramp
        u = t/(len(generated)-1)
        bridge = (1-u)*a+u*b
        alpha = cv2.resize(alpha, (generated.shape[2], generated.shape[1]))[..., None]
        output[t] = np.rint(generated[t]*(1-alpha)+bridge*alpha).clip(0,255).astype(np.uint8)
        weights.append(float(alpha.mean()))
        if progress is not None: progress(t+1, len(generated))
    return output, dict(bypassed=False, mean_reference_weight=float(np.mean(weights)), per_frame_weight=weights)


def interpolate(left, right, fraction, params=PARAMETERS):
    a, b = gray(left), gray(right)
    g = grid(*a.shape)
    if np.mean(np.abs(a.astype(float)-b)) > params['scene_cut_mad']:
        return left.astype(np.float32), np.zeros(a.shape, np.float32)
    engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    forward = engine.calc(a, b, None)
    backward = engine.calc(b, a, None)
    # Invert fractional forward displacement with fixed-point iterations.
    left_map, right_map = g.copy(), g.copy()
    for _ in range(3):
        left_map = g-fraction*remap(forward, left_map)
        right_map = g-(1-fraction)*remap(backward, right_map)
    cycle_a = np.linalg.norm(forward+remap(backward, g+forward), axis=2)
    cycle_b = np.linalg.norm(backward+remap(forward, g+backward), axis=2)
    consistency = np.exp(-(remap(cycle_a,left_map)+remap(cycle_b,right_map))/params['cycle_scale'])
    wa, wb = warp_high(left,left_map).astype(np.float32), warp_high(right,right_map).astype(np.float32)
    difference = np.mean(np.abs(cv2.resize(wa,SIZE)-cv2.resize(wb,SIZE)),axis=2)
    confidence = consistency*np.exp(-(difference/params['photometric_scale'])**2)
    confidence *= in_bounds(left_map)&in_bounds(right_map)
    return (1-fraction)*wa+fraction*wb, confidence


def refine_temporal(frames, progress=None, params=PARAMETERS):
    """Four-frame motion interpolation; no recursion or output FPS change.

    This bounded constant-velocity assumption can miss real sub-4-frame motion;
    source-paired freeze/acceleration measurements are therefore mandatory.
    """
    output=frames.copy(); weights=[]
    for t in range(len(frames)):
        left=(t//params['temporal_stride'])*params['temporal_stride']
        right=min(left+params['temporal_stride'],len(frames)-1)
        if left<t<right:
            candidate,confidence=interpolate(frames[left],frames[right],(t-left)/(right-left),params)
            alpha=cv2.resize(confidence,(frames.shape[2],frames.shape[1]))[...,None]*params['temporal_strength']
            output[t]=np.rint(frames[t]*(1-alpha)+candidate*alpha).clip(0,255).astype(np.uint8)
            weights.append(float(alpha.mean()))
        else:weights.append(0.)
        if progress is not None:progress(t+1,len(frames))
    return output,dict(mean_temporal_weight=float(np.mean(weights)),per_frame_weight=weights)
