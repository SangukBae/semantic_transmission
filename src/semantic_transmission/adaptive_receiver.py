"""Received-only adaptive cadence and visibility-weighted endpoint synthesis.

Parameters are shared across videos. No source pixels or quality scores enter
either operator. These are experimental alternatives to the frozen receiver.
"""
import cv2
import numpy as np

from .sparse_trajectory import SIZE, gray
from .trajectory_receiver import (PARAMETERS, backward_map, grid, in_bounds,
                                  interpolate, remap, warp_high)

CADENCE = dict(spans=(8, 4, 2, 1), confidence=(.85, .65, .35),
               speed=(2., 4., 8.), acceleration=(.15, .65, 1.5),
               track_linearity=(.5, 1., 2.), minimum_motion=.1)
VISIBILITY = dict(photometric_scale=32., selection_power=3., history=.5,
                  confidence_floor=.05, cycle_scale=1.5)


def flow_pair(a, b):
    engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return engine.calc(a, b, None), engine.calc(b, a, None)


def motion_profile(frames):
    images = [gray(f) for f in frames]
    g = grid(*images[0].shape)
    rows, previous = [], None
    for a, b in zip(images, images[1:]):
        forward, backward = flow_pair(a, b)
        target = g + forward
        cycle = np.linalg.norm(forward + remap(backward, target), axis=2)
        photo = np.abs(a.astype(float) - remap(b, target))
        confidence = np.exp(-cycle/1.5) * np.exp(-(photo/18.)**2) * in_bounds(target)
        speed = np.linalg.norm(forward, axis=2)
        acceleration = 0. if previous is None else float(np.percentile(np.abs(speed-previous), 90))
        rows.append(dict(speed=float(np.percentile(speed, 90)), acceleration=acceleration,
                         confidence=float(confidence.mean()),
                         cut=bool(np.mean(np.abs(a.astype(float)-b)) > PARAMETERS['scene_cut_mad'])))
        previous = speed
    return rows


def track_linearity(trajectory, a, b):
    if trajectory is None or not trajectory['xy'].shape[1]:
        return None
    xy, valid = trajectory['xy'], trajectory['valid']
    usable = valid[a:b+1] & valid[a] & valid[b]
    if not usable.any():
        return None
    u = np.linspace(0, 1, b-a+1)[:, None, None]
    error = np.linalg.norm(xy[a:b+1] - ((1-u)*xy[a] + u*xy[b]), axis=2)
    return float(np.percentile(error[usable], 90))


def schedule(profile, trajectory=None, options=CADENCE):
    """Choose spans before interpolation; a span of one preserves that interval."""
    frames = len(profile)+1
    windows, a = [], 0
    while a < frames-1:
        for level, span in enumerate(options['spans']):
            b = min(a+span, frames-1)
            if b-a == 1:
                accepted = True
            else:
                rows = profile[a:b]
                conf = min(r['confidence'] for r in rows)
                speed = max(r['speed'] for r in rows)
                acceleration = max(r['acceleration'] for r in rows)
                error = track_linearity(trajectory, a, b)
                accepted = (not any(r['cut'] for r in rows)
                    and conf >= options['confidence'][level]
                    and speed <= options['speed'][level]
                    and acceleration <= options['acceleration'][level]
                    and (error is None or error <= options['track_linearity'][level])
                    and (span != 8 or speed >= options['minimum_motion']))
            if accepted:
                windows.append((a, b)); a = b; break
    return windows


def refine_temporal(frames, trajectory=None, progress=None):
    if len(frames) < 2:
        raise ValueError('at least two generated frames required')
    if trajectory is not None and trajectory['frames'] != len(frames):
        raise ValueError('trajectory timeline differs')
    profile = motion_profile(frames)
    windows = schedule(profile, trajectory)
    output = frames.copy(); weights = np.zeros(len(frames))
    for a, b in windows:
        for t in range(a+1, b):
            pixels, confidence = interpolate(frames[a], frames[b], (t-a)/(b-a))
            alpha = cv2.resize(confidence, (frames.shape[2], frames.shape[1]))[..., None]
            alpha *= PARAMETERS['temporal_strength']
            output[t] = np.rint(frames[t]*(1-alpha)+pixels*alpha).clip(0,255).astype(np.uint8)
            weights[t] = alpha.mean()
        if progress is not None: progress(b+1, len(frames))
    return output, dict(parameters=CADENCE, windows=windows, motion_profile=profile,
                       span_counts={str(s):sum(b-a==s for a,b in windows) for s in CADENCE['spans']},
                       per_frame_weight=weights.tolist(), mean_temporal_weight=float(weights.mean()))


def endpoint_weights(a, b, generated, confidence, fraction):
    """Choose a visible endpoint when references disagree; retain both if aligned."""
    ca, cb = confidence
    a, b, generated = [cv2.resize(x.astype(np.float32), SIZE) for x in (a,b,generated)]
    mismatch = np.mean(np.abs(a-b), axis=2)
    agreement = np.exp(-(mismatch/PARAMETERS['photometric_scale'])**2)
    support = np.stack([ca*(1-fraction), cb*fraction], axis=-1)
    appearance = np.stack([np.mean(np.abs(x-generated), axis=2) for x in (a,b)], axis=-1)
    fit = np.exp(-(cv2.GaussianBlur(appearance, (0,0), 1)/VISIBILITY['photometric_scale'])**2)
    scores = support * fit
    sharp = scores**VISIBILITY['selection_power']
    sharp /= np.maximum(sharp.sum(axis=-1,keepdims=True), 1e-12)
    plain = support/np.maximum(support.sum(axis=-1,keepdims=True), 1e-12)
    weights = agreement[...,None]*plain+(1-agreement[...,None])*sharp
    dominance = np.abs(sharp[...,0]-sharp[...,1])
    # Existing hard mismatch rejection is not used. Visibility and relative
    # evidence decide which endpoint is usable, with uncertainty reducing alpha.
    strength = agreement*np.minimum(ca,cb)+(1-agreement)*np.maximum(ca,cb)*dominance
    strength *= PARAMETERS['reference_strength']
    return weights, strength


def refine_references(generated, keys, trajectory, progress=None):
    if len(keys)!=2 or any(k.shape != generated[0].shape for k in keys):
        raise ValueError('two matching received endpoints required')
    if trajectory is None:
        return generated.copy(), dict(bypassed=True, reason='no paid trajectory', mean_reference_weight=0.)
    if trajectory['frames'] != len(generated):
        raise ValueError('trajectory timeline differs')
    output = generated.copy(); weights_log = []; one_side = []; previous = None
    previous_gray = None
    g = grid(SIZE[1], SIZE[0])
    for t in range(len(generated)):
        candidates, confidence = [], []
        for endpoint in (0,1):
            origin = 0 if endpoint == 0 else len(generated)-1
            valid = ((trajectory['origins']==endpoint) & trajectory['valid'][t]
                     & trajectory['valid'][origin])
            coordinates, coverage = backward_map(trajectory['xy'][t,valid],
                                                 trajectory['xy'][origin,valid])
            candidates.append(warp_high(keys[endpoint], coordinates).astype(np.float32))
            confidence.append(coverage)
        fraction = t/(len(generated)-1)
        weights, strength = endpoint_weights(*candidates, generated[t], confidence, fraction)
        current_gray = gray(generated[t])
        if previous is not None and np.mean(np.abs(current_gray.astype(float)-previous_gray)) <= PARAMETERS['scene_cut_mad']:
            backward, forward = flow_pair(current_gray, previous_gray)
            coords = g+backward
            cycle = np.linalg.norm(backward+remap(forward,coords),axis=2)
            history = VISIBILITY['history']*np.exp(-cycle/VISIBILITY['cycle_scale'])*in_bounds(coords)
            warped = remap(previous,coords)
            weights = weights*(1-history[...,None])+warped*history[...,None]
        # Do not carry yesterday's visible region through today's occlusion.
        available = np.stack(confidence,axis=-1)>VISIBILITY['confidence_floor']
        weights *= available
        weights /= np.maximum(weights.sum(axis=-1,keepdims=True),1e-12)
        strength *= available.any(axis=-1)
        previous, previous_gray = weights.copy(), current_gray
        ramp = min(1.,t/PARAMETERS['endpoint_ramp'],(len(generated)-1-t)/PARAMETERS['endpoint_ramp'])
        alpha = cv2.resize(strength*ramp,(generated.shape[2],generated.shape[1]))[...,None]
        high_weights = cv2.resize(weights,(generated.shape[2],generated.shape[1]))
        bridge = sum(c*high_weights[...,j:j+1] for j,c in enumerate(candidates))
        output[t] = np.rint(generated[t]*(1-alpha)+bridge*alpha).clip(0,255).astype(np.uint8)
        weights_log.append(float(alpha.mean()))
        one_side.append(float((available.sum(axis=-1)==1).mean()))
        if progress is not None: progress(t+1,len(generated))
    return output, dict(bypassed=False, parameters=VISIBILITY,
        mean_reference_weight=float(np.mean(weights_log)), per_frame_weight=weights_log,
        mean_single_endpoint_area=float(np.mean(one_side)), per_frame_single_endpoint_area=one_side)
