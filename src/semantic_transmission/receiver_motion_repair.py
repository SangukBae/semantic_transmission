"""Experimental receiver repairs using generated RGB and paid received tracks.

No original frames, captions, evaluation scores or per-video settings enter this
module. These are constrained post-decoder candidates, not generator retraining
or a guarantee of source-semantic recovery.
"""
import cv2
import numpy as np

from .sparse_trajectory import SIZE, gray
from .trajectory_receiver import (
    PARAMETERS as LEGACY, backward_map, grid, remap, warp_high, refine_temporal,
)

POLICY = 'received_track_geometry_and_motion_retention_v1'
PARAMETERS = dict(active_speed=.15, minimum_motion_ratio=.90,
                  cycle_threshold=1.5, photometric_threshold=24.,
                  maximum_track_correction=24., warp_strength=.75,
                  minimum_tracks=4, spatial_radius=32., endpoint_ramp=4)


def validate_video(frames):
    if (frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3
            or len(frames) < 2):
        raise ValueError('nonempty uint8 RGB segment with at least two frames required')


def motion_cells(a, b):
    flow = cv2.calcOpticalFlowFarneback(gray(a), gray(b), None, .5, 3, 15, 3, 5, 1.2, 0)
    return np.linalg.norm(flow, axis=2).reshape(5, 32, 8, 36).mean(axis=(1, 3))


def retain_motion(original, smoothed, params=PARAMETERS):
    """Use one spatial retention mask per four-frame block to avoid flicker.

This protects generated motion, which may itself be wrong. It never treats
generated motion as source ground truth. Endpoint frames stay byte-identical.
"""
    validate_video(original)
    if smoothed.shape != original.shape or smoothed.dtype != np.uint8:
        raise ValueError('matching candidate RGB required')
    out = smoothed.copy(); audits = []
    for left in range(0, len(original)-1, LEGACY['temporal_stride']):
        right = min(left+LEGACY['temporal_stride'], len(original)-1)
        loss = np.zeros((5, 8), np.float32)
        for t in range(left+1, right+1):
            before = motion_cells(original[t-1], original[t])
            after = motion_cells(smoothed[t-1], smoothed[t])
            active = before > params['active_speed']
            deficit = np.clip(1-after/np.maximum(params['minimum_motion_ratio']*before, 1e-6), 0, 1)
            loss = np.maximum(loss, active*deficit)
        weight = cv2.resize(loss, (original.shape[2], original.shape[1]),
                            interpolation=cv2.INTER_LINEAR)[..., None]
        for t in range(left+1, right):
            out[t] = np.rint(smoothed[t]*(1-weight)+original[t]*weight).clip(0, 255).astype(np.uint8)
        audits.append(dict(start=left, end=right, mean_restored_weight=float(weight.mean())))
    out[[0, -1]] = original[[0, -1]]
    return out, dict(policy=POLICY, blocks=audits, receiver_reads_source=False,
                     limitation='preserves generated motion, not proof of correct source motion')


def adaptive_cadence(frames, progress=None):
    smooth, old = refine_temporal(frames, progress)
    out, audit = retain_motion(frames, smooth)
    return out, dict(legacy=old, retention=audit)


def sample(field, points):
    return cv2.remap(field, points[:, 0].astype(np.float32).reshape(-1, 1),
                     points[:, 1].astype(np.float32).reshape(-1, 1),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101).reshape(len(points), -1)


def warp_to_received_tracks(frames, keys, trajectory, progress=None, params=PARAMETERS):
    """Correct RGB coordinates once, instead of overlaying displaced contours.

Received tracks give desired positions. Bidirectional optical flow estimates
where their endpoint appearances landed in the generated frame. Only mutually
consistent, photometrically supported correspondences can move generated RGB.
    """
    validate_video(frames)
    if len(keys) != 2 or any(k.shape != frames[0].shape or k.dtype != np.uint8 for k in keys):
        raise ValueError('two matching received keys required')
    if trajectory is None:
        return frames.copy(), dict(policy=POLICY, bypassed=True, reason='no received trajectory')
    if (trajectory['frames'] != len(frames) or not np.isfinite(trajectory['xy']).all()
            or trajectory['xy'].shape[:2] != trajectory['valid'].shape):
        raise ValueError('invalid received trajectory timeline or coordinates')
    output = frames.copy(); audits = []; g = grid(SIZE[1], SIZE[0])
    key_gray = [gray(k) for k in keys]
    map_params = dict(LEGACY, spatial_radius=params['spatial_radius'],
                      max_displacement=params['maximum_track_correction'])
    for t in range(1, len(frames)-1):
        generated = gray(frames[t]); targets = []; predictions = []
        for endpoint, origin in ((0, 0), (1, len(frames)-1)):
            keep = ((trajectory['origins'] == endpoint) & trajectory['valid'][origin]
                    & trajectory['valid'][t])
            source = trajectory['xy'][origin, keep].astype(np.float32)
            desired = trajectory['xy'][t, keep].astype(np.float32)
            if not len(source):
                continue
            engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            forward = engine.calc(key_gray[endpoint], generated, None)
            backward = engine.calc(generated, key_gray[endpoint], None)
            predicted = source+sample(forward, source)
            cycle = np.linalg.norm(predicted+sample(backward, predicted)-source, axis=1)
            photo = np.abs(sample(key_gray[endpoint].astype(np.float32), source)[:, 0]
                           -sample(generated.astype(np.float32), predicted)[:, 0])
            valid = ((cycle <= params['cycle_threshold']) & (photo <= params['photometric_threshold'])
                     & (np.linalg.norm(predicted-desired, axis=1) <= params['maximum_track_correction']))
            for coordinates in (source, desired, predicted):
                valid &= ((coordinates[:, 0] >= 0) & (coordinates[:, 0] < SIZE[0])
                          & (coordinates[:, 1] >= 0) & (coordinates[:, 1] < SIZE[1]))
            targets.extend(desired[valid]); predictions.extend(predicted[valid])
        distance = 0.
        if len(targets) >= params['minimum_tracks']:
            target = np.asarray(targets, np.float32); prediction = np.asarray(predictions, np.float32)
            if np.max(np.abs(target-prediction)) > .05:
                mapped, confidence = backward_map(target, prediction, map_params)
                ramp = min(1., t/params['endpoint_ramp'], (len(frames)-1-t)/params['endpoint_ramp'])
                correction = (mapped-g)*confidence[..., None]*params['warp_strength']*ramp
                output[t] = warp_high(frames[t], g+correction)
                distance = float(np.linalg.norm(correction, axis=2).mean())
        audits.append(dict(frame=t, supported_tracks=len(targets), mean_correction_pixels_288=distance))
        if progress:
            progress(t+1, len(frames))
    return output, dict(policy=POLICY, bypassed=False, parameters=params, frames=audits,
                        receiver_reads_source=False, additional_transmission=0,
                        limitation='cannot recover unseen appearance; unreliable correspondences are skipped')
