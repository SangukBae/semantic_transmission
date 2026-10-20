"""Bounded RX-key correction with a generated-motion gate; no source metrics."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .ltx_conditioning import policy, scene_change
from .receiver_key_context import PARAMETERS, preserve_context
from .sparse_trajectory import gray


def motion(frames):
    images = [gray(f) for f in frames]
    return np.asarray([cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 15, 3, 5, 1.2, 0)
                       for a, b in zip(images, images[1:])])


def motion_gate(before, after, params):
    """Protect existing interior motion; received endpoints remain authoritative."""
    # Transitions touching exact received endpoints are audited separately.
    if len(before) <= 2:
        return True, dict(reason='no interior transitions')
    a, b = np.linalg.norm(before[1:-1], axis=-1), np.linalg.norm(after[1:-1], axis=-1)
    active = a > .15
    ratio = float(b.sum()/max(float(a.sum()), 1e-9)) if a.sum() > 1e-9 else 1.
    frozen = float(((b < .25*a) & active).sum()/max(1, active.sum()))
    difference = float(np.linalg.norm(after[1:-1]-before[1:-1], axis=-1).mean())
    tolerance = params['postprocess_motion_tolerance']
    accept = (1-tolerance <= ratio <= 1+tolerance
              and frozen <= params['postprocess_max_new_freeze_fraction']
              and difference <= max(.15, tolerance*float(a.mean())))
    return bool(accept), dict(motion_ratio=ratio, new_freeze_fraction=frozen, flow_change=difference)


def refine_segment(frames, keys, fps, params=None):
    params = policy() if params is None else params
    if frames.dtype != np.uint8 or len(frames) < 2 or fps <= 0 or len(keys) != 2:
        raise ValueError('uint8 frames, two received keys and positive FPS required')
    if any(k.dtype != np.uint8 or k.shape != frames[0].shape for k in keys):
        raise ValueError('received key shape/dtype mismatch')
    anchored = frames.copy()
    anchored[0], anchored[-1] = keys
    cut = scene_change(keys[0], keys[1], params)
    if cut['suspected_cut'] or np.array_equal(frames[[0, -1]], np.asarray(keys)):
        return anchored, dict(blend=0., reason='suspected cut' if cut['suspected_cut'] else 'keys already exact', cut=cut)
    context = dict(PARAMETERS, min_radius=0,
                   max_radius=max(1, int(round(fps*params['postprocess_max_seconds']))))
    candidate, context_audit = preserve_context(frames, keys, params=context)
    original_motion = motion(anchored)
    attempts = []
    for strength in params['postprocess_blends']:
        result = np.rint(anchored.astype(float) + strength*(candidate.astype(float)-anchored)).clip(0, 255).astype(np.uint8)
        result[0], result[-1] = keys
        accepted, decision = motion_gate(original_motion, motion(result), params)
        # Zero correction is an exact identity on interior frames.
        attempts.append(dict(blend=strength, accepted=accepted, **decision))
        if accepted or strength == 0:
            return result, dict(blend=strength, attempts=attempts, context=context_audit, cut=cut,
                                endpoints_exact=True, receiver_reads_source=False)
    raise AssertionError('policy must include an identity fallback')


def refine(frame_paths, keys, key_directory, destination, fps, callback=None):
    from .bsu_sequence_refinement import validate_keys
    validate_keys(keys, len(frame_paths))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    rows = []
    def rgb(path):
        with Image.open(path) as image:
            return np.asarray(image.convert('RGB'))
    for i, (a, b) in enumerate(zip(keys, keys[1:])):
        frames = np.stack([rgb(p) for p in frame_paths[a:b+1]])
        received = [rgb(Path(key_directory)/f'{k}.png') for k in (a, b)]
        output, audit = refine_segment(frames, received, fps)
        for offset, pixels in enumerate(output):
            if i and offset == 0:
                if not np.array_equal(rgb(destination/f'{a:05d}.png'), pixels):
                    raise ValueError('shared received boundary differs')
                continue
            Image.fromarray(pixels).save(destination/f'{a+offset:05d}.png')
        rows.append(dict(start=a, end=b, **audit))
        if callback:
            callback(i+1, len(keys)-1)
    return rows
