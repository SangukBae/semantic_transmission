"""Training-free LTX conditions from paid received data, with no source access.

Trajectory guides are warped RGB hypotheses, not new transmitted keyframes or
learned trajectory embeddings. Weak/occluded correspondences are omitted.
All policy thresholds are global; inference never reads evaluation scores.
"""
import copy
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import ltx_receiver as ltx
from .artifacts import sha256
from .bsu_sequence_refinement import decode_packet
from .sparse_trajectory import SIZE, gray
from .trajectory_receiver import backward_map, warp_high
from .webvid5 import fingerprint
from .wire import unpack

CONFIG = ltx.REPO / 'configs/bsu_ltx_receiver_v2.json'
VARIANTS = ('baseline', 'postprocess', 'trajectory', 'context', 'prompt', 'all')


def policy():
    return ltx.read(CONFIG)


def flags(variant):
    if variant not in VARIANTS:
        raise ValueError('unknown LTX improvement variant')
    return {name: variant in (name, 'all') for name in VARIANTS[1:-1]}


def prompt(text, flow, clean):
    """Lossless factual text normalization; no LLM, new facts, or flow direction."""
    if not isinstance(text, str) or not text.strip() or not np.isfinite(float(flow)):
        raise ValueError('nonempty received caption and finite motion score required')
    if clean:
        # Do not append Open-Sora's numeric control suffix. Any such words
        # already IN the received caption may be factual and must be retained.
        return ' '.join(text.split())
    return text if 'motion score:' in text else text + f' motion score: {float(flow):.1f}.'


def scene_change(left, right, params):
    """Conservative RX-endpoint cut suspicion, not an oracle scene-cut label."""
    a, b = cv2.resize(left, SIZE), cv2.resize(right, SIZE)
    mad = float(np.abs(a.astype(float) - b).mean())
    hist = []
    for rgb in (left, right):
        small = cv2.resize(rgb, SIZE)
        h = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        hist.append(cv2.normalize(h, h, norm_type=cv2.NORM_L1))
    distance = float(cv2.compareHist(*hist, cv2.HISTCMP_BHATTACHARYYA))
    return dict(suspected_cut=mad > params['scene_cut_mad'] and distance > params['scene_cut_histogram'],
                received_endpoint_mad=mad, received_histogram_distance=distance)


def guide(keys, trajectory, index, params):
    """Conservative bidirectional RGB warp at an actually received time knot."""
    if trajectory is None or index <= 0 or index >= trajectory['frames'] - 1:
        return None, dict(accepted=False, reason='no interior received trajectory')
    candidates, confidence, motions = [], [], []
    for endpoint in (0, 1):
        origin = 0 if endpoint == 0 else trajectory['frames'] - 1
        usable = (trajectory['origins'] == endpoint) & trajectory['valid'][index] & trajectory['valid'][origin]
        if np.count_nonzero(usable) < 3:
            return None, dict(accepted=False, reason='fewer than three visible tracks per endpoint')
        target = trajectory['xy'][index, usable]
        reference = trajectory['xy'][origin, usable]
        cv2.setRNGSeed(2025)
        coordinates, conf = backward_map(target, reference)
        candidates.append(warp_high(keys[endpoint], coordinates).astype(np.float32))
        confidence.append(conf)
        motions.append(float(np.median(np.linalg.norm(target - reference, axis=1))))
    a, b = candidates
    error = np.abs(cv2.resize(a, SIZE) - cv2.resize(b, SIZE)).mean(axis=2)
    joint = np.minimum(*confidence)
    agreement = np.exp(-(error / params['guide_agreement_scale']) ** 2)
    coverage = float((joint >= 0.5).mean())
    confidence_mean = float((joint * agreement).mean())
    accepted = (coverage >= params['guide_min_coverage']
                and confidence_mean >= params['guide_min_confidence']
                and max(motions) >= params['guide_min_motion'])
    audit = dict(accepted=accepted, coverage=coverage, confidence=confidence_mean,
                 endpoint_motion=motions, reason='accepted' if accepted else 'weak coverage, agreement, or motion')
    if not accepted:
        return None, audit
    u = index / (trajectory['frames'] - 1)
    wa, wb = (1-u) * confidence[0], u * confidence[1]
    alpha = cv2.resize(wa / np.maximum(wa + wb, 1e-6), (a.shape[1], a.shape[0]))[..., None]
    pixels = np.rint(alpha * a + (1-alpha) * b).clip(0, 255).astype(np.uint8)
    audit['strength'] = params['guide_max_strength'] * confidence_mean
    return pixels, audit


def guide_times(trajectory, params):
    if trajectory is None:
        return []
    spacing = params['guide_min_spacing']
    available = [int(t) for t in trajectory['times'] if spacing <= t <= trajectory['frames']-1-spacing]
    chosen = []
    for fraction in np.linspace(0, 1, params['guide_count_per_interval']+2)[1:-1]:
        candidates = [t for t in available if all(abs(t-c) >= spacing for c in chosen)]
        if candidates:
            chosen.append(min(candidates, key=lambda t: (abs(t-fraction*(trajectory['frames']-1)), t)))
    return sorted(chosen)


def prepare(packet, key_directory, output, variant='all', trajectory_packet=None, seed=2025):
    """Write a frozen generation request; only received paths are accepted here."""
    from .qwen35_pairs_video import freeze
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    params, enabled = policy(), flags(variant)
    value = ltx.request(packet, key_directory)
    packet, key_directory = Path(packet).resolve(), Path(key_directory).resolve()
    header, _ = unpack(packet.read_bytes())
    keys = [item['index'] for item in header['keyframes']]
    received = {}
    for k in keys:
        with Image.open(key_directory / f'{k}.png') as image:
            received[k] = np.asarray(image.convert('RGB'))
    tracks = {}
    if trajectory_packet is not None:
        path = Path(trajectory_packet).resolve(strict=True)
        tracks = decode_packet(path.read_bytes(), packet.read_bytes())
        value['inputs'][str(path)] = sha256(path)
    original_inputs = copy.deepcopy(value['inputs'])
    implementation = {str(p): sha256(p) for p in (Path(__file__), CONFIG)}
    identity = dict(inputs=original_inputs, policy=params, variant=variant, seed=seed, implementation=implementation)
    freeze(output / 'preparation.json', identity)
    cuts = [scene_change(received[a], received[b], params) for a, b in zip(keys, keys[1:])]
    groups, cursor = [], 0
    while cursor < len(keys)-1:
        stop = cursor+1
        while (enabled['context'] and stop < len(keys)-1 and stop-cursor < params['context_intervals']
               and not any(c['suspected_cut'] for c in cuts[cursor:stop+1])
               and keys[stop+1]-keys[cursor]+1 <= params['context_max_seconds']*header['video']['fps']):
            stop += 1
        groups.append((cursor, stop))
        cursor = stop
    windows, guide_audits = [], []
    for begin, end in groups:
        a, b = keys[begin], keys[end]
        texts = []
        images = [dict(path=str(key_directory/f'{k}.png'), frame_idx=k-a) for k in keys[begin:end+1]]
        for i in range(begin, end):
            segment = header['segments'][i]
            text = prompt(segment['text'], segment['flow'], enabled['prompt'])
            if end-begin > 1:
                lo, hi = (keys[i]-a)/header['video']['fps'], (keys[i+1]-a)/header['video']['fps']
                text = f'From {lo:.3f} to {hi:.3f} seconds: {text}'
            texts.append(text)
            if not enabled['trajectory']:
                continue
            tr = tracks.get((keys[i], keys[i+1]))
            if cuts[i]['suspected_cut'] or tr is None:
                guide_audits.append(dict(interval=i, accepted=False,
                    reason='suspected cut' if cuts[i]['suspected_cut'] else 'no received trajectory'))
                continue
            for t in guide_times(tr, params):
                pixels, audit = guide([received[keys[i]], received[keys[i+1]]], tr, t, params)
                audit.update(interval=i, source_frame=keys[i]+t)
                guide_audits.append(audit)
                if pixels is not None:
                    dest = output / 'guides' / f'{keys[i]+t:06d}.png'
                    dest.parent.mkdir(exist_ok=True)
                    # Preparation is deterministic and frozen before writing guides.
                    if dest.exists():
                        with Image.open(dest) as old:
                            if not np.array_equal(np.asarray(old.convert('RGB')), pixels):
                                raise ValueError('existing derived guide differs')
                    else:
                        Image.fromarray(pixels).save(dest)
                    value['inputs'][str(dest.resolve())] = sha256(dest)
                    images.append(dict(path=str(dest.resolve()), frame_idx=keys[i]+t-a,
                                       strength=audit['strength'], crf=0))
        windows.append(dict(start=a, end=b, frames=b-a+1, generated_frames=ltx.padded_frames(b-a+1),
                            prompt=' '.join(texts), images=sorted(images, key=lambda x: x['frame_idx']), seed=int(seed)))
    value.update(windows=windows, receiver_improvements=dict(policy=params, variant=variant, flags=enabled,
        cuts=cuts, guides=guide_audits, source_inputs=original_inputs, implementation=implementation,
        training=False, additional_transmitted_bytes=0, original_key_indices=keys))
    value['signature'] = fingerprint({k: v for k, v in value.items() if k != 'signature'})
    validate(value)
    freeze(output / 'request.json', value)
    return value


def validate(value):
    ltx.verify_request(value)
    info = value['receiver_improvements']
    if info['policy'] != policy() or info['flags'] != flags(info['variant']):
        raise ValueError('receiver improvement policy changed')
    for path, digest in info['implementation'].items():
        if sha256(path) != digest:
            raise ValueError('conditioning implementation changed')
    cursor = 0
    seen = []
    for w in value['windows']:
        if (w['start'] != cursor or w['end'] <= cursor or w['frames'] != w['end']-cursor+1
                or w['generated_frames'] != ltx.padded_frames(w['frames'])):
            raise ValueError('generation windows do not cover the received timeline')
        indices = []
        for item in w['images']:
            if (set(item)-{'path', 'frame_idx', 'strength', 'crf'} or item['path'] not in value['inputs']
                    or type(item['frame_idx']) is not int or not 0 <= item['frame_idx'] < w['frames']
                    or not 0 < item.get('strength', 1.) <= 1.):
                raise ValueError('invalid received/derived image condition')
            indices.append(item['frame_idx'])
            if 'strength' not in item:
                seen.append(cursor+item['frame_idx'])
        if len(indices) != len(set(indices)) or 0 not in indices or w['frames']-1 not in indices:
            raise ValueError('duplicate image slots or missing received endpoints')
        cursor = w['end']
    if cursor != value['video']['frames']-1 or sorted(set(seen)) != info['original_key_indices']:
        raise ValueError('received key coverage changed')
