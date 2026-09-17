"""Packet-backed STA simulator and RGB-only representation adapter.

Dense PNG keyframes and a zero-order-hold raster decoder are a controlled
surrogate, not the LGVSC decoder. Intervention names are confined to generation.
The scoring adapter receives decoded RGB, received packet status, and no truth.
"""
import hashlib

import cv2
import numpy as np

from .attribution_metric import attribution
from .event_metric import event_pairs
from .object_metric import score_tracks

CLASSES = ('edr', 'clr', 'gfr', 'ghr')
KINDS = ('normal', 'tx_omission', 'packet_loss', 'bit_corruption', 'generation_freeze', 'unsupported_addition')
LABEL = dict(normal='normal', tx_omission='edr', packet_loss='clr', bit_corruption='clr',
             generation_freeze='gfr', unsupported_addition='ghr')


def encode_video(frames):
    packets = []
    for frame in frames:
        ok, value = cv2.imencode('.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError('PNG encoding failed')
        payload = value.tobytes()
        packets.append({'payload': payload, 'sha256': hashlib.sha256(payload).hexdigest()})
    return packets


def receive(packets):
    """Read only transmitted bytes; recover missing slots by last valid image."""
    decoded, statuses = [], []
    for p in packets:
        value, status = None, 'packet_lost'
        if p is not None:
            raw = p['payload']
            if hashlib.sha256(raw).hexdigest() != p['sha256']:
                status = 'checksum_failed'
            else:
                try:
                    value = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                except cv2.error:
                    value = None
                status = 'ok' if value is not None else 'decode_failed'
                if value is not None:
                    value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
        decoded.append(value); statuses.append(status)
    first = next((x for x in decoded if x is not None), None)
    if first is None:
        raise ValueError('no readable keyframe: cannot infer raster dimensions')
    current = first
    frames = []
    indices = []
    current_index = next(i for i, x in enumerate(decoded) if x is not None)
    for i, x in enumerate(decoded):
        if x is not None:
            current, current_index = x, i
        frames.append(current.copy()); indices.append(current_index)
    return np.stack(frames), statuses, indices


def simulate(frames, kind, severity, encoded=None):
    if kind not in KINDS:
        raise ValueError(kind)
    source_packets = encode_video(frames) if encoded is None else encoded
    tx = list(source_packets)
    length = max(2, round(len(frames) * severity))
    start = (len(frames) - length) // 2
    stop = start + length
    if kind == 'tx_omission':
        tx[start:stop] = [None] * length
    rx = list(tx)
    if kind == 'packet_loss':
        rx[start:stop] = [None] * length
    elif kind == 'bit_corruption':
        for i in range(start, stop):
            raw = bytearray(rx[i]['payload']); raw[len(raw) // 2] ^= 1
            rx[i] = {**rx[i], 'payload': bytes(raw)}
    tx_video, tx_status, tx_indices = receive(tx)
    rx_video, rx_status, rx_indices = receive(rx)
    rec = rx_video.copy()
    rec_indices = list(rx_indices)
    if kind == 'generation_freeze':
        rec[start:stop] = rec[start - 1]
        rec_indices[start:stop] = [start - 1] * length
    addition_masks = np.zeros(rec.shape[:3], bool)
    if kind == 'unsupported_addition':
        yy, xx = np.mgrid[:rec.shape[1], :rec.shape[2]]
        for t in range(start, stop):
            # Procedural, fixed new object in the simulator's free lower-left region.
            mask = (xx - 60) ** 2 + (yy - 142) ** 2 <= 18 ** 2
            detail = 8 * np.sin((xx - 60) * .3) * np.cos((yy - 142) * .3)
            color = np.clip(np.array([205, 170, 120]) + detail[..., None], 0, 255).astype(np.uint8)
            rec[t, mask] = color[mask]; addition_masks[t] = mask
    # Transport payloads are retained for independent byte-level replay.
    return {'tx_frames': tx_video, 'rx_frames': rx_video, 'reconstruction': rec,
            'tx_status': tx_status, 'rx_status': rx_status, 'tx_indices': tx_indices,
            'rx_indices': rx_indices, 'reconstruction_indices': rec_indices,
            'tx_packets': tx, 'rx_packets': rx, 'addition_masks': addition_masks,
            'interval': [start, stop], 'caption_policy': 'none; support tested from decoded pixels'}


def _tokens_for_tracks(reference_tracks, reconstruction_tracks, observed_tracks):
    tokens = {}
    for i, j in score_tracks(reference_tracks, observed_tracks).get('track_pairs', []):
        tokens.setdefault(j, set()).add('reference' + str(i))
    for i, j in score_tracks(reconstruction_tracks, observed_tracks).get('track_pairs', []):
        tokens.setdefault(j, set()).add('reconstructed' + str(i))
    return tokens


def representation(observation, statuses, reference_tracks, reconstruction_tracks, fps=8.):
    tokens = _tokens_for_tracks(reference_tracks, reconstruction_tracks, observation['tracks'])
    keyframes = []
    for t, status in enumerate(statuses):
        keyframes.append({'time_s': t / fps, 'decode_status': status, 'content_source': 'decoded_video_pixels',
            'decoded_events': [e for e in observation['events'] if abs(e['time_s'] - t / fps) <= 1e-9] if status == 'ok' else [],
            'decoded_object_tokens': sorted({token for j, track in enumerate(observation['tracks']) if t in track
                                           for token in tokens.get(j, ())}) if status == 'ok' else []})
    return {'keyframes': keyframes, 'captions': []}


def score_video_attribution(observer, reference, reconstruction, tx_frames, rx_frames, tx_status, rx_status):
    """All inputs are observable RGB/status; intervention labels are not accepted."""
    a, b, tx, rx = [observer.observe(x) for x in (reference, reconstruction, tx_frames, rx_frames)]
    units_a = [{**e, 'kind': 'event'} for e in a['events']]
    units_b = [{**e, 'kind': 'event'} for e in b['events']]
    pairs = event_pairs(a['events'], b['events'])
    offset_a, offset_b = len(units_a), len(units_b)
    units_a += [{'kind': 'object', 'object_token': 'reference' + str(i)} for i in range(len(a['tracks']))]
    units_b += [{'kind': 'object', 'object_token': 'reconstructed' + str(i)} for i in range(len(b['tracks']))]
    pairs += [(offset_a + i, offset_b + j) for i, j in score_tracks(a['tracks'], b['tracks']).get('track_pairs', [])]
    tx_rep = representation(tx, tx_status, a['tracks'], b['tracks'])
    rx_rep = representation(rx, rx_status, a['tracks'], b['tracks'])
    result = attribution(units_a, units_b, tx_rep, rx_rep, pairs)
    return {**result, 'units_reference': units_a, 'units_reconstruction': units_b,
            'tx_representation': tx_rep, 'rx_representation': rx_rep,
            'unit_policy': 'union of RGB-predicted events and whole object tracks; equal unit weights'}
