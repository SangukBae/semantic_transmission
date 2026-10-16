"""Wire format and source-free policies for the experimental unified BSU codec.

The cited papers motivate design choices; this is not a reproduction or a port
of their checkpoints. Every receiver condition is in the paid packet or derived
from its decoded video. Source-dependent decisions are confined to the sender.
"""
import hashlib
import json
import math
from pathlib import Path
import zlib

import numpy as np

from .wire import pack, unpack
from .packets import accounting

POLICY = 'bsu_unified_received_video_v1'
REPO = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def profile(path=None):
    p = read(path or REPO / 'configs/bsu_unified.json')
    if p['policy'] != POLICY:
        raise ValueError('unsupported unified BSU profile')
    for key, allowed in [('qps', range(64)), ('strides', (1, 2, 3, 4)), ('scales', (1, 2))]:
        if not p[key] or any(type(x) is not int or x not in allowed for x in p[key]):
            raise ValueError('invalid ' + key)
    if p['window_frames'] != 17 or not 0 < p['window_overlap'] < 17:
        raise ValueError('this VAE profile uses 17-frame windows with overlap')
    for key, low, high in [('start_noise', 0, 1), ('cfg_scale', 0, 20),
                           ('max_rgb_correction', 0, 16), ('motion_threshold', 0, 255),
                           ('cut_threshold', 0, 255)]:
        if not isinstance(p[key], (int, float)) or not math.isfinite(p[key]) or not low < p[key] <= high:
            raise ValueError('invalid ' + key)
    if type(p['diffusion_steps']) is not int or not 2 <= p['diffusion_steps'] <= 100:
        raise ValueError('invalid sampling steps')
    if type(p['adapter_width']) is not int or not 1 <= p['adapter_width'] <= 128:
        raise ValueError('invalid adapter width')
    if p.get('innovation'):
        c = p['innovation']
        if (c.get('tokens') not in (8, 16, 32, 64) or c.get('width') != 32
                or not c.get('steps') or any(type(x) is not int or x not in range(4) for x in c['steps'])
                or not 0 < c.get('rate_weight', 0) <= 1):
            raise ValueError('invalid innovation codec profile')
    return p


def rgb(images):
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3 or len(images) < 2:
        raise ValueError('at least two RGB uint8 frames required')
    return images


def adaptive_indices(images, stride, protected=(), motion_threshold=12., cut_threshold=40.):
    """Protect small moving regions using tile maxima, not only global change.

    A deterministic sender heuristic, not claimed to be a learned ROI detector.
    No task labels or receiver quality metrics enter this policy.
    """
    import cv2
    rgb(images)
    if stride not in (1, 2, 3, 4):
        raise ValueError('invalid temporal stride')
    if any(type(i) is not int or not 0 <= i < len(images) for i in protected):
        raise ValueError('invalid semantic timestamp')
    keys = set(range(0, len(images), stride)) | {len(images)-1} | set(protected)
    small = np.stack([cv2.resize(f, (96, 64), interpolation=cv2.INTER_AREA) for f in images]).astype(np.float32)
    scores = []
    for i in range(1, len(images)):
        change = np.abs(small[i]-small[i-1]).mean(-1)
        tiles = change.reshape(8, 8, 12, 8).mean((1, 3))
        score = float(tiles.max()); scores.append(score)
        if score >= motion_threshold or float(change.mean()) >= cut_threshold:
            keys.update((i-1, i))
    return sorted(keys), scores


def windows(frames, length=17, overlap=5):
    if type(frames) is not int or frames < 2 or length != 17 or not 0 < overlap < length:
        raise ValueError('invalid window timeline')
    starts = list(range(0, max(1, frames-length+1), length-overlap))
    if frames > length and starts[-1]+length < frames:
        starts.append(frames-length)
    return [(s, min(s+length, frames)) for s in starts]


def validate_context(c):
    from .dense_video_transport import validate_context as validate
    validate(c)
    v = c['video']
    if v['frames'] < 2 or v['width'] % 16 or v['height'] % 16:
        raise ValueError('unified BSU needs dimensions divisible by 16 and >=2 frames')


def encode_packet(context, stream, coding, decoder, enhancements=None):
    validate_context(context)
    if not isinstance(stream, bytes) or not stream:
        raise ValueError('empty entropy stream')
    decoder = dict(decoder)
    if decoder.get('profile', {}).get('compact_wire'):
        # All dynamic decoder settings remain paid. No external profile table
        # or unaccounted setup message is needed to interpret this packet.
        fields = ('policy', 'window_frames', 'window_overlap', 'diffusion_steps',
                  'start_noise', 'cfg_scale', 'adapter_width', 'max_rgb_correction', 'seed', 'innovation')
        decoder['profile'] = {k: decoder['profile'][k] for k in fields if k in decoder['profile']}
    side = dict(context=context, coding=coding, decoder=decoder)
    raw = zlib.compress(json.dumps(side, sort_keys=True, separators=(',', ':'), allow_nan=False).encode(), 9)
    h = dict(schema=POLICY, side_bytes=len(raw), stream_sha256=hashlib.sha256(stream).hexdigest())
    if enhancements is not None:
        overlap = decoder.get('profile', {}).get('window_overlap', 5)
        if len(enhancements) != len(windows(context['video']['frames'], overlap=overlap)) or any(not x for x in enhancements):
            raise ValueError('incomplete paid innovation windows')
        blob = b''.join(enhancements)
        h.update(stream_bytes=len(stream), innovation_lengths=[len(x) for x in enhancements],
                 innovation_sha256=hashlib.sha256(blob).hexdigest())
        return pack(h, raw+stream+blob)
    return pack(h, raw+stream)


def decode_packet(data):
    h, payload = unpack(data)
    n = h.get('side_bytes')
    if h.get('schema') != POLICY or type(n) is not int or not 0 < n < len(payload):
        raise ValueError('invalid unified packet')
    decompressor = zlib.decompressobj()
    raw = decompressor.decompress(payload[:n], 4*1024*1024+1)
    if len(raw) > 4*1024*1024 or not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
        raise ValueError('invalid compressed conditions')
    side = json.loads(raw)
    size = h.get('stream_bytes', len(payload)-n)
    if type(size) is not int or not 0 < size <= len(payload)-n:
        raise ValueError('invalid video stream length')
    stream = payload[n:n+size]
    if hashlib.sha256(stream).hexdigest() != h['stream_sha256']:
        raise ValueError('entropy stream integrity failure')
    validate_context(side['context'])
    c = side['coding']; v = side['context']['video']; indices = c['indices']
    if (type(c['qp']) is not int or not 0 <= c['qp'] <= 63 or c['scale'] not in (1, 2)
            or c['stride'] not in (1, 2, 3, 4) or not indices
            or any(type(i) is not int for i in indices) or indices != sorted(set(indices))
            or indices[0] != 0 or indices[-1] != v['frames']-1
            or any(b-a > c['stride'] for a, b in zip(indices, indices[1:]))):
        raise ValueError('invalid coding/timestamp map')
    # The nested codec header is checked independently by DCVC.decode as well.
    inner, _ = unpack(stream)
    if (inner['indices'] != indices or inner['width']*c['scale'] != v['width']
            or inner['height']*c['scale'] != v['height']):
        raise ValueError('coded geometry or timestamps differ from paid context')
    if side['decoder'].get('policy') != POLICY:
        raise ValueError('wrong receiver policy')
    if 'innovation_lengths' in h:
        lengths = h['innovation_lengths']; blob = payload[n+size:]
        overlap = side['decoder'].get('profile', {}).get('window_overlap', 5)
        if (not isinstance(lengths, list) or len(lengths) != len(windows(v['frames'], overlap=overlap))
                or any(type(x) is not int or not 0 < x <= 1_000_000 for x in lengths)
                or sum(lengths) != len(blob) or hashlib.sha256(blob).hexdigest() != h['innovation_sha256']
                or not side['decoder'].get('profile', {}).get('innovation')):
            raise ValueError('invalid paid innovation payload')
        offset = 0; chunks = []
        for length in lengths:
            chunks.append(blob[offset:offset+length]); offset += length
        side['enhancements'] = chunks
    elif n+size != len(payload):
        raise ValueError('unconsumed stream bytes')
    return side, stream


def packet_breakdown(data):
    h, payload = unpack(data); decode_packet(data)
    token_bytes = sum(h.get('innovation_lengths', []))
    return dict(packet_bytes=len(data), header_bytes=len(data)-len(payload),
                side_bytes=h['side_bytes'], video_stream_bytes=len(payload)-h['side_bytes']-token_bytes,
                innovation_bytes=token_bytes, channel_uses=channel_uses(data))


def channel_uses(packet):
    return accounting(len(packet), bits_per_symbol=4)['complex_channel_uses']


def proxy_distortion(source, base):
    """Sender-local screening objective; not a claim about perceptual quality."""
    if rgb(source).shape != rgb(base).shape:
        raise ValueError('proxy timeline differs')
    a, b = source.astype(np.float32), base.astype(np.float32)
    motion = np.abs(np.diff(a, axis=0)).mean(-1)
    weights = 1 + np.minimum(np.pad(motion, ((1, 0), (0, 0), (0, 0))), 48)/12
    mse = float(((a-b)**2*weights[..., None]).mean())
    temporal = float(np.abs(np.diff(a, axis=0)-np.diff(b, axis=0)).mean())
    return dict(score=mse+8*temporal, weighted_mse=mse, temporal_mae=temporal)


def choose_candidate(records, budget):
    if type(budget) is not int or budget < 1:
        raise ValueError('positive total channel budget required')
    fitting = [r for r in records if r['channel_uses'] <= budget and math.isfinite(r['distortion']['score'])]
    if not fitting:
        raise ValueError('no candidate fits the complete paid packet budget')
    return min(fitting, key=lambda r: (r['distortion']['score'], r['channel_uses'], r['name']))


def bounded_refinement(base, generated, qp, max_delta=8.):
    """Received-only, spatially gated detail correction, with no pixel warping.

    This bound protects RGB values, NOT a mathematical guarantee of semantic
    preservation. Strong disagreement and motion suppress generated detail.
    """
    import cv2
    if rgb(base).shape != rgb(generated).shape or not 0 <= qp <= 63 or not 0 <= max_delta <= 16:
        raise ValueError('invalid refinement inputs')
    a, b = base.astype(np.float32), generated.astype(np.float32)
    movement = np.zeros(a.shape[:-1], np.float32)
    movement[1:] = np.abs(a[1:]-a[:-1]).mean(-1)
    movement[:-1] = np.maximum(movement[:-1], movement[1:])
    residuals, gates = [], []
    for i, (x, y) in enumerate(zip(a, b)):
        low_x = cv2.GaussianBlur(x, (5, 5), 1.)
        low_y = cv2.GaussianBlur(y, (5, 5), 1.)
        disagreement = np.abs(low_y-low_x).mean(-1)
        gate = np.exp(-disagreement/12.) / (1+movement[i]/8.)
        strength = .25+.5*(1-qp/63.)
        detail = (y-low_y)-(x-low_x)
        residuals.append(np.clip(detail, -max_delta, max_delta)*gate[..., None]*strength)
        gates.append(gate)
    delta = np.stack(residuals)
    result = np.rint(np.clip(a+delta, 0, 255)).astype(np.uint8)
    return result, dict(max_absolute_rgb_change=int(np.abs(result.astype(int)-base.astype(int)).max()),
        mean_gate=float(np.mean(gates)), source_used=False, temporal_warping=False,
        semantic_preservation_proven=False)
