"""Sender point tracks with adaptive time knots, serialized inside paid metadata.

The decoder accepts bytes only. Track visibility is conservative: an endpoint
track remains invalid after its first failed forward/backward check.
"""
import base64
import struct
import zlib

import numpy as np

FIELD = 'bsu_sparse_trajectory_v1'
SIZE = (288, 160)
HEADER = struct.Struct('!4sHBB')


def gray(frame):
    import cv2
    return cv2.cvtColor(cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)


def track(frames, reverse=False):
    import cv2
    images = [gray(f) for f in (frames[::-1] if reverse else frames)]
    points = cv2.goodFeaturesToTrack(images[0], maxCorners=160, qualityLevel=.015, minDistance=9)
    if points is None:
        return np.zeros((len(frames), 0, 2), np.float32), np.zeros((len(frames), 0), bool)
    alive = np.ones(len(points), bool)
    positions, validity = [points[:, 0].copy()], [alive.copy()]
    for left, right in zip(images, images[1:]):
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(left, right, points, None, winSize=(21, 21), maxLevel=3)
        back, ok, _ = cv2.calcOpticalFlowPyrLK(right, left, nxt, None, winSize=(21, 21), maxLevel=3)
        xy = nxt[:, 0]
        alive &= ((status[:, 0] != 0) & (ok[:, 0] != 0)
                  & (np.linalg.norm(back[:, 0] - points[:, 0], axis=1) < 1.)
                  & np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 2) & (xy[:, 0] < SIZE[0]-2)
                  & (xy[:, 1] >= 2) & (xy[:, 1] < SIZE[1]-2))
        xy = np.nan_to_num(xy).clip([0, 0], [SIZE[0]-1, SIZE[1]-1])
        positions.append(xy.copy()); validity.append(alive.copy())
        points = xy[:, None].astype(np.float32)
    positions, validity = np.asarray(positions), np.asarray(validity)
    if reverse:
        positions, validity = positions[::-1].copy(), validity[::-1].copy()
    return positions, validity


def choose_tracks(xy, valid, count, reverse):
    if not xy.shape[1]:
        return []
    origin = xy[-1 if reverse else 0]
    duration = valid.mean(axis=0)
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=2)
    speed *= valid[1:] & valid[:-1]
    # Favor persistent motion, then spatial coverage; no reconstruction metrics.
    score = duration * (1 + np.minimum(speed.sum(axis=0) / 20, 2))
    chosen = []
    for _ in range(min(count, len(origin))):
        candidate = score.copy()
        if chosen:
            distance = np.min(np.linalg.norm(origin[:, None] - origin[chosen][None], axis=2), axis=1)
            candidate *= np.minimum(distance / 50, 1)
            candidate[chosen] = -1
        i = int(np.argmax(candidate))
        if candidate[i] <= 0 or duration[i] < .25:
            break
        chosen.append(i)
    return chosen


def adaptive_knots(xy, valid, count):
    frames = len(xy)
    selected = [0, frames-1]
    while len(selected) < min(count, frames):
        best = None
        for a, b in zip(selected, selected[1:]):
            if b-a < 2:
                continue
            candidates = np.arange(a+1, b)
            weight = ((candidates-a)/(b-a))[:, None, None]
            prediction = xy[a][None]*(1-weight) + xy[b][None]*weight
            usable = valid[a][None] & valid[b][None] & valid[candidates]
            error = np.where(usable, np.linalg.norm(xy[candidates]-prediction, axis=2), 0)
            score = error.max(axis=1) if xy.shape[1] else np.zeros(len(candidates))
            # A small coverage term also samples nearly constant trajectories.
            score += .02 * np.minimum(candidates-a, b-candidates)
            k = int(np.argmax(score))
            row = (float(score[k]), int(candidates[k]))
            if best is None or row > best:
                best = row
        if best is None:
            break
        selected.append(best[1]); selected.sort()
    return np.asarray(selected, dtype=np.uint16)


def encode(xy, valid, origins, times, frames):
    xy, valid, origins, times = np.asarray(xy), np.asarray(valid), np.asarray(origins), np.asarray(times)
    k, n = valid.shape
    if (not 2 <= frames <= 65535 or not 2 <= k <= 32 or not 0 <= n <= 32
            or xy.shape != (k, n, 2) or origins.shape != (n,) or times.shape != (k,)
            or times[0] != 0 or times[-1] != frames-1 or np.any(np.diff(times.astype(int)) <= 0)
            or not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy > np.asarray(SIZE)-1)
            or np.any((origins != 0) & (origins != 1))):
        raise ValueError('invalid sparse trajectory')
    quantized = np.rint(xy * 4).astype('>u2')
    raw = (HEADER.pack(b'SPT1', frames, k, n) + times.astype('>u2').tobytes()
           + origins.astype(np.uint8).tobytes() + np.packbits(valid).tobytes() + quantized.tobytes())
    return base64.b64encode(zlib.compress(raw, 9)).decode('ascii')


def decode(value, frames=None):
    # Size bounds before interpreting untrusted receiver data.
    compressed = base64.b64decode(value, validate=True)
    decoder = zlib.decompressobj()
    raw = decoder.decompress(compressed, 10000)
    if not decoder.eof or decoder.unused_data or len(raw) < HEADER.size:
        raise ValueError('invalid compressed trajectory')
    magic, count, k, n = HEADER.unpack(raw[:HEADER.size])
    if magic != b'SPT1' or not 2 <= count <= 65535 or not 2 <= k <= 32 or n > 32 or frames not in (None, count):
        raise ValueError('invalid trajectory header')
    mask_bytes = (k*n+7)//8
    if len(raw) != HEADER.size + 2*k + n + mask_bytes + 4*k*n:
        raise ValueError('invalid trajectory length')
    offset = HEADER.size
    times = np.frombuffer(raw, '>u2', k, offset).astype(int); offset += 2*k
    origins = np.frombuffer(raw, 'u1', n, offset).copy(); offset += n
    valid = np.unpackbits(np.frombuffer(raw, 'u1', mask_bytes, offset))[:k*n].reshape(k, n).astype(bool)
    offset += mask_bytes
    xy = np.frombuffer(raw, '>u2', 2*k*n, offset).reshape(k, n, 2).astype(np.float32)/4
    if (times[0] != 0 or times[-1] != count-1 or np.any(np.diff(times) <= 0)
            or np.any(origins > 1) or np.any(xy > np.asarray(SIZE)-1)):
        raise ValueError('invalid trajectory coordinates or timestamps')
    dense = np.empty((count, n, 2), np.float32)
    visibility = np.empty((count, n), bool)
    for t in range(count):
        right = min(np.searchsorted(times, t, side='right'), k-1)
        left = max(0, right-1)
        u = (t-times[left])/max(1, times[right]-times[left])
        dense[t] = xy[left]*(1-u)+xy[right]*u
        exact = np.flatnonzero(times == t)
        visibility[t] = valid[exact[0]] if len(exact) else valid[left] & valid[right]
    return dict(frames=count, times=times, origins=origins, xy=dense, valid=visibility,
                knot_xy=xy, knot_valid=valid)


def fit_packet(packet, frames, capacity):
    from .wire import pack, unpack
    header, payload = unpack(packet)
    if FIELD in header or header['video']['frames'] != len(frames):
        raise ValueError('packet timeline mismatch')
    tracks = [track(frames, reverse=r) for r in (False, True)]
    trials = []
    for points, knots in ((16,9),(12,9),(12,7),(8,7),(8,5),(4,5),(4,3)):
        pairs = []
        for reverse, (xy, valid) in enumerate(tracks):
            selected = choose_tracks(xy, valid, points//2, bool(reverse))
            pairs.append((xy[:, selected], valid[:, selected], np.full(len(selected), reverse)))
        xy = np.concatenate([x[0] for x in pairs], axis=1)
        valid = np.concatenate([x[1] for x in pairs], axis=1)
        origins = np.concatenate([x[2] for x in pairs])
        times = adaptive_knots(xy, valid, knots)
        value = encode(xy[times], valid[times], origins, times, len(frames))
        candidate = pack(dict(header, **{FIELD: value}), payload)
        trials.append(dict(points=xy.shape[1], knots=len(times), packet_bytes=len(candidate)))
        if len(candidate) <= capacity:
            return candidate, dict(selected=trials[-1], trials=trials, timestamps=times.tolist(),
                                   mean_visibility=float(valid.mean()) if valid.size else 0,
                                   bypass=False, selection='first fitting predeclared representation; no generated quality used')
    return packet, dict(trials=trials, bypass=True, reason='no trajectory fits existing LDPC padding')
