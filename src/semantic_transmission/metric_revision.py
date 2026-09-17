"""Development v6: fragment-aware RGB association and a shared event contract.

No annotations, intervention names, model settings or quality scores enter
``observe`` or ``evaluate``. Annotation trajectories use ``trajectory_events``
only in the external audit. All old metric implementations remain unchanged.
"""
import itertools
import math

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

PARAMETERS = dict(fps=8., horizon=4, smooth=5, speed=.5, context=2,
                  persistence=2, turn_degrees=90., separation=3,
                  heading_degrees=45., speed_ratio=.5, order_coverage=.8,
                  stitch_gap=2, stitch_cosine=.8, stitch_distance=.08,
                  geometric_iou=.5, area_ratio=.5, match_cosine=.65,
                  match_distance=.15, feature_iou=.05)
INSTANCES = {'presence': ('exit', 'after'), 'premature': ('enter', 'before'),
             'motion': ('stop', 'after'), 'heading': ('turn', 'after')}
PRIMARY = tuple('fso_' + k for k in INSTANCES) + ('eoi', 'uep')


def _unit(x):
    x = np.asarray(x, dtype=float)
    return x / max(float(np.linalg.norm(x)), 1e-12)


def stitch(tracks, shape):
    """Join short, disjoint fragments; never synthesize an unobserved mask."""
    result, members = [], []
    scale = math.hypot(*shape)
    for index in sorted(range(len(tracks)), key=lambda i: min(tracks[i])):
        tr = tracks[index]
        start = min(tr)
        options = []
        for i, prior in enumerate(result):
            end = max(prior)
            gap = start - end - 1
            if not 0 <= gap <= PARAMETERS['stitch_gap']:
                continue
            prev, now = prior[end], tr[start]
            sim = float(_unit(prev['feature']) @ _unit(now['feature']))
            times = sorted(prior)
            velocity = ((np.asarray(prev['center']) - prior[times[-2]]['center']) /
                        (end - times[-2]) if len(times) > 1 else np.zeros(2))
            distance = float(np.linalg.norm(np.asarray(now['center']) - prev['center'] - velocity * (gap + 1))) / scale
            if sim >= PARAMETERS['stitch_cosine'] and distance <= PARAMETERS['stitch_distance']:
                options.append((distance + .05 * (1 - sim), i))
        # Reject near-tied re-identifications rather than merging uncertain IDs.
        options.sort()
        if options and (len(options) == 1 or options[1][0] - options[0][0] >= .01):
            i = options[0][1]
            result[i].update(tr)
            members[i].append(index)
        else:
            result.append(dict(tr))
            members.append([index])
    return result, members


def kinematics(track):
    """Image-plane centers; contiguous segments only, no gap interpolation."""
    times = sorted(track)
    segments = []
    for t in times:
        if not segments or t != segments[-1][-1] + 1:
            segments.append([])
        segments[-1].append(t)
    points, velocity = {}, {}
    half = PARAMETERS['smooth'] // 2
    for segment in segments:
        raw = np.asarray([track[t]['center'] for t in segment], dtype=float)
        smooth = np.asarray([raw[max(0, i-half):min(len(raw), i+half+1)].mean(0) for i in range(len(raw))])
        points.update(zip(segment, smooth))
        velocity.update((t, smooth[i+1] - smooth[i]) for i, t in enumerate(segment[:-1]))
    return points, velocity


def trajectory_events(tracks, frame_count):
    events = []
    p = PARAMETERS
    for identity, tr in enumerate(tracks):
        if not tr:
            continue
        _, v = kinematics(tr)
        candidates = []
        start, end = min(tr), max(tr)
        if start > 0:
            candidates.append(('enter', start, 1000))
        if end < frame_count - 1:
            candidates.append(('exit', end + 1, 1000))
        for t in range(start + 1, end):
            before = [v.get(i) for i in range(t-p['context'], t)]
            after = [v.get(i) for i in range(t, t+p['context'])]
            if any(x is None for x in before + after):
                continue
            sa, sb = np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)
            if np.all(sa < p['speed']) and np.all(sb >= p['speed']):
                candidates.append(('start', t, 100))
            if np.all(sa >= p['speed']) and np.all(sb < p['speed']):
                candidates.append(('stop', t, 100))
            a, b = np.mean(before, axis=0), np.mean(after, axis=0)
            if min(np.linalg.norm(a), np.linalg.norm(b)) >= p['speed']:
                angle = math.degrees(math.acos(float(np.clip(_unit(a) @ _unit(b), -1, 1))))
                if angle > p['turn_degrees']:
                    candidates.append(('turn', t, 200 + angle))
        chosen = []
        for kind, t, priority in sorted(candidates, key=lambda x: (-x[2], x[1])):
            if all(abs(t - z[1]) >= p['separation'] for z in chosen):
                chosen.append((kind, t))
        events += [dict(type=k, frame=t, track_id=identity, time_s=t/p['fps']) for k, t in chosen]
    return sorted(events, key=lambda e: (e['frame'], e['track_id'], e['type']))


def observe(cached):
    tracks, members = stitch(cached['tracks'], cached['shape'])
    prepared = []
    for track in tracks:
        _, velocities = kinematics(track)
        prepared.append({t: {**d, 'velocity': velocities.get(t),
                              'small_mask': cv2.resize(d['mask'].astype(np.uint8), (80, 48), interpolation=cv2.INTER_NEAREST).reshape(-1).astype(float),
                              'normalized_center': np.asarray(d['center']) / np.asarray(cached['shape'][::-1])}
                         for t, d in track.items()})
    return dict(tracks=prepared, events=trajectory_events(tracks, cached['frame_count']),
                frame_count=cached['frame_count'], shape=cached['shape'], original_tracks=len(cached['tracks']),
                fragment_members=members)


def association(a, b):
    """One-to-one per-frame matching, with explicit track-fragment support."""
    evidence = {}
    for t in range(a['frame_count']):
        aa = [(i, tr[t]) for i, tr in enumerate(a['tracks']) if t in tr]
        bb = [(j, tr[t]) for j, tr in enumerate(b['tracks']) if t in tr]
        if not aa or not bb:
            continue
        ma = np.stack([d['small_mask'] for _, d in aa])
        mb = np.stack([d['small_mask'] for _, d in bb])
        intersection = ma @ mb.T
        size_a, size_b = ma.sum(1)[:, None], mb.sum(1)[None, :]
        overlap = intersection / np.maximum(1, size_a + size_b - intersection)
        ratio = np.minimum(size_a, size_b) / np.maximum(1, np.maximum(size_a, size_b))
        similarity = np.stack([d['feature'] for _, d in aa]) @ np.stack([d['feature'] for _, d in bb]).T
        distance = np.linalg.norm(np.stack([d['normalized_center'] for _, d in aa])[:, None, :] -
                                  np.stack([d['normalized_center'] for _, d in bb])[None, :, :], axis=-1)
        p = PARAMETERS
        geometric = (overlap >= p['geometric_iou']) & (ratio >= p['area_ratio'])
        feature = (similarity >= p['match_cosine']) & (distance <= p['match_distance']) & (overlap >= p['feature_iou'])
        cost = np.where(geometric | feature, .7 * (1-overlap) + .3 * (1-np.clip(similarity, -1, 1)), 1e6)
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] < 1e6:
                evidence.setdefault((aa[i][0], bb[j][0]), []).append(t)
    # A single overlapping frame does not explain an entire long track.
    reliable = {(i, j): ts for (i, j), ts in evidence.items()
                if len(ts) >= min(2, len(a['tracks'][i]), len(b['tracks'][j])) and
                len(ts) >= .5 * len(set(a['tracks'][i]) & set(b['tracks'][j]))}
    groups = {}
    for i, j in reliable:
        groups.setdefault(i, []).append(j)
    return groups, reliable


def _vector(tracks, ids, t):
    values = [tracks[i][t].get('velocity') for i in ids if t in tracks[i]]
    values = [v for v in values if v is not None]
    return np.mean(values, axis=0) if values else None


def occupancy(tracks, ids, anchor, state, side, frame_count):
    p = PARAMETERS
    window = list(range(anchor, anchor+p['horizon']) if side == 'after' else range(anchor-p['horizon'], anchor))
    if not ids or min(window) < 0 or max(window) >= frame_count:
        return None
    if state in ('presence', 'premature'):
        return sum(any(t in tracks[i] for i in ids) for t in window) / p['horizon']
    before = [_vector(tracks, ids, t) for t in range(anchor-p['context'], anchor)]
    if any(v is None for v in before):
        return None
    heading = np.mean(before, axis=0)
    if np.linalg.norm(heading) < p['speed']:
        return None
    floor = max(p['speed'], p['speed_ratio'] * np.linalg.norm(heading))
    bits = []
    for t in window:
        v = _vector(tracks, ids, t)
        if v is None:
            return None
        active = np.linalg.norm(v) >= floor
        aligned = float(_unit(v) @ _unit(heading)) >= math.cos(math.radians(p['heading_degrees']))
        bits.append(active and (state == 'motion' or aligned))
    return sum(bits) / p['horizon']


def fso_records(a, b, groups):
    records = []
    for event in a['events']:
        for state, (kind, side) in INSTANCES.items():
            if event['type'] != kind:
                continue
            i, t = event['track_id'], event['frame']
            ids = groups.get(i, [])
            aa = occupancy(a['tracks'], [i], t, state, side, a['frame_count'])
            bb = occupancy(b['tracks'], ids, t, state, side, b['frame_count'])
            anchor_window = range(t, t+4) if side == 'before' else range(max(0, t-4), t)
            anchored = any(k in b['tracks'][j] for j in ids for k in anchor_window)
            score = max(0., bb-aa) if anchored and aa is not None and bb is not None else None
            records.append(dict(metric='fso_'+state, source_track=i, anchor=t, score=score,
                                source_occupancy=aa, reconstruction_occupancy=bb))
    return records


def order_score(a, b, groups):
    ea, eb = a['events'], b['events']
    cost = np.full((len(ea), len(eb)), 1e6)
    for i, x in enumerate(ea):
        for j, y in enumerate(eb):
            if x['type'] == y['type'] and y['track_id'] in groups.get(x['track_id'], []):
                cost[i, j] = abs(x['frame'] - y['frame'])
    matches = {int(i): int(j) for i, j in zip(*linear_sum_assignment(cost)) if cost[i, j] < 1e6}
    pairs = [(i, j) for i, j in itertools.combinations(range(len(ea)), 2)
             if ea[j]['frame'] - ea[i]['frame'] >= PARAMETERS['separation']]
    retained = [(i, j) for i, j in pairs if i in matches and j in matches]
    raw = (sum(eb[matches[i]]['frame'] - eb[matches[j]]['frame'] >= PARAMETERS['separation']
               for i, j in retained) / len(retained)) if retained else None
    coverage = len(retained) / len(pairs) if pairs else None
    return dict(eoi=raw if coverage is not None and coverage >= PARAMETERS['order_coverage'] else None,
                eoi_raw=raw, eoi_pair_coverage=coverage, eoi_source_pairs=len(pairs), eoi_matched_pairs=len(retained))


def evaluate(a, b, rx=None, status=None):
    if a['frame_count'] != b['frame_count'] or a['shape'] != b['shape']:
        raise ValueError('shared timeline and dimensions required')
    groups, pairs = association(a, b)
    records = fso_records(a, b, groups)
    result = {name: max((r['score'] for r in records if r['metric'] == name and r['score'] is not None), default=None)
              for name in PRIMARY if name.startswith('fso_')}
    result.update(order_score(a, b, groups))
    result.update(uep=None, uep_rows=[])
    if rx is not None and status is not None and b['tracks']:
        if rx['frame_count'] != b['frame_count'] or len(status) != b['frame_count']:
            raise ValueError('RX timeline/status mismatch')
        _, rx_pairs = association(b, rx)
        explained = {j for _, j in pairs}
        support = {}
        for (j, _), times in rx_pairs.items():
            support.setdefault(j, set()).update(t for t in times if status[t] == 'ok')
        for j, tr in enumerate(b['tracks']):
            if j in explained:
                continue
            missing = sorted(set(tr) - support.get(j, set()))
            result['uep_rows'].append(dict(track=j, unsupported_frames=missing,
                                           score=min(1., len(missing)/PARAMETERS['horizon'])))
        result['uep'] = max((r['score'] for r in result['uep_rows']), default=0.)
    result.update(fso_records=records, source_tracks=len(a['tracks']), reconstruction_tracks=len(b['tracks']),
                  source_events=len(a['events']), reconstruction_events=len(b['events']),
                  associated_track_pairs=len(pairs), scope='RGB inferred; image-plane kinematics; development v6')
    return result


def annotation_observation(centers, visible):
    """Independent data adapter; called only by the audit, never by evaluate."""
    tracks = []
    for k in range(centers.shape[1]):
        tr = {t: {'center': centers[t, k]} for t in range(len(centers)) if visible[t, k]}
        _, velocity = kinematics(tr)
        tracks.append({t: {**d, 'velocity': velocity.get(t)} for t, d in tr.items()})
    return dict(tracks=tracks, frame_count=len(centers), events=trajectory_events(tracks, len(centers)))


def annotation_truth(a, b):
    groups = {i: [i] for i in range(len(a['tracks'])) if b['tracks'][i]}
    records = fso_records(a, b, groups)
    result = {name: max((r['score'] for r in records if r['metric'] == name and r['score'] is not None), default=None)
              for name in PRIMARY if name.startswith('fso_')}
    # Repeated same-type events cannot acquire an independent identity merely
    # by repeating the scorer's nearest-time matching rule.
    ambiguous = any(len({(e['track_id'], e['type']) for e in o['events']}) != len(o['events']) for o in (a, b))
    result['eoi'] = None if ambiguous else order_score(a, b, groups)['eoi']
    return result, records
