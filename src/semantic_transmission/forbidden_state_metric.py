"""Forbidden-state occupancy (FSO), event-order inversion (EOI) and
unsupported-presence occupancy (UEP) candidates.

Design rules carried over from the one candidate that passed its controlled gate
(post-exit residual occupancy) and from the three documented failure patterns:

F1 dilution   -> every denominator is a fixed short horizon in frames, never the
                 clip length or an observation count.
F2 invariance  -> a forbidden state is one-sided ("present where absence is
                 required"), the heading state is defined against the
                 reconstruction's own pre-anchor heading, and every occupancy is
                 differenced against the same quantity measured on the source.
F3 extractor   -> a missed source event removes a query from the sample instead
                 of creating a penalty; order pairs that cannot be matched leave
                 the denominator instead of scoring 1.

Only RGB-derived observations, decoded RX status and entity association enter the
scorers. Renderer identity, event truth and intervention labels never do. These
scores describe inferred entities, not certified semantic truth.
"""
import itertools
import math

import numpy as np

from .event_duration_metric import PARAMETERS as DURATION_PARAMETERS, associate_entities, location, prototype

PARAMETERS = {'fps': 8., 'horizon_frames': 4, 'anchor_frames': 4,
              'heading_context_frames': 2, 'heading_tolerance_degrees': 45.,
              'minimum_speed_px_per_frame': .25, 'residual_speed_ratio': .5,
              'order_separation_s': .375,
              'unsupported_horizon_frames': 4, 'minimum_source_track_frames': 3,
              'sibling_cosine': .80, 'sibling_distance': .35}

# instance -> (anchoring source event type, window side, forbidden state)
INSTANCES = {'presence': ('exit', 'after', 'presence'),
             'premature': ('enter', 'before', 'presence'),
             'motion': ('stop', 'after', 'motion'),
             'heading': ('turn', 'after', 'heading')}
NAMES = tuple('fso_' + key for key in INSTANCES) + ('fso_max',)
RAW_NAMES = tuple(name + '_raw' for name in NAMES)


def siblings(tracks, index):
    """Other tracks of the same video that plausibly carry the same entity.

Frame-disjoint, prototype-similar and co-located fragments only. This is what
makes a broken source track stop looking like a real disappearance.
"""
    chosen = [index]
    if not tracks[index]:
        return chosen
    reference, place = prototype(tracks[index]), location(tracks[index])
    for other, track in enumerate(tracks):
        if other == index or not track or set(track) & set(tracks[index]):
            continue
        if (float(reference @ prototype(track)) >= PARAMETERS['sibling_cosine']
                and float(np.linalg.norm(place - location(track))) <= PARAMETERS['sibling_distance']):
            chosen.append(other)
    return chosen


def present(tracks, ids, frame):
    return any(frame in tracks[i] for i in ids)


def velocity(tracks, ids, frame):
    values = [tracks[i][frame]['velocity'] for i in ids
              if frame in tracks[i] and tracks[i][frame].get('velocity') is not None]
    return np.mean(values, axis=0) if values else None


def pre_heading(tracks, ids, anchor):
    """Heading of the entity in its own video just before the anchor frame."""
    window = range(max(0, anchor - PARAMETERS['heading_context_frames']), anchor)
    values = [velocity(tracks, ids, frame) for frame in window]
    values = [v for v in values if v is not None]
    if not values:
        return None
    heading = np.mean(values, axis=0)
    return heading if float(np.linalg.norm(heading)) >= PARAMETERS['minimum_speed_px_per_frame'] else None


def occupancy(tracks, ids, window, state, heading):
    """Per-frame forbidden-state indicator; no cross-video appearance comparison.

The motion and heading states are self-referential twice over: the bar for
"still moving" is a fraction of the entity's own pre-anchor speed in the same
video, and the stale direction is measured against that same video's pre-anchor
heading. Extractor flow noise is small relative to the entity's own motion, so
it cannot manufacture a forbidden state.
"""
    bits = []
    floor = PARAMETERS['minimum_speed_px_per_frame']
    if state in ('motion', 'heading'):
        if heading is None:
            return [0] * len(window)
        floor = max(floor, PARAMETERS['residual_speed_ratio'] * float(np.linalg.norm(heading)))
    for frame in window:
        if not present(tracks, ids, frame):
            bits.append(0)
            continue
        if state == 'presence':
            bits.append(1)
            continue
        vector = velocity(tracks, ids, frame)
        speed = float(np.linalg.norm(vector)) if vector is not None else 0.
        if speed < floor:
            bits.append(0)
            continue
        if state == 'motion':
            bits.append(1)
            continue
        cosine = float(np.dot(vector, heading) / (speed * float(np.linalg.norm(heading))))
        angle = math.degrees(math.acos(max(-1., min(1., cosine))))
        bits.append(int(angle <= PARAMETERS['heading_tolerance_degrees']))
    return bits


def event_record(instance, event, a, b, groups):
    kind, side, state = INSTANCES[instance]
    horizon, fps = PARAMETERS['horizon_frames'], PARAMETERS['fps']
    track_id = event['track_id']
    anchor = int(round(event['time_s'] * fps))
    window = (list(range(anchor, anchor + horizon)) if side == 'after'
              else list(range(anchor - horizon, anchor)))
    source_ids, target_ids = siblings(a['tracks'], track_id), groups.get(track_id, [])
    record = {'instance': instance, 'type': kind, 'source_track_id': track_id,
              'source_time_s': event['time_s'], 'window_frames': window,
              'source_entity_tracks': source_ids, 'reconstruction_entity_tracks': target_ids}
    complete = all(0 <= frame < a['frame_count'] for frame in window)
    readable = (len(a['tracks'][track_id]) >= PARAMETERS['minimum_source_track_frames']
                and bool(b['tracks']) and bool(target_ids))
    checked = range(anchor, anchor + PARAMETERS['anchor_frames']) if side == 'before' else \
        range(max(0, anchor - PARAMETERS['anchor_frames']), anchor)
    anchored = any(present(b['tracks'], target_ids, frame) for frame in checked)
    kinematic = state in ('motion', 'heading')
    heading_a = pre_heading(a['tracks'], source_ids, anchor) if kinematic else None
    heading_b = pre_heading(b['tracks'], target_ids, anchor) if kinematic else None
    if kinematic:
        if complete and readable:
            # A vanished entity is the presence instance's subject, not this one's.
            anchored = anchored and all(present(b['tracks'], target_ids, frame) for frame in window)
        # Without an observed pre-anchor motion in both videos there is no
        # self-referential bar, so the query is unobservable rather than zero.
        anchored = anchored and heading_a is not None and heading_b is not None
    status = ('unobservable' if not readable or not anchored else
              'right_censored' if not complete else 'observed')
    record['status'] = status
    if status != 'observed':
        record.update(occupancy_reconstruction=None, occupancy_source=None, score=None, score_raw=None)
        return record
    bits_b = occupancy(b['tracks'], target_ids, window, state, heading_b)
    bits_a = occupancy(a['tracks'], source_ids, window, state, heading_a)
    raw, base = sum(bits_b) / horizon, sum(bits_a) / horizon
    record.update(occupancy_reconstruction=bits_b, occupancy_source=bits_a,
                  score=max(0., raw - base), score_raw=raw,
                  observed_s=sum(bits_b) / fps,
                  returns=sum(bits_b[k] and not bits_b[k - 1] for k in range(1, len(bits_b))))
    return record


def order_matches(a, b, groups):
    """Identity- and type-conditioned matching over the whole clip.

No causal horizon: an event moved backwards in time must still be matchable, or
its pair leaves the denominator rather than counting as an inversion.
"""
    from scipy.optimize import linear_sum_assignment
    ea, eb = a['events'], b['events']
    cost = np.full((len(ea), len(eb)), 1e6)
    for i, x in enumerate(ea):
        for j, y in enumerate(eb):
            offset = abs(x['direction'] - y['direction'])
            if (x['type'] == y['type'] and y['track_id'] in groups.get(x['track_id'], [])
                    and min(offset, 8 - offset) <= 1):
                cost[i, j] = abs(x['time_s'] - y['time_s'])
    return {int(i): int(j) for i, j in zip(*linear_sum_assignment(cost)) if cost[i, j] < 1e6}


def order_inversions(a, b, matches):
    separation = PARAMETERS['order_separation_s']
    ea, eb = a['events'], b['events']
    rows = []
    for i, j in itertools.combinations(sorted(matches), 2):
        first, second = ea[i], ea[j]
        if second['time_s'] - first['time_s'] < separation - 1e-9:
            continue
        x, y = eb[matches[i]], eb[matches[j]]
        rows.append({'source_events': [i, j], 'reconstruction_events': [matches[i], matches[j]],
                     'source_gap_s': second['time_s'] - first['time_s'],
                     'reconstruction_gap_s': y['time_s'] - x['time_s'],
                     'inverted': bool(x['time_s'] - y['time_s'] >= separation - 1e-9)})
    inverted = sum(r['inverted'] for r in rows)
    return {'eoi': inverted / len(rows) if rows else None, 'eoi_ordered_pairs': len(rows),
            'eoi_inversions': inverted, 'eoi_rows': rows,
            'eoi_scope': 'matched ordered pairs only; unmatched source events reduce the denominator'}


def evaluate(a, b):
    """FSO and EOI from two RGB observations on one shared physical timeline."""
    if a['frame_count'] != b['frame_count']:
        raise ValueError('shared physical timeline required')
    groups = associate_entities(a['tracks'], b['tracks'])
    records = []
    for instance, (kind, _, _) in INSTANCES.items():
        for event in a['events']:
            if event['type'] == kind:
                records.append(event_record(instance, event, a, b, groups))
    result = {'fso_records': records, 'entity_groups': groups,
              'scope': 'RGB-inferred entities and events; no truth, oracle identity or intervention label'}
    for instance in INSTANCES:
        chosen = [r for r in records if r['instance'] == instance]
        for suffix, key in (('', 'score'), ('_raw', 'score_raw')):
            values = [r[key] for r in chosen if r[key] is not None]
            result['fso_' + instance + suffix] = max(values) if values else None
        result['fso_' + instance + '_queries'] = len(chosen)
        result['fso_' + instance + '_measured'] = sum(r['status'] == 'observed' for r in chosen)
    for suffix, key in (('', 'score'), ('_raw', 'score_raw')):
        values = [r[key] for r in records if r[key] is not None]
        result['fso_max' + suffix] = max(values) if values else None
    matches = order_matches(a, b, groups)
    result.update(order_inversions(a, b, matches))
    result['eoi_matches'] = matches
    return result


def unsupported_presence(a, b, rx, rx_status):
    """UEP: reconstruction presence with neither source correspondence nor RX evidence.

RX evidence is read from the decoded receiver video and its per-slot decode
status only. Undecodable slots carry no evidence, exactly as in the frozen
attribution rules.
"""
    from .object_metric import score_tracks
    if not b['tracks'] or not rx_status:
        return {'uep': None, 'uep_status': 'no_reconstruction_tracks' if not b['tracks'] else 'no_status',
                'uep_rows': []}
    horizon = PARAMETERS['unsupported_horizon_frames']
    explained = {j for _, j in score_tracks(a['tracks'], b['tracks']).get('track_pairs', [])}
    evidence = {}
    for i, j in score_tracks(b['tracks'], rx['tracks']).get('track_pairs', []):
        evidence.setdefault(i, set()).update(t for t in rx['tracks'][j]
                                             if t < len(rx_status) and rx_status[t] == 'ok')
    rows = []
    for j, track in enumerate(b['tracks']):
        if j in explained:
            continue
        unsupported = sorted(set(track) - evidence.get(j, set()))
        rows.append({'reconstruction_track': j, 'frames': len(track),
                     'unsupported_frames': len(unsupported),
                     'first_unsupported_frame': unsupported[0] if unsupported else None,
                     'score': min(1., len(unsupported) / horizon)})
    return {'uep': max([r['score'] for r in rows], default=0.), 'uep_status': 'observed',
            'uep_unexplained_tracks': len(rows), 'uep_rows': rows,
            'uep_scope': 'inferred tracks and decoded RX slots; not an open-vocabulary hallucination judgement'}


def truth_occupancy(instance, centers, visible, object_id, frame):
    """Independent numerical oracle from simulator kinematics; never used in scoring."""
    kind, side, state = INSTANCES[instance]
    horizon, fps = PARAMETERS['horizon_frames'], PARAMETERS['fps']
    k = object_id - 1
    window = (list(range(frame, frame + horizon)) if side == 'after'
              else list(range(frame - horizon, frame)))
    if not all(0 <= t < len(visible) for t in window):
        return {'occupancy': None, 'status': 'right_censored'}
    steps = np.diff(centers[:, k], axis=0)
    def truth_velocity(t):
        return steps[t] if 0 <= t < len(steps) and visible[t, k] and visible[t + 1, k] else None
    reference, floor = None, .5
    if state in ('motion', 'heading'):
        values = [truth_velocity(t) for t in range(max(0, frame - PARAMETERS['heading_context_frames']), frame)]
        values = [v for v in values if v is not None]
        reference = np.mean(values, axis=0) if values else None
        if reference is None or float(np.linalg.norm(reference)) < .5:
            return {'occupancy': None, 'status': 'no_pre_anchor_motion'}
        floor = max(floor, PARAMETERS['residual_speed_ratio'] * float(np.linalg.norm(reference)))
    bits = []
    for t in window:
        if not visible[t, k]:
            bits.append(0)
            continue
        if state == 'presence':
            bits.append(1)
            continue
        vector = truth_velocity(t)
        speed = float(np.linalg.norm(vector)) if vector is not None else 0.
        if speed < floor:
            bits.append(0)
            continue
        if state == 'motion':
            bits.append(1)
            continue
        cosine = float(np.dot(vector, reference) / (speed * float(np.linalg.norm(reference))))
        bits.append(int(math.degrees(math.acos(max(-1., min(1., cosine)))) <= PARAMETERS['heading_tolerance_degrees']))
    return {'occupancy': sum(bits) / horizon, 'status': 'observed', 'presence': bits,
            'observed_s': sum(bits) / fps}


DURATION_PARAMETER_REFERENCE = dict(DURATION_PARAMETERS)
