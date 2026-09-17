"""Observed event duration and response delay; no renderer truth enters scoring.

These candidates describe inferred entities, not certified semantic truth.
All matching and horizons are fixed before heldout scoring.
"""
import numpy as np

from .object_metric import score_tracks

PARAMETERS = {'fps': 8., 'ghost_horizon_frames': 4, 'delay_horizon_frames': 8,
              'early_tolerance_frames': 1, 'fragment_cosine': .80,
              'fragment_margin': .03, 'fragment_distance': .35,
              'minimum_source_track_frames': 3}


def prototype(track):
    f = np.mean([r['feature'] for r in track.values()], axis=0)
    return f / max(float(np.linalg.norm(f)), 1e-12)


def location(track):
    return np.mean([np.asarray(r['center']) / np.asarray(r['mask'].shape[::-1])
                    for r in track.values()], axis=0)


def associate_entities(a, b):
    """Frozen global matching plus unambiguous nonoverlapping fragments.

Supplementary fragment assignment uses only RGB features and normalized position.
It never sees true identity, event type or fault metadata.
"""
    pairs = score_tracks(a, b).get('track_pairs', [])
    groups = {i: [j] for i, j in pairs}
    used = {j for _, j in pairs}
    if not a:
        return groups
    ap = np.stack([prototype(t) for t in a])
    ac = np.stack([location(t) for t in a])
    for j, track in enumerate(b):
        if j in used:
            continue
        similarities = ap @ prototype(track)
        order = np.argsort(-similarities)
        i = int(order[0]); margin = float(similarities[i] - similarities[order[1]]) if len(order) > 1 else 1.
        overlap = any(set(track) & set(b[k]) for k in groups.get(i, []))
        if (similarities[i] >= PARAMETERS['fragment_cosine'] and margin >= PARAMETERS['fragment_margin']
                and np.linalg.norm(ac[i] - location(track)) <= PARAMETERS['fragment_distance'] and not overlap):
            groups.setdefault(i, []).append(j)
    return groups


def matched_events(a, b, groups, frame_count):
    """One-to-one, identity-conditioned causal matching inside a fixed horizon."""
    from scipy.optimize import linear_sum_assignment
    ea, eb = a['events'], b['events']; fps = PARAMETERS['fps']
    cost = np.full((len(ea), len(eb)), 1e6)
    for i, x in enumerate(ea):
        for j, y in enumerate(eb):
            delta = (y['time_s'] - x['time_s']) * fps
            direction = abs(x['direction'] - y['direction'])
            if (x['type'] == y['type'] and y['track_id'] in groups.get(x['track_id'], [])
                    and -PARAMETERS['early_tolerance_frames'] <= delta <= PARAMETERS['delay_horizon_frames']
                    and min(direction, 8 - direction) <= 1):
                cost[i, j] = abs(delta) + .01 * max(0., -delta)
    return {int(i): int(j) for i, j in zip(*linear_sum_assignment(cost)) if cost[i, j] < 1e6}


def evaluate_observations(a, b):
    count = a['frame_count']; fps = PARAMETERS['fps']
    if count != b['frame_count']:
        raise ValueError('shared physical timeline required')
    groups = associate_entities(a['tracks'], b['tracks'])
    matches = matched_events(a, b, groups, count)
    records = []
    for i, event in enumerate(a['events']):
        track_id = event['track_id']; start = int(round(event['time_s'] * fps))
        valid_source = len(a['tracks'][track_id]) >= PARAMETERS['minimum_source_track_frames']
        rec_frames = {t for j in groups.get(track_id, []) for t in b['tracks'][j]}
        record = {'source_event_index': i, 'source_track_id': track_id, 'type': event['type'],
                  'source_time_s': event['time_s'], 'associated_reconstruction_tracks': groups.get(track_id, [])}
        readable = valid_source and bool(b['tracks'])
        available = max(0, count - 1 - start)
        delay_complete = available >= PARAMETERS['delay_horizon_frames']
        match = b['events'][matches[i]] if i in matches else None
        delay = max(0., match['time_s'] - event['time_s']) if match is not None and readable else None
        status = ('unobservable' if not readable else 'observed' if delay is not None else
                  'deadline_miss' if delay_complete else 'right_censored')
        record.update(delay_status=status, delay_s=delay, delay_observation_s=min(available, PARAMETERS['delay_horizon_frames']) / fps,
                      delay_complete_horizon=delay_complete,
                      delay_penalty=(delay * fps / PARAMETERS['delay_horizon_frames'] if delay is not None else
                                     1. if status == 'deadline_miss' else None))
        # Do not turn a shortened observation window into a full-horizon AUC.
        if event['type'] == 'exit':
            horizon = PARAMETERS['ghost_horizon_frames']; times = list(range(start, min(count, start + horizon)))
            anchor = bool(rec_frames & set(range(max(0, start - 4), start)))
            bits = [int(t in rec_frames) for t in times]
            ghost_status = ('unobservable' if not readable or not anchor else
                            'right_censored' if len(times) < horizon else 'observed')
            record.update(ghost_status=ghost_status, ghost_presence=bits,
                          ghost_observed_s=sum(bits) / fps,
                          ghost_auc=sum(bits) / horizon if ghost_status == 'observed' else None,
                          ghost_returns=sum(bits[k] and not bits[k-1] for k in range(1, len(bits))))
        records.append(record)
    ghost = [r['ghost_auc'] for r in records if r.get('ghost_auc') is not None]
    delay = [r['delay_penalty'] for r in records if r['delay_penalty'] is not None]
    # A finite number of event queries; max cannot create recall from absent events.
    return {'ghost_max': max(ghost) if ghost else None, 'delay_max': max(delay) if delay else None,
            'events': records, 'entity_groups': groups,
            'ghost_queries': sum(r['type'] == 'exit' for r in records), 'ghost_measured': len(ghost),
            'delay_queries': len(records), 'delay_measured': len(delay),
            'delay_deadline_misses': sum(r['delay_status'] == 'deadline_miss' for r in records),
            'scope': 'RGB-inferred events/entities; no truth, intervention label or oracle identity provided'}


def sfr_observed(a, b, groups):
    """SFR analogue on inferred entity IDs, not original label-packet SFR."""
    reverse = {j: i for i, js in groups.items() for j in js}
    def sets(observation, mapped=False):
        return [{reverse.get(j, -j-1) if mapped else j for j,tr in enumerate(observation['tracks']) if t in tr}
                for t in range(observation['frame_count'])]
    x, y = sets(a), sets(b, True); values = []
    for t in range(1, len(x)):
        births = (y[t] - y[t-1]) - (x[t] - x[t-1])
        deaths = (y[t-1] - y[t]) - (x[t-1] - x[t])
        values.append((len(births) + len(deaths)) / max(1, len(y[t] | y[t-1])))
    return float(np.mean(values)) if values else None
