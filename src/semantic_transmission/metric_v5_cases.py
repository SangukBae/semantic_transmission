"""Fresh scene families and code-recorded truth for the FSO/EOI/UEP experiment.

Every family carries all four anchoring source events (turn, stop, enter, exit)
so each forbidden-state instance receives positives from every source video.
Turns are near-180 degree reversals: the frozen RGB extractor only declares a
turn above 150 degrees, so a 91 degree truth turn would be unobservable by
construction rather than by failure.

Normal controls repeat the frozen v4 control set unchanged (identity, brightness,
texture, camera, palette) so appearance invariance is tested identically.
"""
import cv2
import numpy as np

from .metric_v3_cases import annotation_events

FPS, COUNT, HEIGHT, WIDTH = 8., 32, 192, 320
FAMILIES = ('turn_exit_relay', 'stop_first_relay', 'late_entrance_relay')
CONTROLS = ('identity', 'brightness20', 'texture', 'camera', 'palette')
# family -> (turn, exit, stop, enter) anchor frames. Every pair that the order
# intervention exchanges keeps a gap above the 0.375s event separation even after
# the per-seed jitter, and the exchange never moves object 0's own two events.
SCHEDULE = {'turn_exit_relay': (7, 25, 12, 18),
            'stop_first_relay': (12, 24, 7, 19),
            'late_entrance_relay': (14, 26, 20, 6)}
SWAP = {'turn_exit_relay': ('stop', 'enter'), 'stop_first_relay': ('stop', 'enter'),
        'late_entrance_relay': ('enter', 'stop')}
TARGETS = {'presence': ('exit', 1), 'premature': ('enter', 3), 'motion': ('stop', 2), 'heading': ('turn', 1)}


def scene(seed, family):
    rng = np.random.default_rng(seed)
    turn, exit_t, stop_t, enter_t = SCHEDULE[family]
    times = {'turn': turn + int(rng.integers(-1, 2)), 'exit': exit_t + int(rng.integers(-1, 2)),
             'stop': stop_t + int(rng.integers(-1, 2)), 'enter': enter_t + int(rng.integers(-1, 2))}
    origins = np.array([[62., 46.], [243., 44.], [148., 143.], [272., 139.]])
    origins += rng.uniform(-4, 4, origins.shape)
    speeds = rng.uniform(1.9, 2.4, 3)
    velocities = {0: np.array([speeds[0], 0.6]), 1: np.array([-speeds[1], 0.]),
                  2: np.array([speeds[2] * .6, 0.]), 3: np.zeros(2)}
    colors = np.array([[205., 105., 80.], [95., 190., 125.], [85., 135., 220.], [220., 185., 95.]])
    colors += rng.uniform(-16, 16, (4, 3))
    return {'seed': seed, 'family': family, 'times': times, 'origins': origins,
            'velocities': {k: v for k, v in velocities.items()}, 'colors': colors,
            'sizes': rng.integers(15, 20, 4)}


def trajectory(state, times=None, override=None):
    """Object 0 turns then exits, 1 stops, 2 enters, 3 stays. Overrides edit one object."""
    times = dict(state['times'] if times is None else times)
    override = override or {}
    centers = np.zeros((COUNT, 4, 2), np.float32)
    visible = np.ones((COUNT, 4), bool)
    turn, exit_t, stop_t, enter_t = (times['turn'], times['exit'], times['stop'], times['enter'])
    delay = int(override.get('delay', 0))
    kind = override.get('kind')
    for q in range(COUNT):
        # object 0: reversal at `turn`, with an optional stale-heading overrun.
        shift = turn + (delay if kind == 'turn_overrun' else 0)
        forward = min(q, shift)
        backward = max(0, q - shift)
        centers[q, 0] = state['origins'][0] + state['velocities'][0] * (forward - backward)
        # object 1: stop at `stop`, with an optional residual-motion overrun.
        extent = min(q, stop_t + (delay if kind == 'stop_overrun' else 0))
        centers[q, 1] = state['origins'][1] + state['velocities'][1] * extent
        # object 2: enters at `enter` and then moves steadily.
        centers[q, 2] = state['origins'][2] + state['velocities'][2] * max(0, q - enter_t)
        centers[q, 3] = state['origins'][3]
    visible[:, 2] = np.arange(COUNT) >= enter_t
    visible[exit_t:, 0] = False
    if kind == 'ghost_hold':
        for q in range(exit_t, min(COUNT, exit_t + delay)):
            visible[q, 0] = True
            centers[q, 0] = centers[exit_t - 1, 0]
    elif kind == 'ghost_return':
        for q in range(exit_t + delay, min(COUNT, exit_t + delay + 2)):
            visible[q, 0] = True
            centers[q, 0] = centers[exit_t - 1, 0]
    elif kind == 'premature_enter':
        for q in range(max(0, enter_t - delay), enter_t):
            visible[q, 2] = True
            centers[q, 2] = centers[enter_t, 2]
    return centers, visible


def draw(state, centers=None, visible=None, style='identity'):
    """Same renderer contract as the frozen v4 cases, with this family's shapes."""
    if centers is None or visible is None:
        centers, visible = trajectory(state)
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    frames, labels = [], []
    for t in range(COUNT):
        base = 44 + 5 * np.sin(xx * .075) + 4 * np.cos(yy * .105)
        if style == 'texture':
            base += 20 * np.sin(xx * .4) * np.cos(yy * .29)
        frame = np.clip(base[..., None] + np.array([0, 7, 12]), 0, 255).astype(np.uint8)
        mask = np.zeros((HEIGHT, WIDTH), np.uint8)
        for k in range(4):
            if not visible[t, k]:
                continue
            x, y = centers[t, k]
            size = state['sizes'][k]
            u, w = (xx - x) / size, (yy - y) / size
            shape = (k + FAMILIES.index(state['family'])) % 3
            if shape == 0:
                inside = (np.abs(u) + np.abs(w)) <= 1.3
            elif shape == 1:
                inside = (u / 1.2) ** 2 + (w / .85) ** 2 <= 1
            else:
                inside = (np.abs(u) <= 1) & (np.abs(w) <= 1)
            detail = 9 * np.sin(u * 4) * np.cos(w * 5)
            if style == 'texture':
                detail = 23 * np.sin(u * 10) * np.cos(w * 8)
            color = np.clip(state['colors'][k] + detail[..., None], 0, 255).astype(np.uint8)
            frame[inside] = color[inside]
            mask[inside] = k + 1
        if style == 'brightness20':
            frame = np.clip(frame.astype(float) * 1.15 + 20, 0, 255).astype(np.uint8)
        elif style == 'palette':
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            hsv[..., 0] = (hsv[..., 0].astype(int) + 45) % 180
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        elif style == 'camera':
            matrix = cv2.getRotationMatrix2D((WIDTH / 2, HEIGHT / 2), 1.5 * np.sin(t * .4), 1.)
            matrix[:, 2] += [2 * np.sin(t * .8), 2 * np.cos(t * .6)]
            frame = cv2.warpAffine(frame, matrix, (WIDTH, HEIGHT), borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_NEAREST)
        frames.append(frame)
        labels.append(mask)
    return np.stack(frames), {'centers': np.asarray(centers, np.float32).copy(),
                              'visible': np.asarray(visible).copy(), 'masks': np.stack(labels)}


INTERVENTIONS = (('ghost_hold', 'presence', (2, 4, 8)), ('ghost_return', 'presence', (2,)),
                 ('premature_enter', 'premature', (2, 4)), ('stop_overrun', 'motion', (2, 4, 8)),
                 ('turn_overrun', 'heading', (2, 4)))


def variants(state):
    """Controls first, then one intervention per case with its declared target."""
    for style in CONTROLS:
        frames, truth = draw(state, style=style)
        yield style, frames, truth, {'target': 'control', 'kind': 'control', 'instance': None,
                                     'severity': 0., 'target_object': None, 'target_event': None,
                                     'target_frame': None}
    for kind, instance, delays in INTERVENTIONS:
        event, obj = TARGETS[instance]
        for d in delays:
            centers, visible = trajectory(state, override={'kind': kind, 'delay': d})
            frames, truth = draw(state, centers, visible)
            yield f'{kind}_{d}', frames, truth, {'target': instance, 'kind': kind, 'instance': instance,
                                                 'severity': d / FPS, 'target_object': obj,
                                                 'target_event': event, 'target_frame': state['times'][event]}
    first, second = SWAP[state['family']]
    times = dict(state['times'])
    times[first], times[second] = times[second], times[first]
    centers, visible = trajectory(state, times)
    frames, truth = draw(state, centers, visible)
    yield 'order_swap', frames, truth, {'target': 'order', 'kind': 'order_swap', 'instance': 'order',
                                        'severity': abs(state['times'][first] - state['times'][second]) / FPS,
                                        'target_object': None, 'target_event': None, 'target_frame': None,
                                        'swapped_events': [first, second]}


def truth_order_inversions(source, reconstruction):
    """Independent ordered-pair inversion count from simulator identity and time."""
    from scipy.optimize import linear_sum_assignment
    a = annotation_events(source['centers'], source['visible'])
    b = annotation_events(reconstruction['centers'], reconstruction['visible'])
    cost = np.full((len(a), len(b)), 1e6)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x['object_id'] == y['object_id'] and x['type'] == y['type']:
                cost[i, j] = abs(x['time_s'] - y['time_s'])
    matches = {int(i): int(j) for i, j in zip(*linear_sum_assignment(cost)) if cost[i, j] < 1e6}
    pairs = inverted = 0
    import itertools
    for i, j in itertools.combinations(sorted(matches), 2):
        if a[j]['time_s'] - a[i]['time_s'] < .375 - 1e-9:
            continue
        pairs += 1
        inverted += int(b[matches[i]]['time_s'] - b[matches[j]]['time_s'] >= .375 - 1e-9)
    return {'truth_ordered_pairs': pairs, 'truth_inversions': inverted,
            'truth_eoi': inverted / pairs if pairs else None,
            'truth_source_events': len(a), 'truth_reconstruction_events': len(b)}
