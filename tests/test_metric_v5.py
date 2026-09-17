import numpy as np
import pytest

from semantic_transmission.forbidden_state_metric import (INSTANCES, PARAMETERS, evaluate,
                                                          truth_occupancy, unsupported_presence)
from semantic_transmission.metric_v5_cases import (FAMILIES, INTERVENTIONS, SWAP, TARGETS, draw,
                                                   scene, trajectory, truth_order_inversions, variants)
from semantic_transmission.metric_v3_cases import annotation_events


def track(times, velocity=(0., 0.), feature=(1., 0.), rows=(8, 14), cols=(8, 14), shape=(24, 32)):
    mask = np.zeros(shape, bool)
    mask[rows[0]:rows[1], cols[0]:cols[1]] = True
    vector = np.asarray(feature, float)
    vector = vector / np.linalg.norm(vector)
    center = np.array([(cols[0] + cols[1]) / 2., (rows[0] + rows[1]) / 2.])
    return {int(t): {'mask': mask, 'feature': vector, 'center': center,
                     'velocity': np.asarray(velocity(t) if callable(velocity) else velocity, float),
                     'camera_velocity': np.zeros(2)} for t in times}


def observation(tracks, events, count=20, shape=(24, 32)):
    return {'tracks': list(tracks), 'frame_count': count, 'shape': list(shape),
            'events': [{'type': kind, 'time_s': t / 8., 'track_id': k, 'direction': 0,
                        'speed_bin': 0, 'cell': [1, 1]} for kind, t, k in events]}


def test_presence_instance_reproduces_the_post_exit_occupancy():
    a = observation([track(range(8))], [('exit', 8, 0)])
    b = observation([track(range(10))], [('exit', 10, 0)])
    result = evaluate(a, b)
    assert result['fso_presence'] == .5 and result['fso_presence_raw'] == .5
    assert result['fso_max'] == .5


def test_source_difference_cancels_an_entity_that_never_left_the_source():
    """The documented false-alarm cause: a broken source track, not a real exit."""
    persisting = track(range(8, 20), feature=(1., .02))
    a = observation([track(range(8)), persisting], [('exit', 8, 0)])
    b = observation([track(range(20))], [('exit', 20, 0)])
    result = evaluate(a, b)
    assert result['fso_presence_raw'] == 1.
    assert result['fso_presence'] == 0.


def test_premature_presence_uses_the_window_before_the_entrance():
    a = observation([track(range(10, 20))], [('enter', 10, 0)])
    b = observation([track(range(8, 20))], [('enter', 8, 0)])
    result = evaluate(a, b)
    assert result['fso_premature'] == .5
    assert result['fso_presence'] is None


def test_motion_floor_is_a_fraction_of_the_entity_own_pre_anchor_speed():
    moving = track(range(20), velocity=lambda t: (2., 0.) if t < 8 else (0., 0.))
    noisy = track(range(20), velocity=lambda t: (2., 0.) if t < 8 else (.4, 0.))
    overrun = track(range(20), velocity=lambda t: (2., 0.) if t < 10 else (0., 0.))
    a = observation([moving], [('stop', 8, 0)])
    assert evaluate(a, observation([noisy], [('stop', 8, 0)]))['fso_motion'] == 0.
    assert evaluate(a, observation([overrun], [('stop', 10, 0)]))['fso_motion'] == .5


def test_heading_state_is_measured_against_the_same_video_pre_anchor_heading():
    source = track(range(20), velocity=lambda t: (2., 0.) if t < 8 else (-2., 0.))
    stale = track(range(20), velocity=lambda t: (2., 0.) if t < 10 else (-2., 0.))
    a = observation([source], [('turn', 8, 0)])
    assert evaluate(a, observation([track(range(20), velocity=lambda t: (2., 0.) if t < 8 else (-2., 0.))],
                                   [('turn', 8, 0)]))['fso_heading'] == 0.
    assert evaluate(a, observation([stale], [('turn', 10, 0)]))['fso_heading'] == .5


def test_heading_score_survives_a_global_rotation_of_the_reconstruction():
    angle = np.deg2rad(37.)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    def stale(t):
        return rotation @ (np.array([2., 0.]) if t < 10 else np.array([-2., 0.]))
    a = observation([track(range(20), velocity=lambda t: (2., 0.) if t < 8 else (-2., 0.))], [('turn', 8, 0)])
    b = observation([track(range(20), velocity=stale)], [('turn', 10, 0)])
    assert evaluate(a, b)['fso_heading'] == .5


def test_truncated_and_unobservable_windows_are_not_zero_error():
    a = observation([track(range(18))], [('exit', 18, 0)])
    b = observation([track(range(20))], [('exit', 20, 0)])
    records = [r for r in evaluate(a, b)['fso_records'] if r['instance'] == 'presence']
    assert records[0]['status'] == 'right_censored' and records[0]['score'] is None
    assert evaluate(a, b)['fso_presence'] is None
    empty = observation([], [])
    assert evaluate(a, empty)['fso_records'][0]['status'] == 'unobservable'


def test_motion_without_observed_pre_anchor_motion_is_unobservable():
    a = observation([track(range(20), velocity=(0., 0.))], [('stop', 8, 0)])
    b = observation([track(range(20), velocity=(0., 0.))], [('stop', 8, 0)])
    records = [r for r in evaluate(a, b)['fso_records'] if r['instance'] == 'motion']
    assert records and all(r['status'] == 'unobservable' for r in records)


def test_event_order_inversion_counts_matched_pairs_only():
    a = observation([track(range(20)), track(range(20), feature=(0., 1.), rows=(2, 6), cols=(20, 26))],
                    [('stop', 4, 0), ('enter', 12, 1)])
    swapped = observation([track(range(20)), track(range(20), feature=(0., 1.), rows=(2, 6), cols=(20, 26))],
                          [('stop', 12, 0), ('enter', 4, 1)])
    assert evaluate(a, swapped)['eoi'] == 1.
    ordered = observation([track(range(20)), track(range(20), feature=(0., 1.), rows=(2, 6), cols=(20, 26))],
                          [('stop', 4, 0), ('enter', 12, 1)])
    assert evaluate(a, ordered)['eoi'] == 0.
    partial = observation([track(range(20)), track(range(20), feature=(0., 1.), rows=(2, 6), cols=(20, 26))],
                          [('stop', 4, 0)])
    result = evaluate(a, partial)
    assert result['eoi_ordered_pairs'] == 0 and result['eoi'] is None


def test_unsupported_presence_needs_both_no_source_and_no_received_evidence():
    source = observation([track(range(20))], [])
    added = track(range(8, 14), feature=(0., 1.), rows=(2, 6), cols=(20, 26))
    recon = observation([track(range(20)), added], [])
    rx = observation([track(range(20))], [])
    value = unsupported_presence(source, recon, rx, ['ok'] * 20)
    assert value['uep'] == 1. and value['uep_unexplained_tracks'] == 1
    supported = observation([track(range(20)), added], [])
    assert unsupported_presence(source, recon, supported, ['ok'] * 20)['uep'] == 0.
    # Undecodable slots carry no evidence, so the same RGB no longer supports it.
    assert unsupported_presence(source, recon, supported, ['packet_lost'] * 20)['uep'] == 1.
    assert unsupported_presence(source, source, rx, ['ok'] * 20)['uep'] == 0.


def test_unsupported_presence_saturates_at_the_fixed_horizon():
    source = observation([track(range(20))], [])
    horizon = PARAMETERS['unsupported_horizon_frames']
    for frames, expected in ((2, 2 / horizon), (horizon, 1.), (12, 1.)):
        added = track(range(4, 4 + frames), feature=(0., 1.), rows=(2, 6), cols=(20, 26))
        recon = observation([track(range(20)), added], [])
        assert unsupported_presence(source, recon, observation([track(range(20))], []), ['ok'] * 20)['uep'] == expected


@pytest.mark.parametrize('family', FAMILIES)
def test_new_scene_truth_carries_every_anchor_event(family):
    state = scene(260915300, family)
    centers, visible = trajectory(state)
    events = annotation_events(centers, visible)
    assert {(e['type'], e['object_id']) for e in events} == {('turn', 1), ('exit', 1), ('stop', 2), ('enter', 3)}
    assert len(events) == 4


@pytest.mark.parametrize('family', FAMILIES)
def test_every_intervention_raises_its_own_target_oracle(family):
    state = scene(260915601, family)
    centers, visible = trajectory(state)
    for kind, instance, delays in INTERVENTIONS:
        event, obj = TARGETS[instance]
        frame = state['times'][event]
        for d in delays:
            c, v = trajectory(state, override={'kind': kind, 'delay': d})
            oracle = truth_occupancy(instance, c, v, obj, frame)
            base = truth_occupancy(instance, centers, visible, obj, frame)
            assert oracle['occupancy'] == min(d / 4., 1.) if kind != 'ghost_return' else oracle['occupancy'] == .5
            assert base['occupancy'] == 0.


@pytest.mark.parametrize('family', FAMILIES)
def test_order_intervention_inverts_a_truth_pair_and_controls_do_not(family):
    state = scene(260915601, family)
    frames, truth = draw(state)
    first, second = SWAP[family]
    times = dict(state['times'])
    times[first], times[second] = times[second], times[first]
    centers, visible = trajectory(state, times)
    assert truth_order_inversions(truth, {'centers': centers, 'visible': visible})['truth_inversions'] >= 1
    for name, _, control, meta in variants(state):
        if meta['target'] != 'control':
            continue
        assert np.array_equal(truth['centers'], control['centers'])
        assert np.array_equal(truth['visible'], control['visible'])


def test_instance_map_declares_one_anchor_event_per_candidate():
    assert set(INSTANCES) == {'presence', 'premature', 'motion', 'heading'}
    assert {v[0] for v in INSTANCES.values()} == {'exit', 'enter', 'stop', 'turn'}
