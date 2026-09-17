import copy

import numpy as np
import pytest

from semantic_transmission import metric_revision as m


def observation(items, n=16):
    tracks = []
    for times, x, feature in items:
        mask = np.zeros((48, 80), bool)
        mask[16:28, x:x+10] = True
        tracks.append({t: dict(mask=mask.copy(), feature=np.asarray(feature, float),
                              center=np.array([x+4.5, 21.5])) for t in times})
    return dict(tracks=tracks, frame_count=n, shape=[48, 80])


def test_fragmented_same_object_is_explained_without_inventing_missing_frames():
    raw = observation([(range(8), 10, [1., 0.]), (range(9, 16), 10, [1., 0.])])
    before = copy.deepcopy(raw)
    b = m.observe(raw)
    a = m.observe(observation([(range(16), 10, [1., 0.])]))
    assert len(b['tracks']) == 1 and 8 not in b['tracks'][0]
    assert m.evaluate(a, b, a, ['ok']*16)['uep'] == 0
    assert 'velocity' not in raw['tracks'][0][0]
    np.testing.assert_array_equal(before['tracks'][0][0]['mask'], raw['tracks'][0][0]['mask'])


def test_geometric_match_survives_complete_feature_appearance_change():
    a = m.observe(observation([(range(16), 10, [1., 0.])]))
    b = m.observe(observation([(range(16), 10, [-1., 0.])]))
    assert m.evaluate(a, b, a, ['ok']*16)['uep'] == 0


def test_nearby_simultaneous_object_does_not_disappear_into_shared_identity():
    a = m.observe(observation([(range(16), 10, [1., 0.])]))
    b = m.observe(observation([(range(16), 10, [1., 0.]), (range(4, 8), 18, [1., 0.])]))
    assert len(b['tracks']) == 2
    assert m.evaluate(a, b, a, ['ok']*16)['uep'] == 1
    # Pixel-identical duplicate observations still cannot both consume one source slot.
    duplicate = m.observe(observation([(range(16), 10, [1., 0.]), (range(4, 8), 10, [1., 0.])]))
    assert m.evaluate(a, duplicate, a, ['ok']*16)['uep'] == 1


def test_rx_decode_status_matters_on_fixed_inputs():
    a = m.observe(observation([(range(16), 10, [1., 0.])]))
    b = m.observe(observation([(range(16), 10, [1., 0.]), (range(4, 8), 50, [0., 1.])]))
    assert m.evaluate(a, b, b, ['ok']*16)['uep'] == 0
    assert m.evaluate(a, b, b, ['corrupt']*16)['uep'] == 1
    status = ['corrupt']*16
    status[4:6] = ['ok', 'ok']
    assert m.evaluate(a, b, b, status)['uep'] == .5
    with pytest.raises(ValueError, match='RX timeline'):
        m.evaluate(a, b, b, [])


def test_missing_order_events_abstain_instead_of_returning_false_zero():
    a = dict(events=[dict(type=k, frame=t, track_id=0) for k, t in zip(('enter','turn','stop','exit'), (4,8,12,16))])
    b = dict(events=[dict(type=k, frame=t, track_id=0) for k, t in zip(('enter','exit'), (4,16))])
    value = m.order_score(a, b, {0: [0]})
    assert value['eoi_raw'] == 0 and value['eoi'] is None
    assert value['eoi_pair_coverage'] == pytest.approx(1/6)
    b['events'] = [dict(e, frame=20-e['frame']) for e in a['events']]
    assert m.order_score(a, b, {0: [0]})['eoi'] == 1


def test_annotation_and_observation_share_operational_motion_definition():
    raw = observation([(range(32), 10, [1., 0.])], n=32)
    centers = np.zeros((32, 1, 2))
    centers[:, 0, 1] = 21.5
    centers[:, 0, 0] = np.r_[np.arange(12)*2, np.full(20, 22.)]
    for t in range(32):
        raw['tracks'][0][t]['center'] = centers[t, 0]
        # Deliberately wrong flow must not enter the revised center contract.
        raw['tracks'][0][t]['velocity'] = np.array([999., 999.])
    rgb = m.observe(raw)
    truth = m.annotation_observation(centers, np.ones((32, 1), bool))
    assert rgb['events'] == truth['events']
    assert any(e['type'] == 'stop' for e in truth['events'])
    scores, _ = m.annotation_truth(truth, truth)
    assert scores['fso_motion'] == 0


def test_long_gaps_do_not_become_continuous_motion():
    raw = observation([(range(4), 10, [1., 0.]), (range(12, 16), 10, [1., 0.])])
    assert len(m.observe(raw)['tracks']) == 2
    points, velocity = m.kinematics({**raw['tracks'][0], **raw['tracks'][1]})
    assert 8 not in points and 3 not in velocity and 11 not in velocity


def test_natural_addition_keeps_known_truth_and_never_invents_other_labels():
    from semantic_transmission.metric_revision_audit import natural_truth
    gt, records, source = natural_truth(dict(kind='addition', truth={'uep': .5}), None)
    assert gt == {'uep': .5} and records == [] and source is None


def test_independent_visual_review_rejects_uncertainty_as_a_zero_label():
    import runpy
    from pathlib import Path
    validate = runpy.run_path(str(Path(__file__).parents[1]/'scripts/assess_metric_revision_review.py'))['validate_notes']
    row = dict(review_id='R01', uep=0., uncertain_frames=[[2, 4]], source_visible=[[0, 32]],
               reconstruction_visible=[[0, 32]], observations='unclear', fso_eoi_unavailable_reason='no query')
    notes = dict(schema='metric-revision-ai-visual-notes-v1', rows=[row])
    index = dict(rows=[dict(review_id='R01')])
    with pytest.raises(ValueError, match='Uncertain'):
        validate(notes, index)
    row['uep'] = None
    assert validate(notes, index)['R01']['uep'] is None
    row['uep'] = False
    with pytest.raises(ValueError, match='coarse review'):
        validate(notes, index)


def test_visual_review_duplicate_ids_and_invalid_intervals_fail():
    import runpy
    from pathlib import Path
    validate = runpy.run_path(str(Path(__file__).parents[1]/'scripts/assess_metric_revision_review.py'))['validate_notes']
    row = dict(review_id='R01', uep=None, uncertain_frames=[], source_visible=[[0, 33]],
               reconstruction_visible=[], observations='review', fso_eoi_unavailable_reason='no query')
    notes = dict(schema='metric-revision-ai-visual-notes-v1', rows=[row])
    index = dict(rows=[dict(review_id='R01')])
    with pytest.raises(ValueError, match='interval'):
        validate(notes, index)
    notes['rows'] = [row, row]
    with pytest.raises(ValueError, match='duplicate'):
        validate(notes, index)


def test_saved_revision_rows_cannot_be_silently_overwritten(tmp_path):
    from semantic_transmission.metric_revision_audit import _write_once
    path = tmp_path/'row.json'
    _write_once(path, {'uep': 1.})
    _write_once(path, {'uep': 1.})
    with pytest.raises(ValueError, match='overwrite'):
        _write_once(path, {'uep': 0.})


def test_repeated_oracle_events_do_not_get_fabricated_order_identity():
    a = m.annotation_observation(np.zeros((16, 1, 2)), np.ones((16, 1), bool))
    a['events'] = [dict(type='turn', frame=t, track_id=0) for t in [4, 8]]
    assert m.annotation_truth(a, a)[0]['eoi'] is None
