import json

import numpy as np
import pytest

from semantic_transmission.attribution_metric import attribution, support, decode_keyframe, rule_checks
from semantic_transmission.event_metric import event_pairs, residual_regions, score_events, track_events, presence_events
from semantic_transmission.metric_v3_cases import annotation_events, audit_truth, render_scene, RESTYLES


def event(t, kind="turn", direction=0, cell=(1, 1)):
    return {"kind": "event", "type": kind, "time_s": t, "direction": direction, "cell": list(cell)}


def test_ere_short_missing_event_is_counted_without_time_weight():
    a = [event(.5), event(1.), event(2.), event(3.)]
    assert score_events(a, [a[0], a[2], a[3]])['ere'] == .25
    assert score_events(a[:1], a[:1])['ere'] is None
    assert score_events([], [])['ser'] is None


def test_events_one_to_one_direction_cells_time_and_type():
    a = [event(.5), event(1.)]
    assert len(event_pairs(a, [event(.5), event(.5)])) == 1
    assert not event_pairs(a, [event(.5, direction=4)])
    assert not event_pairs([event(.5, cell=(0, 0))], [event(.5, cell=(2, 2))])
    assert not event_pairs(a, [event(3.)])
    assert not event_pairs(a, [event(.5, 'stop')])
    assert event_pairs([event(.5, direction=7)], [event(.75, direction=0)]) == [(0, 0)]


def test_residual_global_camera_does_not_become_object_motion():
    flow = np.zeros((3, 80, 120, 2), np.float32); flow[..., 0] = 2
    regions, fits = residual_regions(flow)
    assert not any(regions)
    assert min(fits) >= .5
    flow[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        residual_regions(flow)


def test_two_opposed_regions_remain_separate():
    flow = np.zeros((3, 80, 120, 2), np.float32)
    flow[:, 10:25, 10:30, 0] = 2
    flow[:, 50:65, 80:100, 0] = -2
    regions, _ = residual_regions(flow)
    assert all(len(r) == 2 for r in regions)
    assert regions[0][0]['velocity'][0] > 0 > regions[0][1]['velocity'][0]


def test_track_boundary_not_counted_as_enter_exit():
    mask = np.ones((10, 10), bool)
    track = {t: {"center": np.array([t * 2., 5.]), "mask": mask, "velocity": np.array([2., 0.])} for t in range(15)}
    events = track_events([track], 16, (10, 40))
    assert not any(e['type'] in ('enter', 'exit') for e in events)


def test_annotation_truth_rejects_unchanged_motion_accepts_shape_separately():
    _, source = render_scene(260915100)
    assert len(annotation_events(source['centers'], source['visible'])) >= 4
    assert not audit_truth(source, source, 'reverse', 'reverse_0.125')['accepted']
    _, shape = render_scene(260915100, kind='object_shape', severity=.25)
    audited = audit_truth(source, shape, 'object_shape', 'object_shape_0.25')
    assert audited['accepted'] and not audited['event_changed']


@pytest.mark.parametrize('style', RESTYLES)
def test_renderer_restylings_preserve_declared_truth(style):
    source, a = render_scene(260915100)
    restyled, b = render_scene(260915100, restyle=style)
    assert np.any(source != restyled)
    assert audit_truth(a, b, 'control', style)['accepted']


def units():
    return [event(.5, 'start'), event(1.5, 'turn', 4)]


def representation():
    return {"keyframes": [{"time_s": u['time_s'], 'decode_status': 'ok', 'decoded_events': [u]} for u in units()], "captions": []}


def test_sta_distinguishes_four_explicit_support_failures():
    a = units(); rep = representation(); empty = {"keyframes": [], "captions": []}
    control = attribution(a, a, rep, rep, [(0, 0), (1, 1)])
    assert [control[k] for k in ('edr', 'clr', 'gfr', 'ghr')] == [0, 0, 0, 0]
    tx_loss = attribution(a, [], empty, empty, [])
    assert tx_loss['edr'] == 1 and tx_loss['clr'] is None and tx_loss['gfr'] is None
    channel_loss = attribution(a, [], rep, empty, [])
    assert channel_loss['clr'] == 1 and channel_loss['edr'] == 0 and channel_loss['gfr'] is None
    gen_loss = attribution(a, [], rep, rep, [])
    assert gen_loss['gfr'] == 1 and gen_loss['clr'] == 0
    extra = {"kind": "object", "object_token": "triangle"}
    gen_extra = attribution(a, a + [extra], rep, rep, [(0, 0), (1, 1)])
    assert gen_extra['ghr'] == pytest.approx(1 / 3)


def test_sta_support_uses_time_tokens_and_not_substrings():
    unit = event(1., 'turn', 4)
    assert support(unit, {"captions": [{"start_s": .5, "end_s": 1.5, "text": "turn west"}]})
    assert not support(unit, {"captions": [{"start_s": .5, "end_s": 1.5, "text": "return western"}]})
    assert not support(unit, {"captions": [{"start_s": 2., "end_s": 3., "text": "turn west"}]})
    content = {'decode_status': 'ok', 'decoded_events': [unit]}
    assert support(unit, {"keyframes": [{"time_s": 1.25, **content}]})
    assert not support(unit, {"keyframes": [{"time_s": 1.251, **content}]})
    assert not support(unit, {'keyframes': [{'time_s': 1.0}]})


def test_sta_no_support_and_duplicate_pair_handling():
    a = units(); rep = representation()
    assert all(attribution([], [], {}, {}, [])[k] is None for k in ('edr', 'clr', 'gfr', 'ghr'))
    with pytest.raises(ValueError):
        attribution(a, a, rep, rep, [(0, 0), (0, 1)])
    with pytest.raises(ValueError):
        attribution(a, a, rep, rep, [(0, 9)])


def test_sta_retained_corrupt_content_is_channel_loss():
    a = units(); rep = representation()
    corrupt = {"keyframes": [{"time_s": .5, "content_corrupted": True}, {"time_s": 1.5, "content_corrupted": True}]}
    scored = attribution(a, [], rep, corrupt, [])
    assert scored['clr'] == 1 and scored['gfr'] is None


def test_gate_refuses_scoring_when_artifacts_missing(tmp_path):
    from semantic_transmission.metric_v3_validation import require_gates
    with pytest.raises((ValueError, FileNotFoundError)):
        require_gates(tmp_path)


def motion_track(positions):
    positions = np.asarray(positions, float)
    velocity = np.diff(positions, axis=0)
    return {t: {'center': p, 'velocity': velocity[t] if t < len(velocity) else None,
                'camera_velocity': np.zeros(2) if t < len(velocity) else None} for t, p in enumerate(positions)}


def test_presence_retains_stopped_object_without_false_enter_exit():
    points = []; x = 40.
    for t in range(32):
        points.append([x, 45.])
        if 5 <= t < 15 or 25 <= t:
            x += 3
    events = presence_events([motion_track(points)], 32, (100, 200))
    assert [(e['type'], e['time_s']) for e in events] == [('start', 5/8), ('stop', 15/8), ('start', 25/8)]


def test_presence_birth_death_of_stationary_object():
    tr = motion_track([[50., 50.]] * 32)
    tr = {t: v for t, v in tr.items() if 6 <= t <= 23}
    events = presence_events([tr], 32, (100, 200))
    assert [(e['type'], e['time_s']) for e in events] == [('enter', 6/8), ('exit', 24/8)]


def test_persistent_object_turns_instead_of_disappearing():
    points=[];x=40.
    for t in range(32):
        points.append([x, 50.]);x += 3 if t < 16 else -3
    e = presence_events([motion_track(points)], 32, (100, 200))
    assert len(e) == 1 and e[0]['type'] == 'turn'
    assert abs(e[0]['time_s'] - 2.) <= .25


@pytest.mark.parametrize('status', ['packet_lost', 'checksum_failed', 'decode_failed', 'extraction_failed'])
def test_sta_invalid_status_overrides_stale_decoded_content(status):
    unit=units()[0]
    k={'time_s': unit['time_s'], 'decode_status': status, 'decoded_events': [unit], 'decoded_object_tokens': ['triangle']}
    assert not support(unit, {'keyframes': [k]})
    assert not support({'kind':'object', 'object_token':'triangle'}, {'keyframes':[k]})


def test_sta_content_state_must_agree_and_caption_can_supply_independent_support():
    unit=units()[0]
    for state in ({**unit,'direction':4}, {**unit,'type':'stop'}, {**unit,'cell':[2,2]}):
        assert not support({**unit,'cell':[0,0]}, {'keyframes':[{'time_s':.5,'decode_status':'ok','decoded_events':[state]}]})
    rx={'keyframes':[{'time_s':.5,'decode_status':'checksum_failed'}],
        'captions':[{'start_s':0.,'end_s':1.,'text':'start east'}]}
    assert support(unit,rx)


def test_decode_keyframe_checks_bytes_and_extraction_failure():
    import cv2,hashlib
    frame=np.zeros((8,8,3),np.uint8);frame[...,0]=123
    ok,data=cv2.imencode('.png',cv2.cvtColor(frame,cv2.COLOR_RGB2BGR));assert ok
    payload=data.tobytes();checksum=hashlib.sha256(payload).hexdigest()
    def reader(rgb):
        assert np.array_equal(rgb,frame)
        return {'events':[units()[0]],'object_tokens':['triangle']}
    k=decode_keyframe(payload,.5,reader,checksum)
    assert k['decode_status']=='ok' and support(units()[0],{'keyframes':[k]})
    assert decode_keyframe(payload+b'x',.5,reader,checksum)['decode_status']=='checksum_failed'
    assert decode_keyframe(None,.5,reader)['decode_status']=='packet_lost'
    assert decode_keyframe(b'not an image',.5,reader)['decode_status']=='decode_failed'
    assert decode_keyframe(payload,.5,lambda rgb: {'observable':False})['decode_status']=='extraction_failed'


def test_all_six_declared_sta_rule_cases():
    checks=rule_checks()
    assert checks['status']=='PASSED' and checks['passed']==checks['cases']==6


@pytest.mark.parametrize('failure', ['turn', 'sta'])
def test_new_mandatory_gates_block_otherwise_acceptable_extractor(tmp_path, monkeypatch, failure):
    from types import SimpleNamespace
    from semantic_transmission import metric_v3_validation as validation
    from semantic_transmission import attribution_metric
    monkeypatch.setattr(validation, 'verify', lambda root: {})
    events={'results':[{'source_id':'example', 'domain':'rendered', 'true_events':10,
                       'predicted_event_count':10, 'matched_events':8, 'true_by_type':{'turn':2},
                       'matched_by_type':{'turn':1 if failure=='turn' else 2}}]}
    objects={'results':[{'domain':'rendered','annotated_object_frames':100,'matched_object_frames':90}]}
    (tmp_path/'event_extractor_audit.json').write_text(json.dumps(events))
    (tmp_path/'object_extractor_audit.json').write_text(json.dumps(objects))
    if failure=='sta':
        monkeypatch.setattr(attribution_metric,'rule_checks',lambda:{'status':'NOT_PASSED','cases':6,'passed':5})
    validation.gate_summary(SimpleNamespace(output=tmp_path))
    gate=json.loads((tmp_path/'extractor_gate.json').read_text())
    assert gate['gates']['rendered/event']['passed']
    assert gate['gates']['rendered/object']['passed']
    assert not gate['gates']['rendered/turn' if failure=='turn' else 'sta/rules']['passed']
    assert gate['status']=='NOT_PASSED'
