import hashlib

import pytest

from semantic_transmission import bsu_ltx_plus as experiment
from semantic_transmission.artifacts import write_json
from semantic_transmission.bsu_sequence_refinement import POLICY as TRACK_POLICY, decode_packet
from semantic_transmission.wire import pack, unpack


def packets():
    header = dict(schema='lgvsc_model_inputs_v1', video=dict(frames=9, fps=24, width=576, height=320),
        keyframes=[dict(index=0), dict(index=8)], segments=[dict(path='sample', text='A person moves.', flow=1.)])
    old = pack(header)
    new = pack(dict(header, schema=experiment.codec.SCHEMA, visual_codec=experiment.codec.SPEC))
    track = pack(dict(policy=TRACK_POLICY, video=header['video'],
        base_metadata_sha256=hashlib.sha256(old).hexdigest(), windows=[]))
    return old, new, track


def test_trajectory_rebinding_preserves_content_and_requires_new_packet():
    old, new, track = packets()
    bound = experiment.rebind_trajectory(track, old, new)
    before, _ = unpack(track); after, _ = unpack(bound)
    assert len(track) == len(bound)
    assert after.pop('base_metadata_sha256') == hashlib.sha256(new).hexdigest()
    before.pop('base_metadata_sha256')
    assert before == after
    assert decode_packet(bound, new) == {}
    with pytest.raises(ValueError, match='does not match'):
        decode_packet(track, new)


@pytest.mark.parametrize('field,value', [('segments', []), ('keyframes', [dict(index=0), dict(index=4), dict(index=8)]),
                                       ('video', dict(frames=17, fps=24, width=576, height=320))])
def test_codec_ablation_rejects_changed_semantic_inputs(field, value):
    old, new, track = packets()
    header, _ = unpack(new); header[field] = value
    with pytest.raises(ValueError, match='semantic conditions'):
        experiment.rebind_trajectory(track, old, pack(header))


def comparison_fixture():
    q = dict(source_sha256='source', delivered_mp4=dict(frames=9, **{k:1. for k in experiment.previous.KEYS}))
    row = dict(id='video', frames=9, bsu_channel_uses=20, lgvsc_channel_uses=10,
               bsu_quality=q, lgvsc_quality=q, ltx_quality=q)
    result = dict(id='video', status='COMPLETE_LTX_NTSCC_PLUS_VIDEO', frames=9, quality=q, total_channel_uses=23)
    return dict(rows=[row], frames=9), result


def test_report_preserves_measured_cost_difference(tmp_path):
    audit, result = comparison_fixture()
    experiment.summarize(tmp_path, audit, [result], 'identity')
    r = experiment.read(tmp_path / 'RESULT.json')
    assert r['status'] == experiment.COMPLETE
    assert r['arms']['BSU/LTX/NTSCC+']['total_channel_uses'] == 23
    assert r['arms']['BSU/LTX/NTSCC']['total_channel_uses'] == 20
    assert not r['equal_rate_comparison'] and not r['action_accuracy_evaluated']


def test_incomplete_or_wrong_reference_cannot_complete(tmp_path):
    audit, result = comparison_fixture()
    with pytest.raises(ValueError, match='incomplete'):
        experiment.summarize(tmp_path, audit, [], 'identity')
    result = dict(result, frames=8)
    with pytest.raises(ValueError, match='coverage'):
        experiment.summarize(tmp_path, audit, [result], 'identity')
    assert not (tmp_path / 'RESULT.json').exists()


def test_status_distinguishes_completed_intervals_from_current_attempt(tmp_path):
    assert experiment.status(tmp_path) == dict(status='NOT_STARTED')
    write_json(tmp_path / 'status.json', dict(status='RUNNING', video='video', completed=0, total=14))
    dest = tmp_path / 'videos/video'
    write_json(dest / 'conditions/request.json', dict(windows=[{}, {}]))
    write_json(dest / 'raw/stages/interval_00000.json', dict(status='PASSED'))
    write_json(dest / 'raw/stages/interval_00001.json', dict(status='RUNNING'))
    assert experiment.status(tmp_path)['current_video_intervals'] == dict(completed=1, total=2)
