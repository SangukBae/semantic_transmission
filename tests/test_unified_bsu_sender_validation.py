import pytest

from semantic_transmission import unified_bsu_core as core
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.unified_bsu_sender_validation import verified_infeasible
from semantic_transmission.wire import pack


def failed_grid(root):
    context = dict(video=dict(frames=17, height=64, width=96, fps=24), indices=[0, 16],
                   segments=[dict(path='clips/sample/0_16.mp4', text='A person walks.', flow=1.)], decoder={})
    arm = dict(budget=1, profile=dict(scales=[1], strides=[2], qps=[16, 24]))
    rows = []; (root / 'sender').mkdir(); (root / 'logs').mkdir()
    for qp in arm['profile']['qps']:
        coding = dict(indices=list(range(0, 17, 2)), qp=qp, stride=2, scale=1)
        stream = pack(dict(indices=coding['indices'], height=64, width=96), b'video-data')
        data = core.encode_packet(context, stream, coding, dict(policy=core.POLICY, profile={}))
        name = f'scale1_stride2_qp{qp}'; path = root / 'sender' / (name+'.bin'); path.write_bytes(data)
        rows.append(dict(name=name, packet_sha256=sha256(path), packet_bytes=len(data),
                         channel_uses=core.channel_uses(data), fits=False))
    write_json(root / 'sender/candidates.json', dict(rows=rows, budget=1, complete=False))
    write_json(root / 'stages/sender.json', dict(status='FAILED'))
    (root / 'logs/sender.log').write_text('ValueError: no candidate fits the complete paid packet budget')
    return arm


def test_complete_measured_no_fit_grid_is_reported_without_fabricated_quality(tmp_path):
    arm = failed_grid(tmp_path); row = verified_infeasible(tmp_path, arm)
    assert row['status'] == 'INFEASIBLE_REGISTERED_BUDGET'
    assert row['tested_candidates'] == 2 and row['minimum_candidate_channel_uses'] > arm['budget']
    assert row['movie'] is None and not row['actual_channel_run']


@pytest.mark.parametrize('failure', ['partial_grid', 'changed_packet', 'other_error'])
def test_unrelated_or_incomplete_failures_are_not_hidden(tmp_path, failure):
    arm = failed_grid(tmp_path)
    if failure == 'partial_grid':
        path = tmp_path / 'sender/candidates.json'; data = core.read(path); data['rows'].pop(); write_json(path, data)
    elif failure == 'changed_packet':
        path = tmp_path / 'sender/scale1_stride2_qp16.bin'; path.write_bytes(path.read_bytes()+b'bad')
    else:
        (tmp_path / 'logs/sender.log').write_text('CUDA out of memory')
    assert verified_infeasible(tmp_path, arm) is None
