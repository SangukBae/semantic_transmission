import pytest

from semantic_transmission.structure_anchors import add_video, video_payload, anchor_slots
from semantic_transmission.wire import pack, unpack


def packet():
    return pack({'video': {'frames': 384}, 'keyframes': [{'rate_offset': 0, 'rate_bytes': 2}]}, b'ab')


def test_structure_payload_preserves_existing_rates_and_accounts_all_bytes():
    value = add_video(packet(), b'bitstream', 72, 40, list(range(0,384,8)), 384)
    item, data = video_payload(value)
    assert data == b'bitstream' and item['offset'] == 2
    assert unpack(value)[1][:2] == b'ab'


def test_reject_overlapping_structure_payload():
    value = add_video(packet(), b'bitstream', 72, 40, [0,8], 384)
    header, body = unpack(value)
    header['structure_video_v1']['offset'] = 1
    with pytest.raises(ValueError):
        video_payload(pack(header, body))


@pytest.mark.parametrize('times', [[8,0], [0,0], [0,384]])
def test_reject_invalid_source_timestamps(times):
    with pytest.raises(ValueError):
        add_video(packet(), b'x', 72, 40, times, 384)


def test_actual_17_frame_timeline_and_protected_conditions():
    plan = anchor_slots(384, list(range(0,384,8)), [0,110])
    assert len(plan['latent_representative_frames']) == 113
    assert plan['latent_representative_frames'][:6] == [0,4,8,12,16,17]
    assert plan['latent_representative_frames'][-1] == 383
    assert not set(plan['slots']) & {0,110}
    assert all(0 <= s < 113 for s in plan['slots'])


def test_h264_low_fps_preserves_all_timestamped_frames(tmp_path):
    import numpy as np
    from semantic_transmission.structure_anchors import compress, decode_video, expand_video
    frames = np.random.default_rng(7).integers(0,256,(48,40,72,3),dtype=np.uint8)
    bits = compress(frames,tmp_path/'video.h264',3,32)
    item = dict(width=72,height=40,timestamps=list(range(0,384,8)),frames=384)
    decoded = decode_video(bits,item)
    assert decoded.shape == frames.shape
    expanded = expand_video(decoded,item,40,72)
    assert np.array_equal(expanded[::8],decoded)
    assert np.array_equal(expanded[-1],decoded[-1])
