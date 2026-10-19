import numpy as np
import pytest

from semantic_transmission import sparc_selection as s

CONFIG = dict(max_gap=4, proxy_mae_threshold=6.0)


def test_static_sequence_uses_fewer_anchors_and_keeps_semantics():
    proxies = np.zeros((25, 4, 4, 3), np.float32)
    result = s.plan(proxies, [0, 5, 24], [], CONFIG)
    assert {0, 5, 24}.issubset(result['indices'])
    assert len(result['indices']) < len(range(0, 25, 2))
    assert max(np.diff(result['indices'])) <= 4


def test_motion_excursion_requires_extra_anchor_even_if_endpoints_match():
    proxies = np.zeros((5, 4, 4, 3), np.float32)
    proxies[2] = 80
    result = s.plan(proxies, [0, 4], [], CONFIG)
    assert 2 in result['indices']
    assert any(r['reason'] == 'source_motion_proxy' for r in result['additions'])


def test_cut_sides_are_paid_and_only_scene_starts_reset_prediction():
    proxies = np.zeros((12, 4, 4, 3), np.float32)
    result = s.plan(proxies, [0, 3, 8, 11], [6], CONFIG)
    assert {5, 6}.issubset(result['indices'])
    groups = s.scene_groups(result['indices'], [6])
    assert len(groups) == 2 and groups[1][0] == 6
    assert 3 in groups[0] and 8 in groups[1]


@pytest.mark.parametrize('frames', [2, 3, 4, 5, 17, 240, 241])
def test_uniform_control_matches_count_with_legal_gaps(frames):
    proxies = np.zeros((frames, 2, 2, 3), np.float32)
    keys = sorted({0, frames//3, frames-1})
    cuts = [frames//2] if frames > 4 else []
    adaptive = s.plan(proxies, keys, cuts, CONFIG)
    uniform = s.plan(proxies, keys, cuts, CONFIG, 'uniform-matched')
    assert len(adaptive['indices']) == len(uniform['indices'])
    s.validate_anchors(uniform['indices'], frames, 4, cuts)


def test_maximum_gap_and_all_frames_uniform_controls():
    for count in range(61, 241):
        values = s.uniform_matched(240, count, [], 4)
        assert len(values) == count and max(np.diff(values)) <= 4


@pytest.mark.parametrize('indices,cuts', [([0, 8], []), ([0, 4, 8], [4]),
    ([0, 4, 4, 8], []), ([0, True, 8], []), ([0, 4, 8], [9])])
def test_invalid_timelines_rejected(indices, cuts):
    with pytest.raises(ValueError): s.validate_anchors(indices, 9, 4, cuts)


def test_source_proxy_is_deterministic_and_never_drops_keys():
    rng = np.random.default_rng(42)
    for frames in (7, 31, 80):
        pixels = rng.uniform(0, 255, (frames, 3, 3, 3)).astype(np.float32)
        keys = sorted({0, frames//3, frames-1})
        a = s.plan(pixels, keys, [], CONFIG)
        assert a == s.plan(pixels, keys, [], CONFIG)
        assert set(keys).issubset(a['indices'])
        assert not a['receiver_quality_used']
