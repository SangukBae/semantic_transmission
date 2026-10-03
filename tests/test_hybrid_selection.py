import math

import pytest

from semantic_transmission.hybrid_selection import cached_score, choose_keys


def test_required_events_and_endpoints_never_call_skem():
    def forbidden(*args):
        raise AssertionError("required event must not be vetoed")
    keys, _ = choose_keys(50, [0, 8, 9, 49], [8, 9], forbidden, max_gap=24)
    assert keys == [0, 8, 9, 33, 49]


def test_time_anchor_updates_reference_before_next_optional_score():
    seen = []
    def reject(a, b):
        seen.append((a, b))
        return dict(p_yes=0.9, p_no=0.1)
    keys, _ = choose_keys(66, [0, 10, 26, 39, 65], [], reject)
    assert seen == [(0, 10), (24, 26), (24, 39)]
    assert keys == [0, 24, 48, 65]


def test_acceptance_resets_deadline_and_reference_but_rejection_does_not():
    seen = []
    def score(a, b):
        seen.append((a, b))
        return dict(p_yes=0.1, p_no=0.8) if b == 10 else dict(p_yes=0.7, p_no=0.2)
    keys, _ = choose_keys(55, [0, 10, 20, 40, 54], [], score)
    assert seen == [(0, 10), (10, 20), (34, 40)]
    assert keys == [0, 10, 34, 54]


def test_exact_deadline_bypasses_skem_and_keeps_mandatory_label():
    keys, records = choose_keys(49, [0, 24, 48], [24], lambda *_: pytest.fail("unexpected scoring"))
    assert keys == [0, 24, 48]
    assert records[1]["reason"] == "mandatory_and_max_gap"


def test_strict_threshold_and_invalid_probabilities():
    keys, _ = choose_keys(5, [0, 2, 4], [], lambda *_: dict(p_yes=0.0, p_no=0.35))
    assert keys == [0, 4]
    for score in [dict(p_yes=math.nan, p_no=0.8), dict(p_yes=0.7, p_no=0.8)]:
        with pytest.raises(ValueError, match="probabilities"):
            choose_keys(5, [0, 2, 4], [], lambda *_: score)


def test_cache_binds_exact_reference_and_protocol(tmp_path):
    seen = []
    def scorer(a, b):
        seen.append((a,b))
        return dict(p_yes=0.2, p_no=0.7)
    cached_score(tmp_path, {"signature":"a"}, 0, 8, scorer)
    cached_score(tmp_path, {"signature":"a"}, 0, 8, scorer)
    cached_score(tmp_path, {"signature":"a"}, 4, 8, scorer)
    assert seen == [(0,8),(4,8)]
    with pytest.raises(ValueError, match="cache changed"):
        cached_score(tmp_path, {"signature":"b"}, 0, 8, scorer)


def test_invalid_timeline_rejected():
    with pytest.raises(ValueError, match="timeline"):
        choose_keys(8, [0,4,7], [6], lambda *_: {})
