import pytest
from semantic_transmission.motion_sampling import distributed_pairs, normalized_pair_score


@pytest.mark.parametrize('frames', range(1,130))
def test_pairs_are_distinct_and_cover_both_ends(frames):
    pairs=distributed_pairs(frames,24)
    assert pairs==sorted(set(pairs))
    assert len(pairs)<=3
    assert all(0<=a<b<frames and b-a<=10 for a,b in pairs)
    if frames>1:
        assert pairs[0][0]==0 and pairs[-1][1]==frames-1


def test_nominal_long_window_uses_short_local_displacements():
    assert distributed_pairs(240,24)==[(0,10),(114,124),(229,239)]


@pytest.mark.parametrize('fps',[12,24,30,60])
def test_velocity_scale_invariance(fps):
    for n in [2,8,120,600]:
        pairs=distributed_pairs(n,fps)
        d=[6*(b-a)/fps for a,b in pairs]
        assert normalized_pair_score(d,pairs,fps)==pytest.approx(2.5)
    assert normalized_pair_score([],[],fps) is None


def test_reject_duplicate_or_zero_time_measurements():
    for p in [[(0,0)],[(0,1),(0,1)]]:
        with pytest.raises(ValueError):normalized_pair_score([0]*len(p),p,24)
