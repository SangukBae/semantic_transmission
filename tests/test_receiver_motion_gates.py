import cv2
import numpy as np
import pytest
from semantic_transmission.receiver_motion_gate import select_window
from semantic_transmission.receiver_motion_retention import gate_segment


def texture():
    rng=np.random.default_rng(172)
    return cv2.GaussianBlur(rng.integers(0,256,(160,288,3),dtype=np.uint8),(5,5),0)


def shift(image,x):
    return cv2.warpAffine(image,np.float32([[1,0,x],[0,1,0]]),(288,160),borderMode=cv2.BORDER_REFLECT_101)


def test_received_tracks_accept_correct_motion_and_reject_opposite_motion():
    image=texture();points=np.float32([[80,45],[120,60],[160,85],[200,110]])
    trajectory=dict(frames=5,origins=np.zeros(4,int),valid=np.ones((5,4),bool),
        xy=np.stack([points+np.float32([t,0]) for t in range(5)]))
    static=np.repeat(image[None],5,axis=0)
    correct=np.stack([shift(image,t) for t in range(5)])
    opposite=np.stack([shift(image,-t) for t in range(5)])
    selected,audit=select_window(static,correct,trajectory,range(1,5),0)
    assert selected and audit['received_track_error'][1]<audit['received_track_error'][0]
    assert not select_window(correct,opposite,trajectory,range(1,5),0)[0]
    assert not select_window(static,correct,None,range(1,5),0)[0]


def test_retention_gate_recovers_excursion_without_erasing_other_motion():
    image=texture();old=[];new=[]
    for t in range(17):
        base=shift(image,t);base[:,:144]=image[:,:144];old.append(base)
        candidate=base.copy()
        if t<=8:candidate[:,:144]=shift(image,8*np.sin(t*np.pi/8))[:,:144]
        new.append(candidate)
    old=np.stack(old);new=np.stack(new)
    result,audit=gate_segment(old,new,[dict(radius=8),dict(radius=8)])
    assert audit['endpoints'][0]['selected']
    assert not audit['endpoints'][1]['selected']
    assert np.array_equal(result[:9],new[:9]) and np.array_equal(result[9:],old[9:])
    # Reversing the roles would erase the recovered excursion and must be rejected.
    reverse,audit=gate_segment(new,old,[dict(radius=8),dict(radius=8)])
    assert not any(r['selected'] for r in audit['endpoints'])
    assert np.array_equal(reverse,new)


def test_retention_gate_rejects_key_or_far_boundary_mismatch():
    old=np.repeat(texture()[None],17,axis=0);new=old.copy();new[0]=0
    with pytest.raises(ValueError,match='keys differ'):gate_segment(old,new,[dict(radius=8),dict(radius=8)])
    new=old.copy();new[8]=0
    with pytest.raises(ValueError,match='far endpoint differs'):gate_segment(old,new,[dict(radius=8),dict(radius=8)])
