import inspect

import cv2
import numpy as np
import pytest

from semantic_transmission import adaptive_receiver as adaptive


def profile(count=16, **values):
    return [dict(speed=.5, acceleration=.05, confidence=.95, cut=False, **values) for _ in range(count)]


def test_schedule_preserves_timeline_and_shortens_rapid_motion():
    calm = profile()
    fast = [dict(row, speed=6.) for row in calm]
    assert adaptive.schedule(calm) == [(0,8),(8,16)]
    assert all(b-a <= 2 for a,b in adaptive.schedule(fast))
    low_confidence = [dict(row, confidence=.1) for row in calm]
    assert all(b-a == 1 for a,b in adaptive.schedule(low_confidence))


def test_no_interpolation_crosses_a_scene_cut():
    rows = profile(); rows[5]['cut'] = True
    windows = adaptive.schedule(rows)
    assert (5,6) in windows
    assert not any(a <= 5 and b > 6 for a,b in windows)


def test_received_trajectory_bend_limits_interpolation_span():
    xy = np.zeros((17,3,2),np.float32)
    xy[4,:,0]=12
    tracks = dict(xy=xy,valid=np.ones((17,3),bool))
    assert (0,8) not in adaptive.schedule(profile(),tracks)


def test_one_visible_endpoint_can_contribute_without_the_other():
    shape=(160,288)
    a=np.full((*shape,3),80,np.uint8);b=np.full_like(a,220)
    weights,strength=adaptive.endpoint_weights(a,b,a,[np.ones(shape),np.zeros(shape)],.5)
    assert np.allclose(weights[...,0],1) and np.allclose(weights[...,1],0)
    assert strength.mean() > .5


def test_uncertain_disagreeing_keys_do_not_receive_full_strength():
    shape=(160,288);a=np.full((*shape,3),20,np.uint8);b=np.full_like(a,220)
    weights,strength=adaptive.endpoint_weights(a,b,np.full_like(a,120),[np.ones(shape)]*2,.5)
    assert strength.max() < .01


def test_no_visibility_means_no_key_overlay():
    shape=(160,288);a=np.full((*shape,3),80,np.uint8)
    weights,strength=adaptive.endpoint_weights(a,a,a,[np.zeros(shape)]*2,.5)
    assert np.all(strength==0) and np.all(weights==0)


def texture_frames():
    rng=np.random.default_rng(3)
    texture=rng.integers(0,255,(32,48,3),dtype=np.uint8)
    return np.stack([np.roll(texture,t,axis=1) for t in range(9)])


def test_cadence_preserves_endpoints_shape_and_is_deterministic():
    cv2.setNumThreads(1);frames=texture_frames()
    a,audit=adaptive.refine_temporal(frames)
    b,_=adaptive.refine_temporal(frames)
    assert np.array_equal(a,b) and a.shape==frames.shape
    assert np.array_equal(a[[0,-1]],frames[[0,-1]])
    assert audit['windows'][0][0]==0 and audit['windows'][-1][-1]==8


def test_absent_tracks_preserve_original_and_require_no_sender_inputs():
    frames=texture_frames()
    out,audit=adaptive.refine_references(frames,[frames[0],frames[-1]],None)
    assert np.array_equal(out,frames) and audit['bypassed']
    assert 'source' not in inspect.signature(adaptive.refine_references).parameters
    assert 'source' not in inspect.signature(adaptive.refine_temporal).parameters


def test_active_visible_key_synthesis_preserves_both_endpoints():
    frames=np.full((9,32,48,3),60,np.uint8)
    keys=[np.full((32,48,3),90,np.uint8),np.full((32,48,3),200,np.uint8)]
    positions=np.array([[50,40],[220,40],[60,130],[210,130]],np.float32)
    tracks=dict(frames=9,xy=np.repeat(positions[None],9,axis=0),
                valid=np.ones((9,4),bool),origins=np.zeros(4,int))
    out,audit=adaptive.refine_references(frames,keys,tracks)
    assert out[4].mean()>frames[4].mean()
    assert np.array_equal(out[[0,-1]],frames[[0,-1]])
    assert audit['mean_single_endpoint_area']>0


def test_mismatched_trajectory_timeline_is_rejected():
    with pytest.raises(ValueError,match='timeline'):
        adaptive.refine_temporal(texture_frames(),dict(frames=10))
