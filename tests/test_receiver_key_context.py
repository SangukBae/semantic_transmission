import numpy as np
import cv2
import pytest

from semantic_transmission.receiver_key_context import preserve_context,interpolate_pair,radius,PARAMETERS


def pattern():
    rng=np.random.default_rng(81)
    return cv2.GaussianBlur(rng.integers(0,256,(80,144,3),dtype=np.uint8),(7,7),0)


def test_static_input_is_unchanged_and_not_mutated():
    key=pattern();frames=np.repeat(key[None],65,axis=0);before=frames.copy()
    result,audit=preserve_context(frames,[key,key])
    assert np.array_equal(result,frames) and np.array_equal(frames,before)
    assert audit['receiver_reads_source'] is False


@pytest.mark.parametrize('n',[2,3,9,13,65])
def test_exact_keys_short_segments_nonoverlap_and_bounded_support(n):
    key=pattern();frames=np.repeat(key[None],n,axis=0)
    keys=[np.clip(key.astype(int)+15,0,255).astype(np.uint8),np.clip(key.astype(int)-15,0,255).astype(np.uint8)]
    out,audit=preserve_context(frames,keys)
    assert out.shape==frames.shape and out.dtype==np.uint8
    assert np.array_equal(out[0],keys[0]) and np.array_equal(out[-1],keys[1])
    if n>50:assert np.array_equal(out[25:-25],frames[25:-25])
    assert not np.array_equal(frames[0],keys[0])


def test_detected_cut_stops_propagation():
    frames=np.zeros((31,80,144,3),np.uint8);frames[5:]=255
    r,audit=radius(frames,frames[0])
    assert r==4 and audit['detected_cut_transition']==5


def test_translation_interpolates_geometry_and_exact_endpoints():
    a=pattern();b=cv2.warpAffine(a,np.float32([[1,0,8],[0,1,0]]),(144,80),borderMode=cv2.BORDER_REFLECT_101)
    out,_=interpolate_pair(a,b,8)
    expected=cv2.warpAffine(a,np.float32([[1,0,4],[0,1,0]]),(144,80),borderMode=cv2.BORDER_REFLECT_101)
    assert np.array_equal(out[0],a) and np.array_equal(out[-1],b)
    assert np.abs(out[4,10:-10,15:-15].astype(float)-expected[10:-10,15:-15]).mean()<5


def test_reject_mismatched_keys():
    with pytest.raises(ValueError):preserve_context(np.zeros((3,80,144,3),np.uint8),[pattern(),np.zeros((2,2,3),np.uint8)])
