import cv2
import numpy as np
import pytest

from semantic_transmission.receiver_key_context import preserve_context
from semantic_transmission.receiver_motion_context import preserve_motion_context, transfer_residual


def moving_pattern(n=33):
    rng=np.random.default_rng(504)
    texture=cv2.GaussianBlur(rng.integers(0,256,(80,144,3),dtype=np.uint8),(7,7),0)
    # An excursion invisible in either endpoint: linear endpoint interpolation erases it.
    return np.stack([cv2.warpAffine(texture,np.float32([[1,0,5*np.sin(t*np.pi/8)],[0,1,0]]),
        (144,80),borderMode=cv2.BORDER_REFLECT_101) for t in range(n)])


def test_recovers_motion_when_received_endpoints_already_match():
    frames=moving_pattern();keys=[frames[0],frames[-1]]
    old,_=preserve_context(frames,keys)
    new,_=preserve_motion_context(frames,keys)
    assert np.abs(old.astype(float)-frames).mean()>1
    assert np.max(np.abs(new.astype(int)-frames.astype(int)))<=1


@pytest.mark.parametrize('n',[2,3,9,13,65])
def test_exact_keys_bounded_support_and_no_input_mutation(n):
    frames=moving_pattern(n);before=frames.copy();keys=[frames[0]//2,frames[-1]//2]
    result,_=preserve_motion_context(frames,keys)
    assert result.dtype==np.uint8 and result.shape==frames.shape
    assert np.array_equal(result[0],keys[0]) and np.array_equal(result[-1],keys[-1])
    assert np.array_equal(frames,before)
    if n>50:assert np.array_equal(result[25:-25],frames[25:-25])


def test_zero_strength_reproduces_frozen_context_policy():
    frames=moving_pattern();keys=[frames[0]//2,frames[-1]//2]
    old,_=preserve_context(frames,keys);new,_=preserve_motion_context(frames,keys,strength=0)
    assert np.array_equal(old,new)


def test_no_residual_does_not_change_corrected_image():
    a,b=moving_pattern(2)
    out,_=transfer_residual(a,b,b)
    assert np.array_equal(out,a)


def test_rejects_invalid_strength():
    frames=moving_pattern(3)
    with pytest.raises(ValueError):preserve_motion_context(frames,[frames[0],frames[-1]],strength=2)
