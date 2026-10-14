import cv2
import numpy as np
import pytest
from semantic_transmission import receiver_path_context as path
from semantic_transmission.receiver_key_context import interpolate_pair,preserve_context


def pattern():
    a=np.random.default_rng(504).integers(0,256,(80,144,3),dtype=np.uint8)
    return cv2.GaussianBlur(a,(7,7),0)


def shift(a,x):
    return cv2.warpAffine(a,np.float32([[1,0,x],[0,1,0]]),(144,80),borderMode=cv2.BORDER_REFLECT_101)


def error(a,b):return np.abs(a[:,12:-12,20:-20].astype(float)-b[:,12:-12,20:-20]).mean()


def test_actual_adjacent_tracker_recovers_an_excursion_missing_at_both_ends():
    a=pattern();frames=np.stack([shift(a,5*np.sin(t*np.pi/8)) for t in range(17)])
    before=frames.copy();old,_=interpolate_pair(frames[0],frames[-1],16)
    new,audit=path.interpolate_tracked(frames,frames[0])
    assert error(new,frames)<.35*error(old,frames)
    assert audit['adjacent_pairs']==16 and audit['tracking_seconds']>=audit['flow_seconds']>0
    assert np.array_equal(frames,before)


def test_known_translation_preserves_acceleration_with_a_mismatched_key(monkeypatch):
    # Ideal flow isolates endpoint alignment and path composition from estimation.
    a=pattern();positions=[0,2,5,9,10,8,6,7,8]
    frames=np.stack([shift(a,x) for x in positions]);key=shift(a,4)
    def flow(left,right):
        def locate(im):
            return min(range(-1,14),key=lambda x:np.abs(im[15:-15,25:-25].astype(float)-shift(a,x)[15:-15,25:-25]).mean())
        f=np.zeros((160,288,2),np.float32);f[...,0]=2*(locate(right)-locate(left));return f,-f
    monkeypatch.setattr(path,'flows',flow)
    new,_=path.interpolate_tracked(frames,key)
    expected=np.stack([shift(a,x+(1-t/8)*4) for t,x in enumerate(positions)])
    assert error(new,expected)<1.
    assert np.array_equal(new[0],key) and np.array_equal(new[-1],frames[-1])


def test_linear_paths_reduce_to_previous_interpolation(monkeypatch):
    a=pattern();frames=np.stack([shift(a,t) for t in range(9)])
    def straight(values):
        f=np.zeros((len(values),160,288,2),np.float32)
        f[...,0]=np.arange(len(values))[:,None,None]*2
        return f,f-f[-1],dict(tracking_seconds=0,flow_seconds=0,adjacent_pairs=8,directed_flow_fields=16)
    monkeypatch.setattr(path,'track_paths',straight)
    old,_=interpolate_pair(a,frames[-1],8);new,_=path.interpolate_tracked(frames,a)
    assert np.max(np.abs(old.astype(int)-new.astype(int)))<=1


@pytest.mark.parametrize('n',[2,3,9,13,65])
def test_exact_keys_short_windows_bounded_support_and_input_immutability(n):
    a=pattern();frames=np.repeat(a[None],n,axis=0);before=frames.copy()
    keys=[np.clip(a.astype(int)+15,0,255).astype(np.uint8),np.clip(a.astype(int)-15,0,255).astype(np.uint8)]
    new,audit=path.preserve_path_context(frames,keys)
    assert np.array_equal(new[0],keys[0]) and np.array_equal(new[-1],keys[1])
    assert new.shape==frames.shape and new.dtype==np.uint8 and np.array_equal(frames,before)
    if n>50:assert np.array_equal(new[25:-25],frames[25:-25])
    assert not audit['receiver_reads_source']


def test_scene_cut_support_is_identical_to_baseline():
    frames=np.zeros((31,80,144,3),np.uint8);frames[5:]=255
    keys=frames[[0,-1]];old,oa=preserve_context(frames,keys);new,na=path.preserve_path_context(frames,keys)
    assert [a['radius'] for a in oa['endpoints']]==[a['radius'] for a in na['endpoints']]
    assert np.array_equal(new,old)


def test_mismatched_keys_rejected():
    with pytest.raises(ValueError):path.preserve_path_context(np.zeros((3,80,144,3),np.uint8),[pattern(),np.zeros((2,2,3),np.uint8)])
