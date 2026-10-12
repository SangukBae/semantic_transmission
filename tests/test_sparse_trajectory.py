import numpy as np
import pytest

from semantic_transmission.sparse_trajectory import encode,decode,adaptive_knots,fit_packet,FIELD
from semantic_transmission.trajectory_receiver import warp_high,grid,refine_temporal,refine_references
from semantic_transmission.stutter_metrics import evaluate
from semantic_transmission.wire import pack,unpack


def texture(frames=17, stalled=False):
    import cv2
    rng=np.random.default_rng(123)
    image=rng.integers(0,256,(160,288,3),dtype=np.uint8)
    image=cv2.GaussianBlur(image,(5,5),1)
    return np.stack([cv2.warpAffine(image,np.float32([[1,0,(i//4)*4 if stalled else i],[0,1,0]]),(288,160),borderMode=cv2.BORDER_REFLECT_101) for i in range(frames)])


def test_trajectory_roundtrip_visibility_and_quarter_pixel():
    xy=np.array([[[10.13,20.2]],[[20.13,20.2]],[[30.13,20.2]]])
    encoded=encode(xy,[[True],[False],[False]],[0],[0,4,8],9)
    decoded=decode(encoded,9)
    assert np.max(np.abs(decoded['xy'][[0,4,8]]-xy))<=.125
    assert decoded['valid'][0,0] and not decoded['valid'][1:,0].any()
    with pytest.raises(ValueError):decode(encoded,10)
    with pytest.raises(Exception):decode(encoded[:-4])


def test_adaptive_knots_follow_direction_change():
    xy=np.zeros((21,1,2));xy[:,0,0]=np.r_[np.arange(11),np.arange(9,-1,-1)]
    times=adaptive_knots(xy,np.ones((21,1),bool),3)
    assert times.tolist()==[0,10,20]


def test_budget_bypass_preserves_packet():
    frames=texture(9)
    packet=pack({'video':{'frames':9}})
    out,audit=fit_packet(packet,frames,len(packet))
    assert out==packet and audit['bypass']
    out,audit=fit_packet(packet,frames,len(packet)+768)
    assert len(out)<=len(packet)+768 and FIELD in unpack(out)[0]


def test_identity_warp_is_exact():
    image=np.random.default_rng(3).integers(0,256,(320,576,3),dtype=np.uint8)
    assert np.array_equal(warp_high(image,grid(160,288)),image)


def test_static_temporal_and_unpaid_reference_bypass():
    frames=np.repeat(texture(3)[:1],7,axis=0)
    out,_=refine_temporal(frames)
    assert np.array_equal(out,frames)
    out,audit=refine_references(frames,[frames[0],frames[-1]],None)
    assert np.array_equal(out,frames) and audit['bypassed']


def test_cut_does_not_blend_black_and_white():
    frames=np.zeros((9,160,288,3),np.uint8);frames[4:]=255
    output,_=refine_temporal(frames)
    assert np.array_equal(output,frames)


def test_temporal_filter_reduces_synthetic_cadence_without_freezing():
    source=texture();stalled=texture(stalled=True)
    candidate,_=refine_temporal(stalled)
    before=evaluate(source,stalled);after=evaluate(source,candidate)
    assert after['cadence_error_mean']<before['cadence_error_mean']
    assert after['false_freeze_rate']<before['false_freeze_rate']
    assert np.array_equal(candidate[[0,-1]],stalled[[0,-1]])


def test_stutter_metric_penalizes_frozen_output():
    source=texture();frozen=np.repeat(source[:1],len(source),axis=0)
    exact=evaluate(source,source);bad=evaluate(source,frozen)
    assert exact['flow_error_mean']==0 and exact['false_freeze_rate']==0
    assert bad['false_freeze_rate']>.9


def test_received_reference_restores_static_texture_and_keeps_endpoints():
    import cv2
    key=texture(3)[0]
    blurry=cv2.GaussianBlur(key,(15,15),4)
    frames=np.repeat(blurry[None],9,axis=0)
    points=np.array([[40,40],[145,40],[245,40],[40,120],[145,120],[245,120]],np.float32)
    points=np.concatenate([points,points])
    xy=np.repeat(points[None],9,axis=0)
    trajectory=dict(frames=9,xy=xy,valid=np.ones((9,12),bool),origins=np.repeat([0,1],6))
    output,audit=refine_references(frames,[key,key],trajectory)
    assert np.mean((output[4].astype(float)-key)**2)<np.mean((frames[4].astype(float)-key)**2)
    assert np.array_equal(output[[0,-1]],frames[[0,-1]])
    assert audit['mean_reference_weight']>0
