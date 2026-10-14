"""Experimental RX-only, bounded motion interpolation around received keys.

Unlike warping the received key into the old generated geometry at every time,
this operator makes the key an endpoint of a short temporal interpolation.
It can erase real acceleration or hallucinate correspondence; evaluate motion,
non-key fidelity, and boundary errors before adopting it.
"""
import cv2
import numpy as np

from .sparse_trajectory import gray
from .trajectory_receiver import grid, remap, warp_high, in_bounds

PARAMETERS = dict(min_radius=8, max_radius=24, flow_quantile=95.,
                  correction_speed_floor=1., displacement_factor=2.,
                  scene_cut_mad=45., cycle_scale=1.5, confidence_floor=.02)


def flows(left, right):
    engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return engine.calc(gray(left), gray(right), None), engine.calc(gray(right), gray(left), None)


def interpolate_pair(left, right, length, params=PARAMETERS):
    """Bidirectional flow interpolation with visibility weights, exact ends."""
    if length < 1:
        raise ValueError('positive interval required')
    forward, backward = flows(left, right)
    g = grid(*forward.shape[:2])
    ca = np.exp(-np.linalg.norm(forward+remap(backward,g+forward),axis=2)/params['cycle_scale'])
    cb = np.exp(-np.linalg.norm(backward+remap(forward,g+backward),axis=2)/params['cycle_scale'])
    output = [left.copy()]; confidence = []
    for t in range(1,length):
        u=t/length; ma,mb=g.copy(),g.copy()
        for _ in range(4):
            ma=g-u*remap(forward,ma); mb=g-(1-u)*remap(backward,mb)
        a,b=warp_high(left,ma).astype(np.float32),warp_high(right,mb).astype(np.float32)
        wa=(1-u)*(params['confidence_floor']+remap(ca,ma)*in_bounds(ma))
        wb=u*(params['confidence_floor']+remap(cb,mb)*in_bounds(mb))
        alpha=cv2.resize(wa/(wa+wb),(left.shape[1],left.shape[0]))[...,None]
        output.append(np.rint(alpha*a+(1-alpha)*b).clip(0,255).astype(np.uint8))
        confidence.append(float(np.minimum(remap(ca,ma),remap(cb,mb)).mean()))
    output.append(right.copy())
    return np.stack(output), dict(mean_cycle_confidence=float(np.mean(confidence)) if confidence else 1.)


def radius(frames, received, params=PARAMETERS):
    cap=min(params['max_radius'],(len(frames)-1)//2)
    if not cap:
        return 0,dict(reason='no interior support')
    mismatch,_=flows(received,frames[0])
    motion,_=flows(frames[0],frames[1])
    distance=float(np.percentile(np.linalg.norm(mismatch,axis=2),params['flow_quantile']))
    speed=float(np.percentile(np.linalg.norm(motion,axis=2),params['flow_quantile']))
    wanted=max(params['min_radius'],int(np.ceil(params['displacement_factor']*distance/
                                               (speed+params['correction_speed_floor']))))
    actual=min(cap,wanted);cut=None
    for t in range(1,actual+1):
        error=float(np.abs(gray(frames[t]).astype(float)-gray(frames[t-1])).mean())
        if error>params['scene_cut_mad']:
            actual=t-1;cut=t;break
    return actual,dict(displacement_p95=distance,motion_p95=speed,requested_radius=wanted,
                       radius=actual,detected_cut_transition=cut)


def preserve_context(frames, keys, progress=None, params=PARAMETERS):
    if frames.dtype!=np.uint8 or len(frames)<2 or len(keys)!=2:
        raise ValueError('uint8 segment and two received keys required')
    if any(k.shape!=frames[0].shape or k.dtype!=np.uint8 for k in keys):
        raise ValueError('matching received RGB required')
    output=frames.copy();audit=[]
    for endpoint in (0,1):
        sequence=frames if endpoint==0 else frames[::-1]
        r,decision=radius(sequence,keys[endpoint],params)
        if r:
            window,alignment=interpolate_pair(keys[endpoint],sequence[r],r,params)
            if endpoint==0:output[:r+1]=window
            else:output[-r-1:]=window[::-1]
        else:alignment=dict(bypassed=True)
        output[0 if endpoint==0 else -1]=keys[endpoint]
        audit.append(dict(endpoint=endpoint,**decision,alignment=alignment))
        if progress:progress(endpoint+1,2)
    assert np.array_equal(output[0],keys[0]) and np.array_equal(output[-1],keys[1])
    return output,dict(policy='received_key_context_interpolation_v1',parameters=params,
                       endpoints=audit,received_keys_exact=True,receiver_reads_source=False)
