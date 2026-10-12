"""Correspondence gate preventing RX-key overlays at incompatible generated edges.

Added after the first seven-source pilot exposed double contours. Uses only
receiver data; this follow-up is development validation, not an unseen test.
"""
import cv2
import numpy as np

from .trajectory_receiver import PARAMETERS,backward_map,warp_high,SIZE

GUARD=dict(agreement_scale=12.,rejection_error=24.,rejection_neighborhood=3)


def refine_references(generated,keys,trajectory,progress=None,params=PARAMETERS):
    if len(keys)!=2 or keys[0].shape!=generated[0].shape or keys[1].shape!=generated[0].shape:
        raise ValueError('two matching received endpoint images required')
    if trajectory is None:
        return generated.copy(),dict(bypassed=True,reason='no paid trajectory',mean_reference_weight=0.)
    if trajectory['frames']!=len(generated):raise ValueError('trajectory timeline mismatch')
    output=generated.copy();weights=[];rejected=[]
    for t in range(len(generated)):
        candidates=[];confidence=[]
        for endpoint in (0,1):
            origin=0 if endpoint==0 else len(generated)-1
            keep=(trajectory['origins']==endpoint)&trajectory['valid'][t]&trajectory['valid'][origin]
            coordinates,valid=backward_map(trajectory['xy'][t,keep],trajectory['xy'][origin,keep],params)
            candidates.append(warp_high(keys[endpoint],coordinates).astype(np.float32));confidence.append(valid)
        a,b=candidates
        error=np.mean(np.abs(cv2.resize(a,SIZE)-cv2.resize(b,SIZE)),axis=2)
        alpha=params['reference_strength']*np.minimum(*confidence)*np.exp(-(error/params['photometric_scale'])**2)
        alpha=cv2.GaussianBlur(alpha,(0,0),2)
        u=t/(len(generated)-1);bridge=(1-u)*a+u*b
        mismatch=np.mean(np.abs(cv2.resize(bridge,SIZE)-cv2.resize(generated[t].astype(np.float32),SIZE)),axis=2)
        local_error=cv2.GaussianBlur(mismatch,(0,0),1)
        worst=cv2.dilate(mismatch,np.ones((GUARD['rejection_neighborhood'],)*2,np.uint8))
        # Reject incompatible silhouettes before mixing; smoothing cannot leak
        # confidence back into a rejected pixel neighborhood.
        gate=np.exp(-(local_error/GUARD['agreement_scale'])**2)*(worst<GUARD['rejection_error'])
        alpha*=gate*min(1.,t/params['endpoint_ramp'],(len(generated)-1-t)/params['endpoint_ramp'])
        alpha=cv2.resize(alpha,(generated.shape[2],generated.shape[1]))[...,None]
        output[t]=np.rint(generated[t]*(1-alpha)+bridge*alpha).clip(0,255).astype(np.uint8)
        weights.append(float(alpha.mean()));rejected.append(float((worst>=GUARD['rejection_error']).mean()))
        if progress is not None:progress(t+1,len(generated))
    return output,dict(bypassed=False,guard=GUARD,mean_reference_weight=float(np.mean(weights)),
                       mean_rejected_area=float(np.mean(rejected)),per_frame_weight=weights)
