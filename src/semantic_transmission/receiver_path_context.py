"""Endpoint interpolation with paths measured from all received-side frames.

Only endpoint-to-interior displacement changes versus receiver_key_context.
Endpoint RGB, radii, blending weights and cycle confidence remain unchanged.
No RGB residual restoration, region gate, source pixels, or learned tracker.
"""
import time
import cv2
import numpy as np
from .receiver_key_context import PARAMETERS as CONTEXT, flows, radius
from .trajectory_receiver import grid, remap, warp_high, in_bounds

PARAMETERS = dict(tracker='adjacent_bidirectional_DIS_MEDIUM',
                  path_correction='subtract_linear_endpoint_displacement',
                  path_strength=1.,inverse_iterations=4)


def track_paths(frames):
    """Compose adjacent flow along trajectories originating at either endpoint."""
    if len(frames)<2:raise ValueError('at least two frames required')
    start=time.perf_counter()
    pairs=[flows(a,b) for a,b in zip(frames,frames[1:])]
    flow_seconds=time.perf_counter()-start
    g=grid(*pairs[0][0].shape[:2]);left=[np.zeros_like(g)];position=g.copy()
    for forward,_ in pairs:
        position=position+remap(forward,position);left.append(position-g)
    right=[np.zeros_like(g)];position=g.copy()
    for _,backward in reversed(pairs):
        position=position+remap(backward,position);right.append(position-g)
    return np.stack(left),np.stack(right[::-1]),dict(
        flow_seconds=flow_seconds,tracking_seconds=time.perf_counter()-start,
        adjacent_pairs=len(pairs),directed_flow_fields=2*len(pairs))


def interpolate_tracked(frames,received,params=CONTEXT):
    """Add nonlinear path deviations to the original key/end displacement.

    For translation s(t), this keeps s(t) plus a linear endpoint correction.
    Acceleration of the generated path is preserved instead of flattened.
    This is a geometric path residual, not an RGB difference added after synthesis.
    """
    r=len(frames)-1
    if r<1:raise ValueError('positive window required')
    left,right,timing=track_paths(frames)
    forward,backward=flows(received,frames[-1]);g=grid(*forward.shape[:2])
    align=np.zeros_like(g) if np.array_equal(received,frames[0]) else flows(received,frames[0])[0]
    key_to_generated=g+align
    ca=np.exp(-np.linalg.norm(forward+remap(backward,g+forward),axis=2)/params['cycle_scale'])
    cb=np.exp(-np.linalg.norm(backward+remap(forward,g+backward),axis=2)/params['cycle_scale'])
    output=[received.copy()];deviations=[]
    for t in range(1,r):
        u=t/r
        dl=remap(left[t]-u*left[-1],key_to_generated)
        dr=right[t]-(1-u)*right[0]
        path_a=u*forward+dl;path_b=(1-u)*backward+dr
        ma,mb=g.copy(),g.copy()
        for _ in range(PARAMETERS['inverse_iterations']):
            ma=g-remap(path_a,ma);mb=g-remap(path_b,mb)
        a=warp_high(received,ma).astype(np.float32)
        b=warp_high(frames[-1],mb).astype(np.float32)
        wa=(1-u)*(params['confidence_floor']+remap(ca,ma)*in_bounds(ma))
        wb=u*(params['confidence_floor']+remap(cb,mb)*in_bounds(mb))
        alpha=cv2.resize(wa/(wa+wb),(received.shape[1],received.shape[0]))[...,None]
        output.append(np.rint(alpha*a+(1-alpha)*b).clip(0,255).astype(np.uint8))
        deviations.append(dict(offset=t,left_mean_pixels=float(np.linalg.norm(dl,axis=2).mean()),
                               right_mean_pixels=float(np.linalg.norm(dr,axis=2).mean())))
    output.append(frames[-1].copy())
    return np.stack(output),dict(**timing,path_deviations=deviations)


def preserve_path_context(frames,keys,progress=None):
    if frames.dtype!=np.uint8 or len(frames)<2 or len(keys)!=2:
        raise ValueError('uint8 segment and two received keys required')
    if any(k.shape!=frames[0].shape or k.dtype!=np.uint8 for k in keys):
        raise ValueError('matching received RGB required')
    output=frames.copy();audits=[];start=time.perf_counter()
    for endpoint in (0,1):
        sequence=frames if endpoint==0 else frames[::-1]
        r,decision=radius(sequence,keys[endpoint])
        if r:
            window,audit=interpolate_tracked(sequence[:r+1],keys[endpoint])
            if endpoint==0:output[:r+1]=window
            else:output[-r-1:]=window[::-1]
        else:audit=dict(bypassed=True,tracking_seconds=0.,flow_seconds=0.,adjacent_pairs=0,directed_flow_fields=0)
        output[0 if endpoint==0 else -1]=keys[endpoint]
        audits.append(dict(endpoint=endpoint,**decision,**audit))
        if progress:progress(endpoint+1,2)
    return output,dict(policy='received_key_framewise_path_v1',context_parameters=CONTEXT,parameters=PARAMETERS,
        endpoints=audits,algorithm_seconds=time.perf_counter()-start,
        tracking_seconds=sum(a['tracking_seconds'] for a in audits),
        received_keys_exact=True,receiver_reads_source=False,
        limitations='Generated paths can be wrong; occlusions and composed-flow drift are not solved by this ablation.')
