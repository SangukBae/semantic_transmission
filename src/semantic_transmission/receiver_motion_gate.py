"""Select motion restoration using only already received point trajectories."""
import cv2
import numpy as np
from .sparse_trajectory import gray

PARAMETERS=dict(min_speed=.15,min_active_samples=16,min_distinct_tracks=3,
                required_error_reduction=.05,under_motion_ratio=.25)


def select_window(context,motion,trajectory,times,endpoint,params=PARAMETERS):
    """One decision for an endpoint window; its end pixels agree in both arms."""
    if trajectory is None:
        return False,dict(reason='no received trajectory')
    if trajectory['frames']!=len(context) or context.shape!=motion.shape:
        raise ValueError('motion gate timeline mismatch')
    costs=[[],[]];frozen=[[],[]];track_ids=set();count=0
    for t in times:
        keep=((trajectory['origins']==endpoint)&trajectory['valid'][t-1]&trajectory['valid'][t])
        speed=trajectory['xy'][t]-trajectory['xy'][t-1]
        keep &= np.linalg.norm(speed,axis=1)>params['min_speed']
        if not np.any(keep):continue
        ids=np.flatnonzero(keep);track_ids.update(ids.tolist())
        xy=trajectory['xy'][t-1,keep].astype(np.float32);target=speed[keep]
        norm=np.linalg.norm(target,axis=1);count+=len(ids)
        for i,video in enumerate((context,motion)):
            flow=cv2.calcOpticalFlowFarneback(gray(video[t-1]),gray(video[t]),None,.5,3,15,3,5,1.2,0)
            sampled=cv2.remap(flow,xy[:,0].reshape(1,-1),xy[:,1].reshape(1,-1),cv2.INTER_LINEAR)[0]
            costs[i].extend(np.linalg.norm(sampled-target,axis=1).tolist())
            frozen[i].extend((np.linalg.norm(sampled,axis=1)<params['under_motion_ratio']*norm).tolist())
    audit=dict(active_samples=count,distinct_tracks=len(track_ids))
    if count<params['min_active_samples'] or len(track_ids)<params['min_distinct_tracks']:
        return False,dict(**audit,reason='insufficient received moving-track support')
    error=[float(np.mean(v)) for v in costs];under=[float(np.mean(v)) for v in frozen]
    select=error[1]<=(1-params['required_error_reduction'])*error[0] and under[1]<=under[0]
    return bool(select),dict(**audit,received_track_error=error,under_motion_fraction=under,
        reason='received motion improves' if select else 'received motion does not clearly improve')


def gate_segment(context,motion,trajectory,windows,progress=None):
    if context.dtype!=np.uint8 or context.shape!=motion.shape or len(context)<2:
        raise ValueError('matching uint8 videos required')
    if not np.array_equal(context[[0,-1]],motion[[0,-1]]):raise ValueError('keys differ')
    output=context.copy();audits=[]
    for endpoint,window in enumerate(windows):
        radius=window.get('radius',0)
        if not radius:
            audits.append(dict(endpoint=endpoint,selected=False,reason='empty window'));continue
        times=range(1,radius+1) if endpoint==0 else range(len(context)-radius,len(context))
        use,audit=select_window(context,motion,trajectory,times,endpoint)
        if use:
            if endpoint==0:output[:radius+1]=motion[:radius+1]
            else:output[-radius-1:]=motion[-radius-1:]
        audits.append(dict(endpoint=endpoint,radius=radius,selected=use,**audit))
        if progress:progress(endpoint+1,2)
    return output,dict(policy='received_trajectory_motion_gate_v1',parameters=PARAMETERS,
        endpoints=audits,source_only_used_for_evaluation=False,receiver_reads_source=False)
