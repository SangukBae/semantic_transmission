"""Accept restored motion only where it does not broadly erase existing motion."""
import cv2
import numpy as np
from .sparse_trajectory import gray

PARAMETERS=dict(active_speed=.15,loss_retention_ratio=.90,gain_retention_ratio=.25,
                max_lost_fraction=.05,min_active_cells=8)


def motion_cells(left,right):
    flow=cv2.calcOpticalFlowFarneback(gray(left),gray(right),None,.5,3,15,3,5,1.2,0)
    return np.linalg.norm(flow,axis=2).reshape(5,32,8,36).mean(axis=(1,3))


def select_motion(context,motion,times,params=PARAMETERS):
    old=np.stack([motion_cells(context[t-1],context[t]) for t in times])
    new=np.stack([motion_cells(motion[t-1],motion[t]) for t in times])
    active=old>params['active_speed'];new_active=new>params['active_speed']
    lost=int((active&(new<params['loss_retention_ratio']*old)).sum())
    gained=int((new_active&(old<params['gain_retention_ratio']*new)).sum())
    count=int(active.sum());fraction=lost/max(1,count)
    selected=count>=params['min_active_cells'] and gained>lost and fraction<=params['max_lost_fraction']
    return bool(selected),dict(active_cells=count,lost_motion_cells=lost,gained_motion_cells=gained,lost_fraction=fraction)


def gate_segment(context,motion,windows,progress=None):
    if context.dtype!=np.uint8 or motion.dtype!=np.uint8 or context.shape!=motion.shape or len(context)<2:
        raise ValueError('matching uint8 videos required')
    if not np.array_equal(context[[0,-1]],motion[[0,-1]]):raise ValueError('received keys differ')
    output=context.copy();audits=[]
    for endpoint,window in enumerate(windows):
        r=window.get('radius',0)
        if not r:audits.append(dict(endpoint=endpoint,radius=r,selected=False));continue
        far=r if endpoint==0 else len(context)-1-r
        if not np.array_equal(context[far],motion[far]):raise ValueError('far endpoint differs')
        times=range(1,r+1) if endpoint==0 else range(len(context)-r,len(context))
        selected,audit=select_motion(context,motion,times)
        if selected:
            if endpoint==0:output[:r+1]=motion[:r+1]
            else:output[-r-1:]=motion[-r-1:]
        audits.append(dict(endpoint=endpoint,radius=r,selected=selected,**audit))
        if progress:progress(endpoint+1,2)
    return output,dict(policy='motion_retention_gate_v1',parameters=PARAMETERS,endpoints=audits,
        receiver_reads_source=False,limitation='output-motion proxy only; cannot establish correct source motion')
