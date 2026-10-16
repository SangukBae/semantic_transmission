"""Opt-in TAPNext++ sender tracking, serialized by the existing BSU codec."""
import sys
from pathlib import Path
import numpy as np


class TapNextTracker:
    def __init__(self,repo):
        import torch
        sys.path.insert(0,str(Path(repo)/'.local/vendor/tapnet'))
        from tapnet.tapnextpp.votsp2026.model import TAPNextPP
        self.model=TAPNextPP.from_checkpoint(Path(repo)/'.local/checkpoints/tapnextpp/model.pt',device='cuda')
        self.torch=torch

    def queries(self,frames,points):
        import cv2
        xy=[];valid=[];state=None
        for i,frame in enumerate(frames):
            p,v,state=self.model.track_frame(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR),points if i==0 else None,state=state)
            v &= np.isfinite(p).all(axis=-1)&(p[:,0]>=2)&(p[:,0]<frame.shape[1]-2)&(p[:,1]>=2)&(p[:,1]<frame.shape[0]-2)
            xy.append(np.nan_to_num(p).clip([0,0],[frame.shape[1]-1,frame.shape[0]-1]));valid.append(v)
        xy=np.asarray(xy,np.float32);valid=np.asarray(valid,bool)
        # Initial query coordinates are observed sender pixels, not predictions.
        xy[0]=points;valid[0]=True
        return xy,valid

    def track(self,frames,reverse=False):
        import cv2
        from .sparse_trajectory import gray,SIZE
        frames=np.stack([cv2.resize(f,SIZE,interpolation=cv2.INTER_AREA) for f in frames])
        if reverse:frames=frames[::-1].copy()
        points=cv2.goodFeaturesToTrack(gray(frames[0]),maxCorners=160,qualityLevel=.015,minDistance=9)
        if points is None:return np.zeros((len(frames),0,2),np.float32),np.zeros((len(frames),0),bool)
        xy,valid=self.queries(frames,points[:,0])
        return (xy[::-1].copy(),valid[::-1].copy()) if reverse else (xy,valid)


def candidates_from_tracks(tracks,start,end):
    from . import sparse_trajectory as sparse
    from .bsu_sequence_refinement import TRIALS
    candidates=[]
    for points,knots in TRIALS:
        parts=[]
        for reverse,(xy,valid) in enumerate(tracks):
            selected=sparse.choose_tracks(xy,valid,points//2,bool(reverse))
            parts.append((xy[:,selected],valid[:,selected],np.full(len(selected),reverse)))
        xy=np.concatenate([p[0] for p in parts],axis=1);valid=np.concatenate([p[1] for p in parts],axis=1)
        origins=np.concatenate([p[2] for p in parts])
        if xy.shape[1]<3:return []
        times=sparse.adaptive_knots(xy,valid,knots)
        candidates.append(dict(start=start,end=end,trajectory=sparse.encode(xy[times],valid[times],origins,times,end-start+1)))
    return candidates
