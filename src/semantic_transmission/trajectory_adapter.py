"""Experimental received-only trajectory features injected before STDiT blocks.

This is a small BSU prototype, not a reproduction or checkpoint port of Tora.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

POLICY='received_trajectory_patch_residual_adapter_v1'


def latent_positions(frames):
    if type(frames) is not int or frames<1:raise ValueError('positive frame count required')
    positions=[]
    for start in range(0,frames,17):
        count=min(17,frames-start);n=(count+3)//4
        positions.extend((start+np.linspace(0,count-1,n)).round().astype(int).tolist())
    return positions


def condition_maps(tracks,key_rgb,source_indices,height,width):
    """Rasterize received tracks and received-key colors at nominal VAE times.

    source_indices indexes the received trajectory timeline, never source RGB.
    Temporal pooling is explicit per 17-frame VAE block, not across block edges.
    """
    import cv2
    ids=np.asarray(source_indices,dtype=int)[latent_positions(len(source_indices))]
    out=np.zeros((10,len(ids),height,width),np.float32)
    if tracks is None:return out
    if ids.min()<0 or ids.max()>=len(tracks['xy']):raise ValueError('trajectory time outside received interval')
    if len(key_rgb)!=2:raise ValueError('two received keys required')
    keys=[cv2.resize(np.asarray(k),(288,160)).astype(np.float32)/127.5-1 for k in key_rgb]
    yy,xx=np.mgrid[:height,:width];duration=max(1,len(tracks['xy'])-1)
    for ti,t in enumerate(ids):
        for pi,origin in enumerate(tracks['origins']):
            if not tracks['valid'][t,pi]:continue
            point=tracks['xy'][t,pi];base=tracks['xy'][-1 if origin else 0,pi]
            if not tracks['valid'][-1 if origin else 0,pi]:continue
            col=keys[int(origin)][int(round(base[1])),int(round(base[0]))]
            displacement=(point-base)/np.array([288,160])
            at=point/np.array([288,160])*np.array([width,height])
            weight=np.exp(-((xx-at[0])**2+(yy-at[1])**2)/(2*1.5**2))
            values=np.r_[col,displacement,point/np.array([288,160]),float(origin),t/duration,1.]
            out[:,ti]+=values[:,None,None]*weight[None]
        support=out[9,ti].copy();out[:9,ti]/=np.maximum(support[None],1e-6)
        out[9,ti]=np.clip(support,0,1)
    if not np.isfinite(out).all():raise ValueError('nonfinite condition map')
    return out


class TrajectoryAdapter(nn.Module):
    def __init__(self,hidden=1152,width=32):
        super().__init__()
        self.features=nn.Sequential(nn.Conv3d(10,width,3,padding=1),nn.SiLU(),nn.Conv3d(width,width,3,padding=1),nn.SiLU())
        self.projection=nn.Conv3d(width,hidden,1)
        nn.init.zeros_(self.projection.weight);nn.init.zeros_(self.projection.bias)
        self.condition=None

    def forward(self,condition):
        feature=self.features(condition)
        feature=F.avg_pool3d(feature,(1,2,2),(1,2,2))
        valid=(condition[:,9:10].sum((2,3,4),keepdim=True)>0).to(feature.dtype)
        residual=self.projection(feature).tanh()*.25*valid
        return residual.flatten(2).transpose(1,2)

    def attach(self,model):
        def hook(module,args,output):
            if self.condition is None:return output
            delta=self(self.condition.to(device=output.device,dtype=next(self.parameters()).dtype))
            if output.shape[0]%delta.shape[0]:raise ValueError('CFG condition batch mismatch')
            delta=delta.repeat(output.shape[0]//delta.shape[0],1,1)
            if delta.shape!=output.shape:raise ValueError(f'condition/token alignment differs: {delta.shape} {output.shape}')
            return output+delta.to(output.dtype)
        return model.x_embedder.register_forward_hook(hook)
