"""Sender-only motion/visibility proposals; full codec trials decide acceptance."""
import math

from .sparc_selection import validate_anchors


def protected(semantic, cuts, frames):
    return set(semantic) | {0, frames-1} | {i for c in cuts for i in (c-1,c)}


def eligible(indices, mandatory, frames, cuts, max_gap=4):
    result=[]
    for i in indices:
        if i in mandatory: continue
        candidate=[k for k in indices if k!=i]
        try: validate_anchors(candidate,frames,max_gap,cuts)
        except ValueError: continue
        result.append(i)
    return result


def removal_proposals(indices, ranked, mandatory, frames, cuts, sizes):
    proposals=[]
    for size in sizes:
        selected=list(indices); removed=[]
        for i in ranked:
            if i in eligible(selected,mandatory,frames,cuts):
                selected.remove(i); removed.append(i)
                if len(removed)==size: break
        if removed and all(selected!=p['indices'] for p in proposals):
            proposals.append(dict(indices=selected,removed=removed,added=[],operation='remove'))
    # Single alternatives avoid stopping because only the first candidate fails.
    for i in ranked:
        selected=[k for k in indices if k!=i]
        if all(selected!=p['indices'] for p in proposals):
            proposals.append(dict(indices=selected,removed=[i],added=[],operation='remove'))
    return proposals


def uniform_semantic(frames, count, mandatory, cuts, gap=4):
    anchors=sorted(mandatory)
    lengths=[b-a for a,b in zip(anchors,anchors[1:])]
    parts=[math.ceil(n/gap) for n in lengths]
    if not sum(parts)+1<=count<=frames: raise ValueError('infeasible uniform count')
    while sum(parts)+1<count:
        k=max((i for i,n in enumerate(lengths) if parts[i]<n),key=lambda i:(lengths[i]/parts[i],-i))
        parts[k]+=1
    result=[anchors[0]]
    for a,b,n in zip(anchors,anchors[1:],parts):
        result.extend(a+round(j*(b-a)/n) for j in range(1,n+1))
    validate_anchors(result,frames,gap,cuts)
    assert len(result)==count and mandatory.issubset(result)
    return result


def quality_gate(candidate, baseline, config):
    """All deltas are against the immutable baseline, never the previous step."""
    import numpy as np
    c,b=candidate['quality'],baseline['quality']; reasons=[]
    if not all(math.isfinite(v) for quality in (c,b) for v in quality.values()):
        raise ValueError('nonfinite quality cannot pass acceptance')
    for name in ('psnr_db','lpips_vgg'):
        left,right=candidate['per_frame'][name],baseline['per_frame'][name]
        if not left or len(left)!=len(right) or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError('invalid per-frame quality timeline')
    if c['psnr_db']<b['psnr_db']-config['psnr_drop_db']: reasons.append('psnr')
    if c['lpips_vgg']>b['lpips_vgg']+config['lpips_increase']: reasons.append('lpips')
    for name,key in [('roi_mae','roi_error_ratio'),('temporal_mae','temporal_error_ratio')]:
        if c[name]>max(b[name]*config[key],b[name]+1e-9): reasons.append(name)
    n=config['window_frames']
    for name,limit,sign in [('psnr_db',config['window_psnr_drop_db'],-1),('lpips_vgg',config['window_lpips_increase'],1)]:
        delta=sign*(np.asarray(candidate['per_frame'][name])-np.asarray(baseline['per_frame'][name]))
        means=np.convolve(delta,np.ones(min(n,len(delta)))/min(n,len(delta)),mode='valid')
        if float(means.max())>limit: reasons.append('window_'+name)
    return dict(passed=not reasons,reasons=reasons)


class Signals:
    """Optical-flow heuristics, not semantic segmentation or measured RIFE error."""
    def __init__(self, source, config, legacy_windows=()):
        import cv2
        import numpy as np
        self.np=np;self.cv2=cv2;self.config=config
        self.frames=np.stack([cv2.resize(f,tuple(config['ranking_resolution']),interpolation=cv2.INTER_AREA) for f in source])
        self.gray=[cv2.cvtColor(f,cv2.COLOR_RGB2GRAY) for f in self.frames]
        h,w=self.gray[0].shape;self.xx,self.yy=np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
        self.flows={};self.visibility={};self.risk=np.zeros(len(source));self.track_risk=np.zeros(len(source))
        self.masks=np.zeros((len(source),h,w),bool)
        previous=None
        for i in range(len(source)-1):
            f=self.flow(i,i+1); residual=f-np.median(f,axis=(0,1))
            magnitude=np.linalg.norm(residual,axis=-1)
            change=np.zeros_like(magnitude) if previous is None else np.linalg.norm(residual-previous,axis=-1)
            strength=magnitude+change
            self.masks[i]=strength>max(.15,float(np.quantile(strength,.8)))
            self.risk[i]=float(np.quantile(change,.95)); previous=residual
        self.masks[-1]=self.masks[-2];self.risk[-1]=self.risk[-2]
        if legacy_windows:
            from .sparse_trajectory import decode
            for win in legacy_windows:
                a,b=win['start'],win['end'];data=decode(win['trajectory'],b-a+1)
                xy,valid=data['xy'],data['valid']
                for j in range(1,len(xy)-1):
                    good=valid[j-1]&valid[j]&valid[j+1]
                    if good.any():
                        acceleration=np.linalg.norm(xy[j+1,good]-2*xy[j,good]+xy[j-1,good],axis=-1)
                        self.track_risk[a+j]=max(self.track_risk[a+j],float(np.quantile(acceleration,.9)))
        self.risk=self.risk/max(float(np.quantile(self.risk,.95)),1e-6)
        self.track_risk=self.track_risk/max(float(np.quantile(self.track_risk,.95)),1e-6)

    def flow(self,a,b):
        if (a,b) not in self.flows:
            self.flows[a,b]=self.cv2.calcOpticalFlowFarneback(self.gray[a],self.gray[b],None,.5,3,15,3,5,1.2,0)
        return self.flows[a,b]

    def unseen(self,target,anchor):
        key=(target,anchor)
        if key not in self.visibility:
            np,cv2=self.np,self.cv2
            f=self.flow(target,anchor);back=self.flow(anchor,target)
            x,y=self.xx+f[...,0],self.yy+f[...,1]
            warped=cv2.remap(self.frames[anchor],x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            reverse=cv2.remap(back,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            residual=np.abs(warped.astype(float)-self.frames[target]).mean(-1)
            self.visibility[key]=((x<0)|(y<0)|(x>self.xx.max())|(y>self.yy.max())|
                (np.linalg.norm(f+reverse,axis=-1)>1.5)|(residual>20))
        return self.visibility[key]

    def score(self,i,a,b):
        both=self.unseen(i,a)&self.unseen(i,b)
        # High quantile over tiles keeps small newly visible regions relevant.
        tiles=both.reshape(8,10,12,12).mean(axis=(1,3))
        return float(self.risk[i]+.5*self.track_risk[i]+4*both.mean()+2*self.np.quantile(tiles,.95))

    def rank(self,indices,mandatory,cuts):
        ranking=[]
        for i in eligible(indices,mandatory,len(self.frames),cuts):
            k=indices.index(i);ranking.append((self.score(i,indices[k-1],indices[k+1]),i))
        return [i for _,i in sorted(ranking)]

    def relocations(self,indices,mandatory,cuts):
        results=[]
        for k,i in enumerate(indices[1:-1],1):
            if i in mandatory:continue
            a,b=indices[k-1],indices[k+1]
            before=self.score(i,a,b)
            for j in range(max(a+1,i-self.config['relocation_radius']),min(b,i+self.config['relocation_radius']+1)):
                if j in indices:continue
                candidate=sorted((set(indices)-{i})|{j})
                try:validate_anchors(candidate,len(self.frames),4,cuts)
                except ValueError:continue
                results.append(dict(indices=candidate,removed=[i],added=[j],operation='relocate',
                    risk_gain=self.score(j,a,b)-before))
        return sorted(results,key=lambda p:(-p['risk_gain'],p['removed'],p['added']))
