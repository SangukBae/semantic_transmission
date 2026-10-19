"""Untrained sparse-key RIFE and received-trajectory output correction."""
from . import dense_temporal_receiver as old
from .sparc_motion_transport import validate_timeline


def interpolate(anchors, indices, video, vendor, checkpoints, progress=None):
    """Same pinned IFNet/math as SPARC, with an explicitly new gap contract."""
    import sys
    import numpy as np
    import torch
    import torch.nn.functional as F
    from .artifacts import sha256
    validate_timeline(video, indices)
    if anchors.shape != (len(indices), video['height'], video['width'], 3):
        raise ValueError('received anchor shape differs')
    for path,digest in [(checkpoints/'flownet.pkl',old.WEIGHT_SHA),
                        (checkpoints/'IFNet_HDv3.py',old.NETWORK_SHA),
                        (vendor/'model/warplayer.py',old.WARP_SHA)]:
        if sha256(path) != digest: raise ValueError('RIFE asset changed')
    sys.path[:0] = [str(checkpoints),str(vendor)]
    from IFNet_HDv3 import IFNet
    net=IFNet().cuda().eval()
    state=torch.load(checkpoints/'flownet.pkl',map_location='cpu',weights_only=True)
    state={k.removeprefix('module.'):v for k,v in state.items()}
    if any(not k.startswith(('teacher.','caltime.')) for k in set(state)-set(net.state_dict())):
        raise ValueError('unexpected RIFE state')
    net.load_state_dict({k:v for k,v in state.items() if k in net.state_dict()},strict=True)
    h,w=anchors.shape[1:3]; result=np.empty((video['frames'],h,w,3),np.uint8)
    def tensor(image):
        value=torch.from_numpy(image.copy()).permute(2,0,1)[None].cuda().float()/255
        return F.pad(value,(0,(-w)%64,0,(-h)%64))
    with torch.inference_mode():
        left=tensor(anchors[0]);result[0]=anchors[0]
        for k,(a,b) in enumerate(zip(indices,indices[1:])):
            right=tensor(anchors[k+1]);result[b]=anchors[k+1]
            for i in range(a+1,b):
                _,_,merged=net(torch.cat((left,right),1),(i-a)/(b-a),[16,8,4,2,1])
                result[i]=(merged[-1][0,:,:h,:w].permute(1,2,0).clamp(0,1)*255).round().byte().cpu().numpy()
            left=right
            if progress:progress(b+1,video['frames'])
    if not np.array_equal(result[indices],anchors):raise ValueError('RIFE changed received anchors')
    return result


def correct(frames, anchors, indices, trajectories, *, visibility=True, progress=None):
    import copy
    import cv2
    import numpy as np
    from .trajectory_receiver import refine_references
    cv2.setNumThreads(2);cv2.setRNGSeed(2025)
    result=frames.copy(); audits=[]
    for k,(a,b) in enumerate(zip(indices,indices[1:])):
        trajectory=trajectories.get((a,b))
        if trajectory is not None and not visibility:
            trajectory=copy.deepcopy(trajectory)
            trajectory['valid']=np.ones_like(trajectory['valid'])
        window,audit=refine_references(frames[a:b+1],anchors[k:k+2],trajectory)
        result[a:b+1]=window
        audits.append(dict(start=a,end=b,**audit))
        if progress:progress(k+1,len(indices)-1)
    if not np.array_equal(result[indices],anchors):raise ValueError('motion correction changed anchors')
    return result,dict(windows=audits,visibility_used=visibility,source_pixels_used=False,
        corrected_frames=int(np.any(result!=frames,axis=(1,2,3)).sum()))
