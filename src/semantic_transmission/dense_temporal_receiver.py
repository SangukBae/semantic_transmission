"""Learned temporal recovery, using only received decoded anchors."""
from pathlib import Path

WEIGHT_SHA = '6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2'
NETWORK_SHA = '655b4c772b037967b86c2dd31c8fa3b5323b79dd9a0e0088708d89149bbc8a32'
WARP_SHA = 'feb3af9475b724749b023d1913c3b135431eff13d0045b8efb472a699058385f'
POLICY = 'bsu_short_gap_rife425_v1'
VARIABLE_POLICY = 'sparc_variable_gap_rife425_v1'


def select_indices(images, stride=2, protected=()):
    """Pay both sides of abrupt RGB changes; never interpolate across a cut."""
    import numpy as np
    if stride not in (2, 3, 4) or len(images) < 2:
        raise ValueError('2-4 frame spacing and at least two frames required')
    if any(type(i) is not int or not 0 <= i < len(images) for i in protected):
        raise ValueError('invalid protected semantic timestamp')
    keys = set(range(0, len(images), stride)) | {len(images)-1} | set(protected)
    for i in range(1, len(images)):
        if np.abs(images[i].astype(np.float32)-images[i-1]).mean() > 40:
            keys.update((i-1, i))
    return sorted(keys)


def validate_indices(indices, frames, stride):
    if (type(frames) is not int or frames < 2 or
            (stride is not None and (type(stride) is not int or stride not in (2,3,4))) or
            not indices or any(type(x) is not int for x in indices) or
            indices != sorted(set(indices)) or indices[0] != 0 or indices[-1] != frames-1 or
            (stride is not None and any(b-a > stride for a,b in zip(indices,indices[1:])))):
        raise ValueError('invalid temporal timestamp map')


def interpolate(anchors, indices, frames, stride, vendor, checkpoints):
    import sys
    import numpy as np
    import torch
    import torch.nn.functional as F
    from .artifacts import sha256
    validate_indices(indices,frames,stride)
    if len(anchors) != len(indices):
        raise ValueError('anchor count mismatch')
    for path,digest in [(checkpoints/'flownet.pkl',WEIGHT_SHA),
                        (checkpoints/'IFNet_HDv3.py',NETWORK_SHA),
                        (vendor/'model/warplayer.py',WARP_SHA)]:
        if sha256(path) != digest:
            raise ValueError('RIFE implementation or weights changed')
    sys.path[:0] = [str(checkpoints),str(vendor)]
    from IFNet_HDv3 import IFNet
    net=IFNet().cuda().eval()
    state=torch.load(checkpoints/'flownet.pkl',map_location='cpu',weights_only=True)
    state={k.removeprefix('module.'):v for k,v in state.items()}
    # Official export also carries training-only teacher/time-prediction heads.
    # Retain strict matching for every inference parameter instead of the
    # upstream wrapper's unrestricted strict=False loading.
    extras=set(state)-set(net.state_dict())
    if any(not k.startswith(('teacher.','caltime.')) for k in extras):
        raise ValueError('unexpected RIFE checkpoint parameters')
    net.load_state_dict({k:v for k,v in state.items() if k not in extras},strict=True)
    h,w=anchors.shape[1:3];result=np.empty((frames,h,w,3),dtype=np.uint8)
    def tensor(a):
        x=torch.from_numpy(a.copy()).permute(2,0,1)[None].cuda().float()/255
        return F.pad(x,(0,(-w)%64,0,(-h)%64))
    with torch.inference_mode():
        left=tensor(anchors[0]);result[0]=anchors[0]
        for k,(a,b) in enumerate(zip(indices,indices[1:])):
            right=tensor(anchors[k+1]);result[b]=anchors[k+1]
            for i in range(a+1,b):
                _,_,merged=net(torch.cat((left,right),1),(i-a)/(b-a),[16,8,4,2,1])
                x=merged[-1][0,:,:h,:w].permute(1,2,0).clamp(0,1)
                result[i]=(x*255).round().byte().cpu().numpy()
            left=right
    return result
