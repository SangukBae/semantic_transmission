"""Bounded short-pair motion sampling across a complete segment.

This changes an opt-in sender measurement, not the pretrained motion-score scale.
"""
import math

POLICY = 'distributed_short_pairs_legacy_time_scale_v1'


def distributed_pairs(frames, fps, count=3, reference_seconds=10/24):
    if (type(frames) is not int or frames < 1 or not math.isfinite(fps) or fps <= 0
            or type(count) is not int or count < 1 or reference_seconds <= 0):
        raise ValueError('invalid motion sampling dimensions')
    if frames == 1:
        return []
    gap = min(frames-1, max(1, round(fps*reference_seconds)))
    span = frames-1-gap
    starts = [round(i*span/(count-1)) for i in range(count)] if count > 1 else [span//2]
    return sorted({(a, a+gap) for a in starts})


def normalized_pair_score(displacements, pairs, fps, reference_seconds=10/24):
    if (not math.isfinite(fps) or fps <= 0 or len(displacements) != len(pairs)
            or pairs != sorted(set(pairs)) or any(a < 0 or a >= b for a,b in pairs)
            or any(not math.isfinite(v) or v < 0 for v in displacements)):
        raise ValueError('invalid measured motion pairs')
    if not pairs:
        return None
    # Each temporal stratum has equal weight. Keep the legacy L1 pixel measure.
    return sum(v*fps/(b-a)*reference_seconds for v,(a,b) in zip(displacements,pairs))/len(pairs)


def read_frames(path, indices):
    import cv2
    requested = sorted(set(indices)); found = {}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'cannot decode motion input: {path}')
    try:
        for i in range(max(requested, default=-1)+1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'missing frame {i}: {path}')
            if i in requested:
                found[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
    return found


class UniMatchScorer:
    def __init__(self, repo, device='cuda'):
        import sys
        import torch
        sys.path.insert(0,str(repo/'.local/vendor/Open-Sora'))
        from tools.scoring.optical_flow.unimatch import UniMatch
        self.device=device
        self.model=UniMatch(feature_channels=128,num_scales=2,upsample_factor=4,num_head=1,
            ffn_dim_expansion=4,num_transformer_layers=6,reg_refine=True,task='flow')
        self.model.load_state_dict(torch.load(repo/'.local/checkpoints/unimatch.pth',map_location='cpu',weights_only=True)['model'])
        self.model.to(device).eval()

    def measure(self, frames, pairs):
        import numpy as np
        import torch
        import torch.nn.functional as F
        if not pairs:return []
        ids=sorted({x for p in pairs for x in p}); index={t:i for i,t in enumerate(ids)}
        x=torch.from_numpy(np.stack([frames[t] for t in ids])).permute(0,3,1,2).float().to(self.device)
        x=F.interpolate(x,size=(320,576),mode='bilinear',align_corners=True)
        with torch.inference_mode():
            y=self.model(x[[index[a] for a,b in pairs]],x[[index[b] for a,b in pairs]],
                attn_type='swin',attn_splits_list=[2,8],corr_radius_list=[-1,4],prop_radius_list=[-1,1],
                num_reg_refine=6,task='flow',pred_bidir_flow=False)['flow_preds'][-1]
        return y.abs().mean(dim=(1,2,3)).cpu().tolist()
