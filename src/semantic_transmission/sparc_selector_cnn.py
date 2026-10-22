"""Source-only spatial/temporal marginal rate-distortion predictor.

The CNN is trained jointly with temporal attention. Each action is evaluated as
a difference of two set values, so reverse actions are antisymmetric at inference.
Nothing in this module decodes a codec stream or reads quality measurements.
"""
import math

import numpy as np
import torch
from torch import nn

FRAME_HW = (96, 160)
CONTEXT_DIM = 18
GLOBAL_DIM = 6
MODEL_CONFIG = dict(width=128, heads=4, layers=2, dropout=.1)


def thumbnails(frames):
    import cv2
    h, w = FRAME_HW
    return np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
                     for f in frames]).astype(np.uint8)


def scene_ids(n, cuts):
    return np.searchsorted(np.asarray(cuts), np.arange(n), side='right')


def state_context(n, selected, keys, cuts, fps=24.):
    """UF group/rank/padding are computed separately after every scene reset."""
    if fps <= 0 or n < 2:
        raise ValueError('invalid source timeline')
    ids = sorted(set(selected))
    required = set(keys) | {0, n-1} | {i for c in cuts for i in (c-1, c)}
    if not required.issubset(ids) or any(i < 0 or i >= n for i in ids):
        raise ValueError('selection removes a protected frame')
    sid = scene_ids(n, cuts)
    context = np.zeros((n, CONTEXT_DIM), np.float32)
    times = np.arange(n)/fps
    context[:, 0] = times/12
    context[:, 1] = np.arange(n)/max(1, n-1)
    context[ids, 2] = 1
    context[list(keys), 3] = 1
    context[list({0, *cuts}), 4] = 1
    context[list({n-1, *(c-1 for c in cuts)}), 5] = 1
    # Continuous time encodings support variable gaps without a frame cap.
    for k, frequency in enumerate((.5, 2.)):
        context[:, 6+2*k] = np.sin(times*frequency)
        context[:, 7+2*k] = np.cos(times*frequency)
    total_pad = total_groups = 0
    maximum_gap = 0.
    for s in np.unique(sid):
        positions = np.flatnonzero(sid == s)
        anchors = np.asarray([i for i in ids if sid[i] == s])
        p_count = len(anchors)-1
        groups = (p_count+7)//8
        padding = (-p_count) % 8
        total_pad += padding
        total_groups += groups
        rank = np.searchsorted(anchors, positions, side='right')-1
        left = anchors[np.maximum(rank, 0)]
        right = anchors[np.minimum(np.searchsorted(anchors, positions), len(anchors)-1)]
        context[positions, 10] = (positions-left)/fps
        context[positions, 11] = (right-positions)/fps
        context[positions, 12] = (positions-positions[0])/max(1, len(positions)-1)
        context[positions, 13] = len(anchors)/len(positions)
        context[positions, 14] = np.maximum(rank-1, 0)//8/16
        context[positions, 15] = np.where(rank > 0, (rank-1) % 8/7, 0)
        context[positions, 16] = padding/8
        context[anchors[0], 17] = 1  # Actual scene I frame, not each UF group start.
        if len(anchors) > 1:
            maximum_gap = max(maximum_gap, float(np.diff(anchors).max()/fps))
    global_state = np.asarray([len(ids)/n, len(np.unique(sid))/n,
        total_groups/n, total_pad/n, maximum_gap, n/fps/12], np.float32)
    return context, global_state, sid.astype(np.int64)


def pair_context(n, records, keys, cuts, fps=24.):
    states, global_states = [], []
    for record in records:
        for name in ('before', 'after'):
            c, g, sid = state_context(n, record[name], keys, cuts, fps)
            states.append(c)
            global_states.append(g)
    return (torch.from_numpy(np.stack(states)),
            torch.from_numpy(np.stack(global_states)), torch.from_numpy(sid))


class SpatialGrid(nn.Module):
    """Keep 3x5 ordered regions with deterministic CUDA backward (Torch 2.2)."""
    def forward(self, x):
        h,w = x.shape[-2:]
        return torch.stack([
            x[..., i*h//3:math.ceil((i+1)*h/3),
                   j*w//5:math.ceil((j+1)*w/5)].mean((-2,-1))
            for i in range(3) for j in range(5)], -1)


class SpatialTemporalSelector(nn.Module):
    def __init__(self, width=128, heads=4, layers=2, dropout=.1):
        super().__init__()
        self.config = dict(width=width, heads=heads, layers=layers, dropout=dropout)
        self.heads = heads
        self.spatial = nn.Sequential(
            nn.Conv2d(6, 16, 5, stride=2, padding=2), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.GroupNorm(4, 24), nn.GELU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.GELU(),
            SpatialGrid(), nn.Flatten(),
            nn.Linear(32*3*5, width), nn.LayerNorm(width), nn.GELU())
        self.context = nn.Linear(CONTEXT_DIM, width)
        self.temporal = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, 2*width, dropout=dropout,
                                       activation='gelu', batch_first=True, norm_first=True)
            for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.quality = nn.Sequential(nn.Linear(2*width+GLOBAL_DIM, width),
                                     nn.GELU(), nn.Linear(width, 4))
        self.rate = nn.Sequential(nn.Linear(2*width+GLOBAL_DIM, width),
                                  nn.GELU(), nn.Linear(width, 1))

    def encode_frames(self, frames, scenes):
        """One CNN pass per clip, shared by all action candidates in a batch."""
        x = frames.float()/255
        previous = torch.cat((x[:1], x[:-1]), 0)
        delta = x-previous
        starts = torch.cat((torch.ones(1, device=x.device, dtype=torch.bool),
                            scenes[1:] != scenes[:-1]))
        delta = delta.masked_fill(starts[:, None, None, None], 0)
        pixels = torch.cat((2*x-1, delta), 1)
        # Bound the workspace without detaching CNN gradients.
        return torch.cat([self.spatial(chunk) for chunk in pixels.split(32)], 0)

    def predict_encoded(self, visual, context, global_state, scenes):
        z = visual.unsqueeze(0) + self.context(context)
        # No image-context exchange across scene cuts.
        mask = scenes[:, None] != scenes[None, :]
        for layer in self.temporal:
            z = layer(z, src_mask=mask)
        z = self.norm(z)
        selected = context[:, :, 2:3]
        selected_pool = (z*selected).sum(1)/selected.sum(1).clamp_min(1)
        pooled = torch.cat((z.mean(1), selected_pool, global_state), 1)
        distortion = self.quality(pooled).reshape(-1, 2, 4)
        rate = self.rate(pooled).reshape(-1, 2, 1)
        return torch.cat((distortion[:, 0]-distortion[:, 1],
                          rate[:, 1]-rate[:, 0]), 1)

    def forward(self, frames, context, global_state, scenes):
        return self.predict_encoded(self.encode_frames(frames, scenes),
                                    context, global_state, scenes)


def select(frames, keys, cuts, checkpoint_path, rate_lambda, fps=24., device='cpu'):
    """Variable-count greedy selection; source pixels only, protected anchors."""
    from .sparc_selection import adaptive, validate_anchors
    import cv2
    if not math.isfinite(rate_lambda) or rate_lambda < 0:
        raise ValueError('invalid validation-selected rate multiplier')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint['architecture'] != 'spatial_temporal_selector_v2':
        raise ValueError('incompatible selector checkpoint')
    model = SpatialTemporalSelector(**checkpoint['model_config']).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    small = thumbnails(frames)
    # Keep the v1 source-only initializer identical; CNN resolution is separate.
    initial = np.stack([cv2.resize(f, (128,72), interpolation=cv2.INTER_AREA) for f in frames])
    chosen = adaptive(initial, keys, cuts, max_gap=None, threshold=6.)['indices']
    protected = set(keys) | {0, len(frames)-1} | {i for c in cuts for i in (c-1,c)}
    pixels = torch.from_numpy(small).permute(0, 3, 1, 2).to(device)
    scenes = torch.from_numpy(scene_ids(len(frames), cuts)).to(device)
    scale = checkpoint['target_scale'].to(device)
    visited = {tuple(chosen)}
    with torch.inference_mode():
        visual = model.encode_frames(pixels, scenes)
        for _ in range(len(frames)):
            options = [sorted(set(chosen) ^ {i}) for i in range(len(frames))
                       if i not in protected and tuple(sorted(set(chosen) ^ {i})) not in visited]
            if not options:
                break
            scores = []
            for start in range(0, len(options), 16):
                records = [dict(before=chosen, after=a) for a in options[start:start+16]]
                c,g,s = pair_context(len(frames), records, keys, cuts, fps)
                y = model.predict_encoded(visual, c.to(device), g.to(device), s.to(device))*scale
                scores.extend((y[:,0]+.01*y[:,1]+.1*y[:,2]+.01*y[:,3]-rate_lambda*y[:,4]).tolist())
            best = int(np.argmax(scores))
            if scores[best] <= 0:
                break
            chosen = options[best]
            visited.add(tuple(chosen))
    validate_anchors(chosen, len(frames), None, cuts)
    return dict(indices=chosen, semantic_keys=keys, cuts=cuts, max_gap=None,
                selector='learned_cnn_attention_v2_experimental', rate_lambda=rate_lambda,
                receiver_quality_used=False, selection_reads_source_only=True)
