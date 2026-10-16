"""Received dense-video conditions and optional pretrained spatial restoration."""
from pathlib import Path


def restore(guide, size, mode='bicubic', *, model_root=None, checkpoint=None):
    import cv2
    import numpy as np
    if guide.ndim != 4 or guide.shape[-1] != 3 or guide.dtype != np.uint8 or not len(guide):
        raise ValueError('nonempty RGB uint8 received guide required')
    if mode == 'bicubic':
        return np.stack([cv2.resize(f, tuple(size), interpolation=cv2.INTER_CUBIC) for f in guide])
    if mode != 'swinir' or tuple(size) != (guide.shape[2]*2, guide.shape[1]*2):
        raise ValueError('SwinIR receiver requires exact 2x spatial ratio')
    import sys
    import torch
    sys.path.insert(0, str(model_root))
    from models.network_swinir import SwinIR
    model = SwinIR(upscale=2, in_chans=3, img_size=64, window_size=8,
        img_range=1., depths=[6,6,6,6], embed_dim=60, num_heads=[6,6,6,6],
        mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state.get('params', state), strict=True)
    model = model.cuda().eval().requires_grad_(False)
    result = []
    with torch.inference_mode():
        for frame in guide:
            x = torch.from_numpy(frame.copy()).permute(2,0,1)[None].float().cuda()/255
            h, w = x.shape[-2:]; ph, pw = (-h)%8, (-w)%8
            x = torch.nn.functional.pad(x, (0,pw,0,ph), mode='reflect')
            y = model(x)[0, :, :h*2, :w*2].permute(1,2,0).float().cpu().numpy()
            result.append(np.rint(np.clip(y, 0, 1)*255).astype(np.uint8))
    return np.stack(result)


def detail_projection(guide, generated, *, strength=.25, max_delta=12.):
    """Bound generated detail using received RGB agreement; never move pixels/time.

    Neither original source nor evaluation results enter this function. This
    explicit receiver policy is fixed before evaluation, not an oracle selector.
    """
    import cv2
    import numpy as np
    if guide.shape != generated.shape or guide.dtype != np.uint8 or generated.dtype != np.uint8:
        raise ValueError('matching RGB guide and generation required')
    if not 0 <= strength <= 1 or not 0 <= max_delta <= 255:
        raise ValueError('invalid detail bounds')
    result, changes = [], []
    for a, b in zip(guide, generated):
        a, b = a.astype(np.float32), b.astype(np.float32)
        low_a, low_b = (cv2.GaussianBlur(x, (0,0), 1.2) for x in (a,b))
        agreement = np.exp(-np.mean(np.abs(low_a-low_b), axis=-1, keepdims=True)/12.)
        delta = np.clip((b-low_b)-(a-low_a), -max_delta, max_delta)*agreement*strength
        out = np.rint(np.clip(a+delta,0,255)).astype(np.uint8)
        changes.append(float(np.abs(out.astype(float)-a).mean())); result.append(out)
    return np.stack(result), dict(mean_absolute_pixel_change=float(np.mean(changes)),
        maximum_possible_change=strength*max_delta, pixel_warping=False, temporal_interpolation=False)
