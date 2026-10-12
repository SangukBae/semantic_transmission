"""Experimental received-motion latent prior, with a small learned residual.

The STDiT/VAE are frozen. Training targets are VAE latents of separate originals.
The learned prior guides denoising; output RGB frames are never warped or blended.
This is a pilot architecture, not an implementation of Tora.
"""
import torch
from torch import nn
from torch.nn import functional as F


class MotionPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(22, 16, 3, padding=1), nn.SiLU(), nn.Conv3d(16, 4, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        time = features[:, 8:9]
        base = features[:, :4] * (1-time) + features[:, 4:8] * time
        return base + .5 * torch.tanh(self.net(features))


def latent_times(frames):
    # OpenSora VAE independently encodes 17-frame microblocks; the last can be shorter.
    result = []
    for start in range(0, frames, 17):
        count = min(17, frames-start)
        result.extend(start + min(i*4, count-1) for i in range((count-1)//4+1))
    return torch.tensor(result, dtype=torch.float32)/(frames-1)


def features(first, last, motion, frames, *, zero_motion=False):
    """Endpoint latents + decoded motion only. No access to source images."""
    import numpy as np
    if first.shape != last.shape or first.ndim != 4 or first.shape[1] != 4:
        raise ValueError('expected B,4,H,W endpoint conditions')
    device = first.device
    first, last = first.float(), last.float()
    times = latent_times(frames).to(device)
    n, h, w = len(times), first.shape[-2], first.shape[-1]
    knots = np.linspace(0, 1, len(motion['values']))
    pose = np.stack([np.interp(times.cpu().numpy(), knots, motion['values'][:, j]) for j in range(12)], axis=1)
    confidence = np.interp(times.cpu().numpy(), knots, motion['confidence'])
    if zero_motion:
        pose[:] = 0
    pose = torch.tensor(pose, device=device, dtype=torch.float32)
    confidence = torch.tensor(confidence, device=device, dtype=torch.float32)
    # Work in physical aspect-ratio coordinates, then normalize for grid_sample.
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h, device=device), torch.linspace(0, 1, w, device=device), indexing='ij')
    aspect = h/w
    positions = torch.stack([xx, yy*aspect], -1)
    matrices = torch.stack([1+pose[:, 0], -pose[:, 1], pose[:, 1], 1+pose[:, 0]], -1).reshape(n, 2, 2)
    shifts = pose[:, 2:4] * torch.tensor([1., aspect], device=device)
    region = F.interpolate(pose[:, 4:].reshape(n, 2, 2, 2).permute(0, 3, 1, 2), size=(h, w), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
    region = region * torch.tensor([1., aspect], device=device)
    origin = torch.einsum('nij,nhwj->nhwi', torch.linalg.inv(matrices), positions[None]-shifts[:, None, None]-region)
    final = torch.einsum('ij,nhwj->nhwi', matrices[-1], origin) + shifts[-1] + region[-1]
    divisor = torch.tensor([1., aspect], device=device)
    grids = [origin/divisor*2-1, final/divisor*2-1]
    warped = [F.grid_sample(value.expand(n, -1, -1, -1), grid, mode='bilinear', padding_mode='border', align_corners=True)
              for value, grid in zip((first, last), grids)]
    # B,C,T,H,W: 4+4 received endpoint features, time, 12 motion values, confidence.
    extra = torch.cat([times[:, None], pose, confidence[:, None]], 1)[:, :, None, None].expand(-1, -1, h, w)
    return torch.cat([*warped, extra], 1).permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def guidance_hook(prior, strength):
    """Add identical velocity residuals to conditional/unconditional branches."""
    if not 0 <= strength <= 1:
        raise ValueError('guidance strength out of range')
    def hook(module, inputs, output):
        z, time = inputs[:2]
        # First four output channels contain rectified-flow velocity.
        velocity = output[:, :4]
        conditional, unconditional = velocity.chunk(2)
        cfg_velocity = unconditional + 7.0*(conditional-unconditional)
        tau = time[:1].float().reshape(-1, 1, 1, 1, 1)/1000
        clean = z[:1].float() + tau*cfg_velocity.float()
        delta = strength * (prior-clean)/tau.clamp_min(.001)
        result = output.clone()
        result[:, :4] += delta.to(result.dtype).expand_as(velocity)
        return result
    return hook
