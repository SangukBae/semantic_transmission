"""Learned decoded-video conditions for frozen Open-Sora STDiT-v3.

The adapter transmits no features of its own. Its six input channels are four
received VAE latents, receiver-derived temporal activity and a paid QP value.
Text conditions continue through the existing pretrained text attention path.
"""
import torch
from torch import nn
from torch.nn import functional as F

POLICY = 'bsu_unified_dense_patch_adapter_v1'


def condition(prior, qp):
    if prior.ndim != 5 or prior.shape[1] != 4 or not torch.isfinite(prior).all() or not 0 <= qp <= 63:
        raise ValueError('invalid received latent condition')
    motion = torch.zeros_like(prior[:, :1])
    motion[:, :, 1:] = (prior[:, :, 1:]-prior[:, :, :-1]).abs().mean(1, keepdim=True)
    quality = torch.full_like(motion, qp/63.)
    return torch.cat((prior, motion.clamp(0, 4), quality), 1)


class DenseConditionAdapter(nn.Module):
    def __init__(self, hidden=1152, width=32):
        super().__init__()
        self.hidden, self.width = hidden, width
        self.features = nn.Sequential(nn.Conv3d(6, width, 3, padding=1), nn.SiLU(),
                                      nn.Conv3d(width, width, 3, padding=1), nn.SiLU())
        self.projection = nn.Conv3d(width, hidden, 1)
        nn.init.zeros_(self.projection.weight); nn.init.zeros_(self.projection.bias)
        self.received = None

    def forward(self, values):
        # PatchEmbed3D pads odd spatial dimensions before its 1x2x2 projection.
        values = F.pad(values, (0, values.shape[-1] % 2, 0, values.shape[-2] % 2))
        features = F.avg_pool3d(self.features(values), (1, 2, 2), (1, 2, 2))
        return (.25*self.projection(features).tanh()).flatten(2).transpose(1, 2)

    def attach(self, model):
        def inject(module, args, output):
            if self.received is None:
                raise ValueError('received condition was not set')
            delta = self(self.received.to(device=output.device, dtype=next(self.parameters()).dtype))
            if output.shape[0] % delta.shape[0]:
                raise ValueError('condition/CFG batch mismatch')
            delta = delta.repeat(output.shape[0]//delta.shape[0], 1, 1)
            if delta.shape != output.shape:
                raise ValueError('received condition/patch timeline mismatch')
            return output+delta.to(output.dtype)
        return model.x_embedder.register_forward_hook(inject)


def flow_objective(predicted, target, noise, z, time):
    """Standard rectified-flow target plus latent structure/motion supervision."""
    expected = target-noise
    clean = z+time[:, None, None, None, None]/1000*predicted
    flow = F.mse_loss(predicted, expected)
    temporal = F.l1_loss(clean[:, :, 1:]-clean[:, :, :-1], target[:, :, 1:]-target[:, :, :-1])
    edges = (F.l1_loss(clean[..., 1:]-clean[..., :-1], target[..., 1:]-target[..., :-1]) +
             F.l1_loss(clean[..., 1:, :]-clean[..., :-1, :], target[..., 1:, :]-target[..., :-1, :]))/2
    return flow+.1*temporal+.05*edges, dict(flow=flow, temporal=temporal, edges=edges)


def sample(model, adapter, prior, text, null, *, qp, height, width, fps, steps=12,
           start_noise=.3, cfg_scale=3., seed=2025):
    """Partial rectified-flow reconstruction initialized by decoded video.

    Actual multi-step integration; never labelled one-step distillation.
    Initial noise and all conditioning originate at RX, not from source pixels.
    """
    if not 0 < start_noise <= 1 or steps < 2:
        raise ValueError('invalid partial flow schedule')
    device, dtype = prior.device, next(model.parameters()).dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(prior.shape, device=device, dtype=torch.float32, generator=generator)
    z = (1-start_noise)*prior.float()+start_noise*noise
    adapter.received = condition(prior.float(), qp)
    times = torch.linspace(start_noise*1000, 0, steps+1, device=device)
    metadata = {k: torch.full((2,), v, device=device, dtype=dtype)
                for k, v in [('height', height), ('width', width), ('fps', fps)]}
    y = torch.cat((text['y'].to(device=device, dtype=dtype), null.to(device=device, dtype=dtype)), 0)
    # The null embedding uses the same packed text lengths as the conditional
    # branch, matching Open-Sora's released classifier-free guidance interface.
    mask = text['mask'].to(device)
    with torch.no_grad():
        for current, following in zip(times, times[1:]):
            prediction = model(z.repeat(2, 1, 1, 1, 1).to(dtype), current.repeat(2),
                               y=y, mask=mask, **metadata).chunk(2, dim=1)[0].float()
            conditional, unconditional = prediction.chunk(2)
            velocity = unconditional+cfg_scale*(conditional-unconditional)
            z = z+(current-following)/1000*velocity
            if not torch.isfinite(z).all():
                raise ValueError('nonfinite generated latent')
    adapter.received = None
    return z.to(dtype)
