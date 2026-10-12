"""Training-free spatial-frequency guidance for the pinned Open-Sora RFLOW.

The received prior constrains selected temporal positions in latent space only.
No RGB blending, new random draws, weights, source pixels, or temporal filtering.
Latent frequency separation does not guarantee source-faithful RGB detail.
"""
import torch


def lowpass(value, fraction):
    """Orthogonal rectangular Fourier projector; fraction of each Nyquist band.

    Float32 FFT over H/W independently per batch/channel/time. The periodic
    boundary and hard spectral cutoff are explicit experimental choices.
    """
    if not 0 < fraction <= 1:
        raise ValueError('frequency fraction must be in (0, 1]')
    value = value.float()
    if fraction == 1:
        return value
    height, width = value.shape[-2:]
    fy = torch.fft.fftfreq(height, device=value.device).abs()
    fx = torch.fft.rfftfreq(width, device=value.device)
    keep = (fy[:, None] <= fraction / 2) & (fx[None, :] <= fraction / 2)
    spectrum = torch.fft.rfft2(value, norm='ortho')
    return torch.fft.irfft2(spectrum * keep, s=(height, width), norm='ortho')


def correction(z, velocity, prior, tau, dt, slots, strength, fraction):
    """Delta to the Euler update: strength * dt/tau * P(prior - x_clean).

    RFLOW uses x(t)=(1-t)*clean+t*noise and v=clean-noise, so the clean
    estimate is z+tau*v. A positive residual pulls towards the received prior.
    Returning zero outside slots keeps all other temporal positions untouched.
    """
    if z.shape != prior.shape or z.shape != velocity.shape or z.ndim != 5:
        raise ValueError('latent shapes must agree as B,C,T,H,W')
    if not 0 <= strength <= 1 or not 0 < fraction <= 1:
        raise ValueError('invalid guidance strength/frequency fraction')
    if len(slots) != len(set(slots)) or any(i < 0 or i >= z.shape[2] for i in slots):
        raise ValueError('invalid temporal slots')
    if torch.any(tau <= 0) or torch.any(dt < 0) or torch.any(dt > tau + 1e-6):
        raise ValueError('invalid RFLOW timestep')
    delta = torch.zeros_like(z, dtype=torch.float32)
    if not strength or not slots:
        return delta
    tau = tau.float().reshape(-1, 1, 1, 1, 1)
    dt = dt.float().reshape(-1, 1, 1, 1, 1)
    clean = z[:, :, slots].float() + tau * velocity[:, :, slots].float()
    residual = prior[:, :, slots].float() - clean
    delta[:, :, slots] = strength * dt / tau * lowpass(residual, fraction)
    return delta


def sample(scheduler, model, text_encoder, z, prompts, device, additional_args,
           mask, prior, slots, strength, fraction, tick=None):
    """Pinned RFLOW.sample mathematics with one post-CFG Euler correction.

    Binary masks only: original hard keys stay fixed, remaining positions are
    generated from the original noise. Disabled guidance executes identical
    RFLOW operations and is checked against actual historical PNG output.
    """
    from opensora.schedulers.rf.rectified_flow import timestep_transform
    if mask is None or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError('this experiment requires binary key masks')
    if any(not bool(torch.all(mask[:, i] == 1)) for i in slots):
        raise ValueError('guidance overlaps a protected key')
    model_args = text_encoder.encode(prompts)
    model_args['y'] = torch.cat([model_args['y'], text_encoder.null(len(prompts))], 0)
    model_args.update(additional_args)
    steps = scheduler.num_sampling_steps
    timesteps = [(1.0 - i / steps) * scheduler.num_timesteps for i in range(steps)]
    if scheduler.use_discrete_timesteps:
        timesteps = [int(round(t)) for t in timesteps]
    timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
    if scheduler.use_timestep_transform:
        timesteps = [timestep_transform(t, additional_args, num_timesteps=scheduler.num_timesteps)
                     for t in timesteps]
    noise_added = torch.zeros_like(mask, dtype=torch.bool) | (mask == 1)
    audit = []
    for i, t in enumerate(timesteps):
        mask_t = mask * scheduler.num_timesteps
        x0 = z.clone()
        x_noise = scheduler.scheduler.add_noise(x0, torch.randn_like(x0), t)
        active = mask_t >= t.unsqueeze(1)
        model_args['x_mask'] = active.repeat(2, 1)
        add_noise = active & ~noise_added
        z = torch.where(add_noise[:, None, :, None, None], x_noise, x0)
        noise_added = active
        prediction = model(torch.cat([z, z], 0), torch.cat([t, t], 0), **model_args).chunk(2, dim=1)[0]
        conditional, unconditional = prediction.chunk(2, dim=0)
        velocity = unconditional + scheduler.cfg_scale * (conditional - unconditional)
        dt = (timesteps[i] - timesteps[i + 1] if i < steps - 1 else timesteps[i]) / scheduler.num_timesteps
        updated = z + velocity * dt[:, None, None, None, None]
        row = dict(step=i + 1, tau=float(t[0] / scheduler.num_timesteps), dt=float(dt[0]))
        if strength:
            delta = correction(z, velocity, prior, t / scheduler.num_timesteps, dt, slots, strength, fraction)
            if not torch.isfinite(delta).all():
                raise ValueError('nonfinite structure correction')
            row['correction_rms'] = float(delta.square().mean().sqrt())
            row['correction_highpass_rms'] = float((delta - lowpass(delta, fraction)).square().mean().sqrt())
            updated = updated + delta
        z = torch.where(active[:, None, :, None, None], updated, x0)
        audit.append(row)
        if tick:
            tick(i + 1, steps)
    return z, audit
