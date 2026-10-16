"""Training-free received-video guidance inside BSU rectified-flow denoising.

This does not load the user's still-training trajectory adapter. It uses the
frozen BSU VAE and applies a bounded clean-latent consistency term to both CFG
branches, preserving the existing key masks, prefix context, and noise draws.
"""
from pathlib import Path
import json

from .artifacts import sha256, write_json

POLICY = 'bsu_received_dense_latent_consistency_v1'
TRACE = 'receiver/dense_condition_trace.json'


def guide_latents(vae, pixels):
    """Posterior modes avoid random draws inside BSU's scoped sampling noise.

    Only this new guide uses deterministic posterior modes. Received-key VAE
    sampling and all baseline generation noise remain on the original path.
    """
    import torch
    b,c,t,h,w=pixels.shape
    flat=pixels.permute(0,2,1,3,4).reshape(b*t,c,h,w)
    spatial=vae.spatial_vae;batch=spatial.micro_batch_size or len(flat)
    encoded=torch.cat([spatial.module.encode(flat[i:i+batch]).latent_dist.mode()*spatial.scaling_factor
                       for i in range(0,len(flat),batch)],dim=0)
    z=encoded.reshape(b,t,*encoded.shape[1:]).permute(0,2,1,3,4)
    chunk=vae.micro_frame_size or t
    prior=torch.cat([vae.temporal_vae.encode(z[:,:,i:i+chunk]).mode() for i in range(0,t,chunk)],dim=2)
    return (prior-vae.shift)/vae.scale


def corrected_velocity(z, output, time, prior, free, *, cfg_scale=7., strength=.12):
    import torch
    if z.shape[0] != 2 or output.shape[0] != 2 or output.shape[1] != 2*z.shape[1]:
        raise ValueError('expected single-video classifier-free guidance branches')
    if prior.shape != z[:1].shape or free.shape != (1,z.shape[2]):
        raise ValueError('received dense prior shape differs from generated latents')
    if not 0 <= strength <= 1:
        raise ValueError('invalid consistency strength')
    c = z.shape[1]; conditional, unconditional = output[:,:c].float().chunk(2)
    tau = time[:1].float().reshape(-1,1,1,1,1)/1000
    estimate = z[:1].float()+tau*(unconditional+cfg_scale*(conditional-unconditional))
    delta = strength*(1-tau).clamp(0,1)**2*(prior.float()-estimate).clamp(-2,2)/(tau+.05)
    delta *= free[:,None,:,None,None]
    result = output.clone(); result[:,:c] += delta.to(result.dtype)
    return result


def install(run):
    import numpy as np
    from PIL import Image
    import torch
    from opensora import registry
    from opensora.schedulers.rf import RFLOW
    read = lambda p:json.loads(Path(p).read_text())
    config = read(run/'dense_runtime.json'); keys = read(run/'keyframes.json')['indices']
    guide = Path(config['guide_frames'])
    if config['policy'] != POLICY:
        raise ValueError('dense conditioning policy differs')
    for name,digest in config['guide_sha256'].items():
        if sha256(guide/name) != digest:
            raise ValueError('received-only guide changed')
    original_build, original_sample = registry.build_module, RFLOW.sample
    captured, records = {}, []

    def build(*args,**kwargs):
        module = original_build(*args,**kwargs)
        if all(hasattr(module,k) for k in ('get_latent_size','encode','decode','out_channels')):
            captured['vae'] = module
        return module

    def sample(scheduler, model, text_encoder, z, *args, **kwargs):
        index=len(records);a,b=keys[index:index+2];vae=captured['vae']
        frames=np.stack([np.asarray(Image.open(guide/f'{t:05d}.png').convert('RGB')) for t in range(a,b+1)])
        # No stochastic VAE calls are inserted into the scoped RFLOW sampler.
        with torch.inference_mode():
            tensor=torch.from_numpy(frames.copy()).permute(3,0,1,2)[None].to(device=z.device,dtype=z.dtype)/127.5-1
            prior=guide_latents(vae,tensor)
            if index:
                prior=torch.cat([torch.zeros_like(z[:,:,:5]),prior],dim=2)
        if prior.shape!=z.shape:
            raise ValueError('guide VAE timeline mismatch')
        free=kwargs['mask'].clone()
        if index:free[:,:5]=0
        calls=[]
        def hook(module,inputs,output):
            x,time=inputs[:2]
            result=corrected_velocity(x,output,time,prior,free,
                cfg_scale=kwargs.get('guidance_scale') or scheduler.cfg_scale,strength=config['strength'])
            calls.append(dict(time=float(time[0]),mean_change=float((result-output).float().abs().mean())))
            return result
        handle=model.register_forward_hook(hook)
        try:result=original_sample(scheduler,model,text_encoder,z,*args,**kwargs)
        finally:handle.remove()
        records.append(dict(segment=index,start=a,end=b,prior_shape=list(prior.shape),
            prefix_latents_excluded=5 if index else 0,calls=calls,source_reads=False,
            all_free_slots_guided=True,strength=config['strength'],guide_vae_posterior='mode'))
        write_json(run/TRACE,records)
        return result

    registry.build_module=build;RFLOW.sample=sample
    return records
