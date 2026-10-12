"""Common-grid sequential versus joint overlap RFLOW, without pixel blending.

This local velocity-consensus experiment is not a reproduction of FreeNoise.
"""
from .receiver_temporal_alignment import temporal_layout

POLICY = 'common_vae_grid_overlap_velocity_consensus_v1'


def grid_plan(frames, keys):
    blocks = temporal_layout(frames)
    if not keys or keys[0] != 0 or keys[-1] != frames-1 or any(b <= a for a, b in zip(keys, keys[1:])):
        raise ValueError('ordered endpoint-inclusive keys required')
    anchors = []
    for frame in keys:
        block = next(b for b in blocks if b['frame_start'] <= frame < b['frame_start']+b['frames'])
        slot = (frame-block['frame_start']+block['left_padding'])//4
        anchors.append(dict(frame=frame, block_frames=block['frames'], block_slot=slot,
                            latent=block['latent_start']+slot))
    if len({a['latent'] for a in anchors}) != len(anchors):
        raise ValueError('received keys collide on the shared grid')
    windows = []
    for i in range(max(1, len(blocks)-1)):
        first, last = blocks[i], blocks[min(i+1, len(blocks)-1)]
        windows.append(dict(start=first['latent_start'], stop=last['latent_start']+last['latents'],
            first_frame=first['frame_start'], frames=last['frame_start']+last['frames']-first['frame_start']))
    return dict(frames=frames, latent_frames=sum(b['latents'] for b in blocks), blocks=blocks,
                anchors=anchors, windows=windows)


def key_condition(vae, capture, anchor):
    posterior = vae.temporal_vae.encode(capture['spatial'].repeat(1, 1, anchor['block_frames'], 1, 1))
    i = anchor['block_slot']
    sample = posterior.mean[:, :, i:i+1] + posterior.std[:, :, i:i+1]*capture['epsilon']
    return (sample-vae.shift)/vae.scale


def sample_windows(initial, fixed, windows, times, predict, mode, progress=None):
    """Binary-condition RFLOW. Native add-noise draws have zero contribution.

predict(window, z, t, active) returns CFG velocity; native single-window parity
and zero RNG consumption are tested. Received-key anchors stay fixed.
"""
    import torch
    if mode not in {'sequential', 'joint'}:
        raise ValueError('unknown window sampling mode')
    if initial.ndim != 5 or fixed.shape != (initial.shape[0], initial.shape[2]) or fixed.dtype != torch.bool:
        raise ValueError('invalid latent/fixed-mask shape')
    if not windows or not times:
        raise ValueError('windows and timesteps required')
    coverage = torch.zeros(initial.shape[2], device=initial.device, dtype=torch.float32)
    for a, b in windows:
        if not 0 <= a < b <= initial.shape[2]:
            raise ValueError('window outside latent timeline')
        coverage[a:b] += 1
    if (coverage == 0).any():
        raise ValueError('uncovered latent timeline')
    if any(not torch.isfinite(t).all() or (t <= 0).any() or (t > 1000).any() for t in times):
        raise ValueError('invalid timesteps')
    if any((a <= b).any() for a, b in zip(times, times[1:])):
        raise ValueError('timesteps must decrease strictly')
    state = initial.float().clone()
    anchor = state.clone()
    calls, total = 0, len(windows)*len(times)
    def velocity(index, value, t, active):
        nonlocal calls
        result = predict(index, value, t, active)
        if result.shape != value.shape or not torch.isfinite(result).all():
            raise ValueError('invalid predicted velocity')
        calls += 1
        if progress is not None:
            progress(calls, total)
        return result.float()
    def dt(i):
        return ((times[i] - times[i+1] if i+1 < len(times) else times[i])/1000)[:, None, None, None, None]
    if mode == 'joint':
        active = ~fixed
        for i, t in enumerate(times):
            total_velocity = torch.zeros_like(state)
            for index, (a, b) in enumerate(windows):
                total_velocity[:, :, a:b] += velocity(index, state[:, :, a:b], t, active[:, a:b])
            updated = state + total_velocity/coverage[None, None, :, None, None]*dt(i)
            state = torch.where(active[:, None, :, None, None], updated, anchor)
    else:
        completed = torch.zeros_like(fixed)
        for index, (a, b) in enumerate(windows):
            active = ~(fixed[:, a:b] | completed[:, a:b])
            value = state[:, :, a:b].clone()
            reference = value.clone()
            for i, t in enumerate(times):
                updated = value + velocity(index, value, t, active)*dt(i)
                value = torch.where(active[:, None, :, None, None], updated, reference)
            state[:, :, a:b] = value
            completed[:, a:b] = True
    if not torch.equal(state.transpose(1, 2)[fixed], anchor.transpose(1, 2)[fixed]):
        raise ValueError('received anchor changed during generation')
    return state
