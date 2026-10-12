"""Experimental VAE-aware endpoint conditioning; never blends output pixels.

The single-image reference used by the released receiver has one temporal slot.
A partial video block can have a different left padding and temporal context.
This module tests a phase-matched endpoint reference while retaining the learned
VAE and STDiT weights. It is an experiment, not an adopted receiver policy.
"""
from contextlib import contextmanager
import math

POLICY = "experimental_vae_phase_endpoint_v1"


def temporal_layout(frames, block_frames=17, factor=4):
    if type(frames) is not int or frames < 1:
        raise ValueError("positive integer frame count required")
    blocks, frame_offset, latent_offset = [], 0, 0
    while frame_offset < frames:
        count = min(block_frames, frames-frame_offset)
        length = math.ceil(count/factor)
        blocks.append(dict(frame_start=frame_offset, frames=count, latent_start=latent_offset,
                           latents=length, left_padding=(-count) % factor))
        frame_offset += count
        latent_offset += length
    return blocks


@contextmanager
def capture_single_image(vae):
    """Capture the *same* spatial latent and temporal epsilon as normal encode.

    The original sampling expression and RNG draw are preserved exactly. Later
    phase conditioning reuses this epsilon; it does not consume additional RNG.
    """
    import torch
    original = vae.temporal_vae.encode
    record = {}

    def encode(x):
        posterior = original(x)
        if x.shape[2] != 1 or record:
            raise ValueError("capture is only valid for one received image")
        record["spatial"] = x.detach().clone()
        def sample():
            epsilon = torch.randn(posterior.mean.shape).to(device=x.device, dtype=x.dtype)
            record["epsilon"] = epsilon.detach().clone()
            return posterior.mean + posterior.std * epsilon
        posterior.sample = sample
        return posterior

    vae.temporal_vae.encode = encode
    try:
        yield record
        if set(record) != {"spatial", "epsilon"}:
            raise ValueError("single-image sample was not captured")
    finally:
        vae.temporal_vae.encode = original


def phase_endpoint(vae, record, block_frames):
    """Encode constant received-key context and retain its final temporal slot.

    Spatial pixels are not sent to the output. Only a condition latent changes.
    The same already-sampled spatial latent and epsilon are used in every arm.
    """
    if not 1 <= block_frames <= vae.micro_frame_size:
        raise ValueError("endpoint context must fit one VAE temporal block")
    spatial, epsilon = record["spatial"], record["epsilon"]
    if spatial.shape[2] != 1 or epsilon.shape[2] != 1:
        raise ValueError("expected one received image and one epsilon slot")
    posterior = vae.temporal_vae.encode(spatial.repeat(1, 1, block_frames, 1, 1))
    sampled = posterior.mean[:, :, -1:] + posterior.std[:, :, -1:] * epsilon
    return (sampled - vae.shift) / vae.scale


def endpoint_condition(z, legacy, phase, mode):
    """Single-block diagnostic injection; callers supply a copied latent block."""
    if mode not in {"legacy_aligned", "exact_last", "phase_last"}:
        raise ValueError("unknown endpoint mode")
    if legacy.shape[2] != 1 or phase.shape != legacy.shape:
        raise ValueError("endpoint conditions must be single temporal slots")
    slot = 0 if mode == "legacy_aligned" else z.shape[2]-1
    z[:, :, slot:slot+1] = phase if mode == "phase_last" else legacy
    return z
