import sys
from pathlib import Path

import pytest
import torch

from semantic_transmission.frequency_anchors import lowpass, correction, sample


def test_lowpass_passes_dc_rejects_checkerboard_and_is_projection():
    y, x = torch.meshgrid(torch.arange(16), torch.arange(24), indexing='ij')
    checker = ((x + y) % 2).float() * 2 - 1
    torch.testing.assert_close(lowpass(checker + 3, .25), torch.full_like(checker, 3))
    value = torch.randn(1, 4, 3, 16, 24)
    torch.testing.assert_close(lowpass(lowpass(value, .5), .5), lowpass(value, .5), atol=5e-7, rtol=1e-5)


def test_clean_estimate_sign_strength_and_temporal_isolation():
    clean = torch.randn(1, 4, 5, 16, 24)
    noise = torch.randn_like(clean)
    tau, dt = torch.tensor([.4]), torch.tensor([.1])
    z, velocity = .6 * clean + .4 * noise, clean - noise
    prior = clean + 2
    delta = correction(z, velocity, prior, tau, dt, [1, 3], .5, 1)
    torch.testing.assert_close(delta[:, :, [1, 3]], torch.full_like(delta[:, :, [1, 3]], .25))
    assert torch.count_nonzero(delta[:, :, [0, 2, 4]]) == 0
    assert torch.count_nonzero(correction(z, velocity, prior, tau, dt, [1], 0, .25)) == 0


def test_guidance_cannot_directly_replace_high_frequency_detail():
    z, v, prior = [torch.randn(1, 4, 5, 16, 24) for _ in range(3)]
    delta = correction(z, v, prior, torch.tensor([.3]), torch.tensor([.1]), [1, 2], .7, .25)
    torch.testing.assert_close(delta - lowpass(delta, .25), torch.zeros_like(delta), atol=2e-7, rtol=0)
    assert delta.square().sum() > 0


@pytest.mark.parametrize('fraction', [0, -1, 1.1])
def test_reject_invalid_band(fraction):
    with pytest.raises(ValueError):
        lowpass(torch.zeros(4, 4), fraction)


def test_reject_zero_timestep():
    z = torch.zeros(1, 4, 3, 8, 8)
    with pytest.raises(ValueError):
        correction(z, z, z, torch.tensor([0.]), torch.tensor([.1]), [1], .5, .25)


@pytest.mark.parametrize('transformed', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_disabled_matches_actual_vendor_scheduler_and_random_state(transformed, dtype):
    vendor = Path(__file__).resolve().parents[1] / '.local/vendor/Open-Sora'
    sys.path.insert(0, str(vendor))
    from opensora.schedulers.rf import RFLOW

    class Encoder:
        def encode(self, prompts):
            return dict(y=torch.ones(len(prompts), 1))

        def null(self, n):
            return torch.zeros(n, 1)

    class Model:
        def __call__(self, z, t, **kwargs):
            velocity = (z * .13 + kwargs['y'][:, :, None, None, None] * .03).to(dtype)
            return torch.cat([velocity, velocity * 0], 1)

    scheduler = RFLOW(num_sampling_steps=5, cfg_scale=7, use_timestep_transform=transformed)
    z = torch.randn(1, 4, 5, 8, 8).to(dtype)
    mask = torch.tensor([[0, 1, 1, 1, 0]], dtype=dtype)
    extra = dict(height=torch.tensor([320]), width=torch.tensor([576]), num_frames=torch.tensor([17]))
    torch.manual_seed(7)
    expected = scheduler.sample(Model(), Encoder(), z.clone(), ['test'], 'cpu', additional_args=extra,
                                mask=mask, progress=False)
    rng = torch.get_rng_state()
    torch.manual_seed(7)
    actual, _ = sample(scheduler, Model(), Encoder(), z.clone(), ['test'], 'cpu', extra,
                       mask, z, [1, 2, 3], 0, .25)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), rng)
    torch.manual_seed(7)
    guided, _ = sample(scheduler, Model(), Encoder(), z.clone(), ['test'], 'cpu', extra,
                       mask, z + 2, [1, 2, 3], .7, .25)
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(guided[:, :, [0, 4]], expected[:, :, [0, 4]])
    assert torch.isfinite(guided).all() and not torch.equal(guided, expected)
    with pytest.raises(ValueError, match='protected'):
        sample(scheduler, Model(), Encoder(), z, ['test'], 'cpu', extra, mask, z, [0], .7, .25)
