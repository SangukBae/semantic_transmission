"""Reference-span intervention and noise replay must remain independently checkable."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("tail_probe", SCRIPTS / "etri_tail_reference_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_crop_and_rng_replay_survive_different_encoder_random_consumption(tmp_path):
    observed = []

    def append(vae, video, refs, strategies, loop, length, edit):
        observed.append(video.clone())
        # Model the shape-dependent random consumption of the stochastic VAE.
        torch.rand(video.shape[2] * 3)
        count = 5 * (video.shape[2] // 17) + (video.shape[2] % 17 + 3) // 4
        refs[0].append(torch.zeros(4, count, 1, 1))
        return refs, strategies

    utils = SimpleNamespace(append_generated=append,
        apply_mask_strategy=lambda z, refs, strategies, loop, align: torch.ones(z.shape[0], z.shape[2]),
        parse_mask_strategy=lambda strategy: [])
    video = torch.arange(38.).reshape(1, 1, 38, 1, 1)
    torch.manual_seed(2025)
    baseline = probe.Hooks(utils, probe.Replay(tmp_path/'bank', 'record'), tmp_path/'base.json')
    baseline.append(None, video, [[]], [""], 1, 5, 0)
    noise = torch.randn(1, 4, 7, 1, 1)
    expected_noise = noise.clone()
    baseline.apply(noise, [[]], [""], 1, align=5)
    expected_next = torch.randn(11)

    torch.rand(999)  # Deliberately perturb the stream before replay.
    variant = probe.Hooks(utils, probe.Replay(tmp_path/'bank', 'tail17'), tmp_path/'tail.json')
    variant.append(None, video, [[]], [""], 1, 5, 0)
    noise = torch.randn(1, 4, 7, 1, 1)
    assert not torch.equal(noise, expected_noise)
    variant.apply(noise, [[]], [""], 1, align=5)
    assert torch.equal(noise, expected_noise)
    assert torch.equal(torch.randn(11), expected_next)
    assert torch.equal(observed[0], video)
    assert torch.equal(observed[1], video[:, :, -17:])
    assert baseline.records['references'][0]['reference_rng_sha256'] == variant.records['references'][0]['reference_rng_sha256']
    assert variant.records['references'][0]['latent_shape'][1] == 5


def test_record_mode_leaves_native_noise_and_random_stream_unchanged(tmp_path):
    torch.manual_seed(21)
    noise = torch.randn(2, 3)
    state, original = torch.get_rng_state(), noise.clone()
    probe.Replay(tmp_path, "record").synchronize("sampling", 0, noise.device, noise)
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(original, noise)


def test_replay_rejects_incompatible_noise_shape(tmp_path):
    noise = torch.zeros(2, 3)
    probe.Replay(tmp_path, "record").synchronize("sampling", 0, noise.device, noise)
    with pytest.raises(ValueError, match="dimensions/dtype"):
        probe.Replay(tmp_path, "tail17").synchronize("sampling", 0, noise.device, torch.zeros(3, 3))


@pytest.mark.parametrize("field", [None, "noise_sha256", "sampling_rng_sha256", "reference_rng_sha256", "align"])
def test_summary_rejects_unmatched_conditions(tmp_path, monkeypatch, field):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import diagnose_etri_tail_reference as app
    mask = {'loop': 0, 'align': 5, 'noise_sha256': 'same', 'sampling_rng_sha256': {'cpu': 'same'},
            'active': [{'reference': 0, 'reference_sha256': 'key'}]}
    reference = {'loop': 1, 'reference_rng_sha256': {'cpu': 'same'},
                 'encoded_pixel_shape': [1, 3, 17, 2, 2], 'latent_shape': [4, 5, 1, 1]}
    a = {'mask': [mask, dict(mask, loop=1)], 'references': [reference]}
    b = copy.deepcopy(a)
    if field == 'reference_rng_sha256':
        b['references'][0][field] = {'cpu': 'wrong'}
    elif field:
        b['mask'][1][field] = 'wrong'
    if field:
        with pytest.raises(ValueError):
            app.validate_pair(a, b, 3)
    else:
        app.validate_pair(a, b, 3)
