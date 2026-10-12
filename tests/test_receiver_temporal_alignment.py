import pytest
import torch

from semantic_transmission.receiver_temporal_alignment import (
    capture_single_image, endpoint_condition, phase_endpoint, temporal_layout)


def test_actual_partial_block_padding_and_end_position():
    assert temporal_layout(42) == [
        dict(frame_start=0, frames=17, latent_start=0, latents=5, left_padding=3),
        dict(frame_start=17, frames=17, latent_start=5, latents=5, left_padding=3),
        dict(frame_start=34, frames=8, latent_start=10, latents=2, left_padding=0)]
    for frames in range(1, 130):
        blocks = temporal_layout(frames)
        assert sum(b['frames'] for b in blocks) == frames
        assert sum(b['latents'] for b in blocks) == frames//17*5 + ((frames%17)+3)//4


@pytest.mark.parametrize('frames', [0, -1, 2.5, True])
def test_invalid_lengths(frames):
    with pytest.raises(ValueError):
        temporal_layout(frames)


def test_only_requested_condition_slot_changes():
    original = torch.arange(12.).view(1, 1, 12, 1, 1)
    old = torch.tensor([[[[[30.]]]]]);new = old+1
    for mode, slot, value in [('legacy_aligned', 0, 30), ('exact_last', 11, 30), ('phase_last', 11, 31)]:
        result = endpoint_condition(original.clone(), old, new, mode)
        assert result[0, 0, slot, 0, 0] == value
        keep = torch.arange(12) != slot
        assert torch.equal(result[:, :, keep], original[:, :, keep])


def test_capture_preserves_rng_and_restores_encoder():
    from types import SimpleNamespace
    class Posterior:
        def __init__(self, x):
            self.mean = x*2;self.std = torch.ones_like(x)*.2
        def sample(self):
            return self.mean + self.std*torch.randn(self.mean.shape).to(self.mean)
    temporal = SimpleNamespace(encode=lambda x: Posterior(x))
    vae = SimpleNamespace(temporal_vae=temporal)
    original = temporal.encode
    x = torch.ones(1, 2, 1, 3, 4)
    torch.manual_seed(11);expected = temporal.encode(x).sample();next_random = torch.randn(3)
    torch.manual_seed(11)
    with capture_single_image(vae) as record:
        actual = temporal.encode(x).sample()
    assert torch.equal(expected, actual)
    assert torch.equal(next_random, torch.randn(3))
    assert temporal.encode is original
    assert torch.equal(record['spatial'], x)
    with pytest.raises(RuntimeError):
        with capture_single_image(vae):
            raise RuntimeError('injected failure')
    assert temporal.encode is original


def test_phase_reference_uses_same_epsilon_without_random_draw():
    from types import SimpleNamespace
    inputs=[]
    def encode(x):
        inputs.append(x)
        return SimpleNamespace(mean=x[:, :, ::4]*2, std=torch.ones_like(x[:, :, ::4])*.3)
    vae=SimpleNamespace(temporal_vae=SimpleNamespace(encode=encode),micro_frame_size=17,shift=.1,scale=2)
    record=dict(spatial=torch.ones(1, 2, 1, 3, 4),epsilon=torch.ones(1, 2, 1, 3, 4)*.5)
    rng=torch.get_rng_state().clone()
    value=phase_endpoint(vae,record,8)
    assert torch.equal(torch.get_rng_state(),rng)
    assert inputs[0].shape[2]==8
    assert torch.allclose(value,torch.full_like(value,(2+.3*.5-.1)/2))


@pytest.mark.parametrize('loop,latents', [(0, 7), (1, 12)])
def test_paired_masks_preserve_history_and_change_only_end_condition(loop, latents):
    from pathlib import Path
    from semantic_transmission.condition_collision import upstream_functions, apply_guard
    from semantic_transmission.receiver_alignment_experiment import mask_for_arm
    functions = upstream_functions(Path(__file__).resolve().parents[1])
    functions.apply_mask_strategy.__globals__['torch'] = torch
    refs = [[torch.full((2, 1, 2, 3), float(i+10)) for i in range(3)]]
    if loop:
        refs[0].append(torch.full((2, 5, 2, 3), 13.))
    strategy = '0;0,1,0,-1,1;1,2,0,-1,1;1,3,-5,0,5,0'
    z = torch.randn(1, 2, latents, 2, 3)
    existing = z.clone()
    existing_mask, _ = apply_guard(existing, refs, [strategy], loop, 5, 3, functions)
    baseline = z.clone()
    baseline_mask, _ = mask_for_arm(baseline, refs, strategy, loop, 'legacy_aligned', functions, None)
    assert torch.equal(baseline, existing)
    assert torch.equal(baseline_mask, existing_mask)
    saved = [r.clone() for r in refs[0]]
    phase = torch.full((1, 2, 1, 2, 3), 99.)
    for mode in ['exact_last', 'phase_last']:
        actual = z.clone()
        mask, _ = mask_for_arm(actual, refs, strategy, loop, mode, functions, phase)
        start_length = 1 if loop == 0 else 5
        assert torch.equal(actual[:, :, :start_length], existing[:, :, :start_length])
        assert torch.equal(actual[:, :, start_length:-1], z[:, :, start_length:-1])
        target = phase if mode == 'phase_last' else refs[0][loop+1].unsqueeze(0)
        assert torch.equal(actual[:, :, -1:], target)
        assert mask[0, -1] == 0
    assert all(torch.equal(a, b) for a, b in zip(saved, refs[0]))


def test_completed_arm_resume_rejects_changed_artifacts(tmp_path):
    from semantic_transmission.artifacts import sha256, write_json
    from semantic_transmission.receiver_alignment_experiment import complete_arm
    assert not complete_arm(tmp_path, 'frozen')
    path = tmp_path/'frames.bin'
    path.write_bytes(b'receiver generated pixels')
    write_json(tmp_path/'result.json', dict(signature='frozen', status='GENERATED', artifacts={'frames.bin': sha256(path)}))
    assert complete_arm(tmp_path, 'frozen')
    with pytest.raises(ValueError, match='incompatible'):
        complete_arm(tmp_path, 'other')
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        complete_arm(tmp_path, 'frozen')


def test_command_resumes_a_json_round_tripped_protocol(tmp_path, monkeypatch):
    import sys
    from semantic_transmission import receiver_alignment_experiment as experiment
    monkeypatch.setattr(sys, 'argv', ['experiment', '--output', str(tmp_path), '--stage', 'generate'])
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setattr(experiment, 'cases', lambda repo: [])
    # Real configs contain tuples, while JSON reloads them as lists.
    monkeypatch.setattr(experiment, 'make_protocol', lambda repo, cases: (None, {'image_size': (320, 576)}))
    calls = []
    monkeypatch.setattr(experiment, 'generate', lambda *args: calls.append(args))
    experiment.main()
    experiment.main()
    assert len(calls) == 2
    assert calls[0][-1] == calls[1][-1]


def test_temporal_evaluation_does_not_reward_freezing_as_motion_accuracy():
    import numpy as np
    from semantic_transmission.receiver_alignment_evaluation import temporal_scores
    source = np.zeros((4, 160, 288, 3), dtype=np.uint8)
    for i in range(4):
        source[i, 40:100, 60+i*4:120+i*4] = 255
    perfect = temporal_scores(source, source, 2)
    frozen = temporal_scores(source, np.repeat(source[:1], 4, axis=0), 2)
    assert perfect['flow_error_mean'] == 0
    assert frozen['generated_seam_mad'] == 0
    assert frozen['flow_error_mean'] > .1
