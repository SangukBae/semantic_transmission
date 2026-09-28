"""Factorial separation and matched replay are required before comparing quality."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from etri_tail_reference_probe import Replay
from etri_combined_reference_probe import Hooks
from diagnose_etri_combined_reference import CASES, validate_conditions


@pytest.mark.parametrize("crop", [False, True])
def test_rounding_disabled_and_cropping_independent_with_replayed_noise(tmp_path, crop):
    seen = []
    def append(vae, video, refs, strategies, loop, length, edit):
        seen.append(video.clone())
        torch.rand(video.shape[2]*3)
        refs[0].append(torch.zeros(4,5 if video.shape[2] == 17 else 11,1,1))
        return refs, strategies
    def apply(z, refs, strategies, loop, align):
        assert align is None
        return torch.ones(z.shape[0], z.shape[2])
    utils = SimpleNamespace(append_generated=append, apply_mask_strategy=apply,
                           parse_mask_strategy=lambda _: [(1,0,0,-1,1,0)])
    video = torch.arange(38.).reshape(1,1,38,1,1)
    device = torch.device('cpu')
    record = Replay(tmp_path/'bank', 'record')
    torch.manual_seed(2025)
    record.synchronize('reference',1,device)
    torch.rand(38*3)
    expected = torch.randn(1,4,7,1,1)
    record.synchronize('sampling',1,device,expected)
    next_expected = torch.randn(10)
    hook = Hooks(utils, Replay(tmp_path/'bank','tail17'),tmp_path/'trace.json',crop)
    torch.rand(456)
    refs, _ = hook.append(None,video,[[torch.zeros(4,1,1,1)]],[''],1,5,0)
    noise = torch.randn_like(expected)
    hook.apply(noise,refs,[''],1)
    assert torch.equal(noise,expected)
    assert torch.equal(torch.randn(10),next_expected)
    assert torch.equal(seen[0],video[:,:,-17:] if crop else video)
    active = hook.records['mask'][0]['active'][0]
    assert active['actual_target_start'] == active['requested_target_start'] == 6
    with pytest.raises(ValueError, match='disable rounding'):
        hook.apply(noise,refs,[''],1,align=5)


def fixture_traces():
    traces = {}
    for case in CASES:
        mask = {'loop':0,'align':None if case in {'combined','no_rounding'} else 5,
                'noise_shape':[1,4,7,1,1],'noise_sha256':'same','sampling_rng_sha256':{'cpu':'same'},
                'active':[{'reference':0,'reference_sha256':'key','actual_ref_start':0,
                           'requested_ref_start':0,'actual_target_start':0,'requested_target_start':0}]}
        ref = {'loop':1,'reference_rng_sha256':{'cpu':'same'},'input_pixel_shape':[1,3,38,2,2],
               'encoded_pixel_shape':[1,3,17 if case in {'combined','tail17'} else 38,2,2],
               'latent_shape':[4,5 if case in {'combined','tail17'} else 11,1,1]}
        traces[case] = {'mask':[mask,dict(copy.deepcopy(mask),loop=1)],'references':[ref]}
    return traces


@pytest.mark.parametrize('fault',[None,'noise','key','rounding','crop','reference_rng','incomplete'])
def test_factorial_validation_rejects_confounding_changes(fault):
    traces = fixture_traces()
    c = traces['combined']
    if fault == 'noise': c['mask'][1]['noise_sha256'] = 'different'
    if fault == 'key': c['mask'][1]['active'][0]['reference_sha256'] = 'different'
    if fault == 'rounding': c['mask'][1]['active'][0]['actual_target_start'] = 5
    if fault == 'crop': traces['no_rounding']['references'][0]['encoded_pixel_shape'][2] = 17
    if fault == 'reference_rng': c['references'][0]['reference_rng_sha256']['cpu'] = 'different'
    if fault == 'incomplete': c['mask'].pop()
    if fault:
        with pytest.raises(ValueError): validate_conditions(traces,3)
    else:
        validate_conditions(traces,3)
