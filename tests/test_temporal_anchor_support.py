import numpy as np
import pytest
import torch

from semantic_transmission.structure_anchors import anchor_slots
from semantic_transmission.frequency_anchors import correction
from semantic_transmission.temporal_anchor_support import experiment_arms, brightness_diagnostic


def test_support_intervention_and_constant_residual_coefficient_control():
    plan = anchor_slots(384, list(range(0, 384, 8)), [0, 110])
    arms = experiment_arms(plan)
    assert [a['positions'] for a in arms] == [46, 111, 46, 111]
    assert arms[0]['slots'] == plan['slots']
    assert arms[0]['strength'] == arms[1]['strength'] == .7
    assert arms[2]['strength'] == arms[3]['strength']
    z = torch.zeros(1, 4, 113, 8, 8)
    values = []
    for arm in arms:
        delta = correction(z, z, z + 1, torch.tensor([.5]), torch.tensor([.1]),
                           arm['slots'], arm['strength'], arm['fraction'])
        assert torch.count_nonzero(delta[:, :, [0, 110]]) == 0
        support = torch.nonzero(delta.abs().sum(dim=(0, 1, 3, 4))).flatten().tolist()
        assert support == arm['slots']
        values.append(delta.sum())
    torch.testing.assert_close(values[0], values[3])
    assert values[1] > values[0] > values[2]


@pytest.mark.parametrize('slots', [[0, 2], [2, 2], [-1, 2], [113]])
def test_invalid_support_is_rejected(slots):
    plan = anchor_slots(384, list(range(0, 384, 8)), [0, 110])
    plan['slots'] = slots
    with pytest.raises(ValueError):
        experiment_arms(plan)


def test_brightness_diagnostic_removes_true_brightness_change_and_finds_flash():
    source = np.stack([np.full((12, 16, 3), 30 + i, dtype=np.uint8) for i in range(40)])
    generated = source.copy()
    generated[35] += 40
    original = np.abs(np.diff(source.astype(float), axis=0)).mean(axis=(1, 2, 3)).tolist()
    actual = np.abs(np.diff(generated.astype(float), axis=0)).mean(axis=(1, 2, 3)).tolist()
    temporal = dict(per_transition=dict(source_mad=original, generated_mad=actual))
    result = brightness_diagnostic(source, generated, temporal)
    assert result['target_luma_impulse'] == 40
    assert result['max_luma_impulse_frame'] == 35
    assert result['max_source_adjusted_luma_step'] == 40
    assert result['transitions_above_mad10'] == 2
    same = brightness_diagnostic(source, source, dict(per_transition=dict(source_mad=original, generated_mad=original)))
    assert same['max_source_adjusted_luma_step'] == same['target_luma_impulse'] == 0
