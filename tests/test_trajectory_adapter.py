import numpy as np
import torch
from semantic_transmission.trajectory_adapter import latent_positions,condition_maps,TrajectoryAdapter


def test_temporal_alignment_does_not_cross_seventeen_frame_blocks():
    assert latent_positions(17)==[0,4,8,12,16]
    assert len(latent_positions(33))==9
    assert latent_positions(34)[5:]==[17,21,25,29,33]


def test_no_received_track_cannot_create_unconditional_residual():
    a=TrajectoryAdapter(hidden=8,width=4)
    torch.nn.init.ones_(a.projection.bias)
    empty=torch.from_numpy(condition_maps(None,[None,None],list(range(33)),20,36))[None]
    assert torch.count_nonzero(a(empty))==0


def test_zero_initialization_is_exact_and_training_reaches_projection():
    a=TrajectoryAdapter(hidden=8,width=4)
    c=torch.ones(1,10,9,20,36)
    y=a(c);assert y.shape==(1,1620,8) and torch.count_nonzero(y)==0
    (y-1).square().mean().backward()
    assert a.projection.weight.grad.abs().sum()>0


def test_native_context_keeps_five_reference_slots_before_current_tracks():
    from semantic_transmission.trajectory_adapter_runtime import native_condition
    tracks=dict(xy=np.tile(np.array([[[100.,80.]]]),(18,1,1)),valid=np.ones((18,1),bool),origins=np.array([0]))
    keys=np.full((2,160,288,3),128,np.uint8)
    first=native_condition(tracks,keys,18,20,36,False)
    continued=native_condition(tracks,keys,18,20,36,True)
    assert first.shape==(10,6,20,36)
    assert continued.shape==(10,11,20,36)
    assert np.count_nonzero(continued[:,:5])==0
    np.testing.assert_array_equal(continued[:,5:],first)


def test_native_absent_paid_track_is_identity_even_after_training():
    from semantic_transmission.trajectory_adapter_runtime import native_condition
    a=TrajectoryAdapter(hidden=8,width=4)
    torch.nn.init.normal_(a.projection.weight);torch.nn.init.ones_(a.projection.bias)
    c=native_condition(None,[None,None],33,40,72,True)
    assert torch.count_nonzero(a(torch.from_numpy(c)[None]))==0
