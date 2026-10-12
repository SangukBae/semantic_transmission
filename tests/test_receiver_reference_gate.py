import numpy as np
import pytest
import torch

from semantic_transmission import receiver_reference_gate as gate


def test_source_identity_blocks_different_normalizations_of_same_original():
    cases=[dict(split='train',source_id='same',source_sha256='a'),dict(split='test',source_id='same',source_sha256='b')]
    with pytest.raises(ValueError,match='original videos'):gate.assert_source_split(cases)
    cases[1]['source_id']='other';gate.assert_source_split(cases)
    cases[1]['source_sha256']='a'
    with pytest.raises(ValueError,match='bytes'):gate.assert_source_split(cases)


def test_zero_gate_is_identity_and_nonzero_only_changes_older_history_slots():
    mask=torch.tensor([[0.,0.,0.,0.,0.,1.,1.,1.,1.,1.,0.,1.]])
    record=dict(loop=1,operations=[dict(target=0,length=5),dict(target=10,length=1)])
    original=mask.clone()
    actual,slots=gate.relax_history(mask,record,0.)
    assert torch.equal(actual,original) and torch.equal(mask,original)
    actual,slots=gate.relax_history(mask,record,.35)
    assert slots==[0,1,2,3]
    assert torch.all(actual[:,:4]==.35)
    assert torch.equal(actual[:,4:],mask[:,4:])
    assert torch.equal(gate.relax_history(mask,dict(record,loop=0),.35)[0],mask)
    with pytest.raises(ValueError):gate.relax_history(mask,record,float('nan'))


def test_constant_receiver_pixels_have_finite_zero_discrepancy_features():
    image=np.full((32,48,3),100,np.uint8)
    values=gate.features(np.repeat(image[None],17,axis=0),image,image,[24,48],24)
    assert len(values)==len(gate.FEATURES)
    assert values[:12]==pytest.approx([0.]*12,abs=1e-12)
    assert values[-2:]==[1.,2.]
    with pytest.raises(ValueError):gate.features(np.repeat(image[None],16,axis=0),image,image,[24,24],24)


def test_utility_penalizes_quality_regression_even_when_motion_improves():
    before=dict(flow_error=.1,psnr_db=20,lpips_vgg=.3)
    assert gate.utility(before,dict(before,flow_error=.08))==pytest.approx(.2)
    assert gate.utility(before,dict(before,flow_error=.08,psnr_db=19))<0
    assert gate.utility(before,dict(before,flow_error=.08,lpips_vgg=.32))<0


def test_training_reproducibility_and_predict_does_not_advance_rng():
    records=[]
    for i in range(12):
        records.append(dict(features=[i/12]+[0.]*(len(gate.FEATURES)-1),utility=(i-6)/12,source_id=str(i%3)))
    a,loss=gate.fit(records,steps=150)
    b,_=gate.fit(records,steps=150)
    assert a['parameter_count']==385 and loss<.01
    assert all(torch.equal(a['state_dict'][k],b['state_dict'][k]) for k in a['state_dict'])
    state=torch.get_rng_state().clone()
    assert gate.predict(a,records[0]['features'])[1]==0.
    assert gate.predict(a,records[-1]['features'])[1]==gate.EDIT_RATIO
    assert torch.equal(state,torch.get_rng_state())
