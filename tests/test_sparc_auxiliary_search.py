import copy
import json
from pathlib import Path

import numpy as np
import pytest

from semantic_transmission import sparc_auxiliary_selection as selection


@pytest.fixture
def config():
    return json.loads((Path(__file__).parents[1]/'configs/sparc_auxiliary_search.json').read_text())


def sample(psnr=27.,lpips=.27,frames=24):
    return dict(quality=dict(psnr_db=psnr,lpips_vgg=lpips,roi_mae=10.,temporal_mae=5.),
        per_frame=dict(psnr_db=[psnr]*frames,lpips_vgg=[lpips]*frames))


def test_semantic_cut_endpoints_and_joint_removal_gap_are_preserved():
    keys=list(range(17));cuts=[8]
    mandatory=selection.protected([0,4,12,16],cuts,17)
    ranked=selection.eligible(keys,mandatory,17,cuts)
    assert {0,4,7,8,12,16}.issubset(mandatory)
    for p in selection.removal_proposals(keys,ranked,mandatory,17,cuts,[8,4,2,1]):
        selection.validate_anchors(p['indices'],17,4,cuts)
        assert mandatory.issubset(p['indices'])


def test_gap_eligibility_must_be_recomputed_after_each_deletion():
    keys=[0,2,4,6,8]
    proposals=selection.removal_proposals(keys,[2,4,6],{0,8},9,[],[3])
    assert proposals[0]['indices']==[0,4,8]


def test_uniform_control_preserves_semantic_keys_and_exact_count():
    mandatory={0,7,8,13,31}
    for count in (12,17,25,32):
        result=selection.uniform_semantic(32,count,mandatory,[8])
        assert len(result)==count and mandatory.issubset(result)
        selection.validate_anchors(result,32,4,[8])


def test_quality_loss_does_not_accumulate_against_changing_baseline(config):
    original=sample();first=sample(psnr=26.85);second=sample(psnr=26.70)
    assert selection.quality_gate(first,original,config)['passed']
    assert selection.quality_gate(second,first,config)['passed']
    assert not selection.quality_gate(second,original,config)['passed']


def test_local_damage_is_rejected_even_if_global_mean_is_acceptable(config):
    original=sample();candidate=copy.deepcopy(original)
    candidate['per_frame']['psnr_db'][:12]=[25.]*12
    result=selection.quality_gate(candidate,original,config)
    assert 'window_psnr_db' in result['reasons']


@pytest.mark.parametrize('metric',['roi_mae','temporal_mae'])
def test_motion_or_roi_regression_is_rejected(config,metric):
    original=sample();candidate=copy.deepcopy(original)
    candidate['quality'][metric]*=1.04
    assert metric in selection.quality_gate(candidate,original,config)['reasons']


def test_nan_cannot_be_accepted(config):
    original=sample();candidate=sample(lpips=float('nan'))
    with pytest.raises(ValueError,match='nonfinite'):
        selection.quality_gate(candidate,original,config)


def test_middle_only_appearance_receives_higher_visibility_priority(config):
    frames=np.zeros((5,80,144,3),np.uint8)
    frames[2,24:56,48:96]=255
    signals=selection.Signals(frames,config)
    assert signals.score(2,0,4)>signals.score(1,0,4)
    mandatory={0,4}
    for proposal in signals.relocations([0,1,3,4],mandatory,[]):
        assert mandatory.issubset(proposal['indices'])
        selection.validate_anchors(proposal['indices'],5,4,[])


def test_trial_has_its_own_signed_worker_protocol_without_changing_parent(tmp_path):
    from semantic_transmission.sparc_auxiliary_search import trial_protocol
    from semantic_transmission.webvid5 import fingerprint
    parent=dict(code={'test':'digest'},vendor_code={},output='/parent')
    parent['signature']=fingerprint(parent)
    original=copy.deepcopy(parent)
    child=trial_protocol(parent,tmp_path,[0,2,4])
    assert child['output']==str(tmp_path) and child['parent_protocol_signature']==parent['signature']
    assert child['signature']==fingerprint({k:v for k,v in child.items() if k!='signature'})
    assert parent==original
