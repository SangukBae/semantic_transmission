import copy
import json
from pathlib import Path
import pytest
import torch
from semantic_transmission.qwen35_grounded_data import split_reserve,validate_grounded
from semantic_transmission.qwen35_grounded_training import support_sequences,evidence_objective,grounding_prompt,eligible_facts,epoch_order
from semantic_transmission.qwen35_reasoning_probe import parse_observations
from evaluate_qwen35_reasoning_probe import normalized_notes
from test_qwen35_astra_transfer import sample


def test_original_split_deterministic_and_disjoint():
    reserve=[dict(id=f'webvid_{i}') for i in range(18)]+[dict(id='tvsum_0')]
    parts=split_reserve(reserve)
    assert parts==split_reserve(list(reversed(reserve)))
    assert [len(parts[s]) for s in ['train','validation','test']]==[12,3,4]
    assert len({r['id'] for rows in parts.values() for r in rows})==19
    assert dict(id='tvsum_0') in parts['train']


def test_grounded_does_not_turn_uncertain_motion_into_negative():
    r=sample();r['motion_review']=dict(camera='uncertain',objects='moving',reason='View is ambiguous.',evidence=[0,7])
    validate_grounded(r,dict(id='row',keys=[0,2,5,7]))
    assert r.get('confirmed_no_transition_indices',[])==[]
    r['motion_review']['evidence']=[3]
    with pytest.raises(ValueError):validate_grounded(r,dict(id='row',keys=[0,2,5,7]))


def test_ordered_paths_admit_alternatives_but_not_equal_or_reversed():
    f=dict(evidence_groups=[[0,2],[2,5,7]])
    assert support_sequences(f,[0,2,5,7])==[(0,1),(0,2),(0,3),(1,2),(1,3)]
    with pytest.raises(ValueError):support_sequences(dict(evidence_groups=[[7],[2]]),[0,2,5,7])


def test_evidence_loss_rewards_supported_alternatives_and_has_gradients():
    before=torch.tensor([2.,0.,-1.,-2.],requires_grad=True)
    after=torch.tensor([-2.,-1.,1.,2.],requires_grad=True)
    paths=[(0,2),(0,3)];good=evidence_objective([before,after],paths)
    reverse=evidence_objective([after,before],paths)
    assert good<reverse
    good.backward();assert torch.isfinite(before.grad).all() and before.grad.abs().sum()>0
    alternative=evidence_objective([before.detach(),torch.tensor([-2.,-1.,2.,1.])],paths)
    assert torch.allclose(good,alternative)


def test_state_loss_excludes_wrong_frames_and_skips_uninformative_states():
    f=dict(text='A hand is visible.',kind='state',evidence_groups=[[0,2,5,7]])
    assert eligible_facts(dict(facts=[f],keys=[0,2,5,7]))==[]
    assert evidence_objective([torch.tensor([3.,0.,0.,0.])],[(0,)])<evidence_objective([torch.tensor([0.,0.,0.,3.])],[(0,)])


def test_prompt_does_not_contain_ground_truth_evidence_indices():
    f=dict(text='A hand rises.',evidence_groups=[[0],[7]])
    alternative=dict(f,evidence_groups=[[2],[5]])
    assert grounding_prompt(f,[0,2,5,7],'before')==grounding_prompt(alternative,[0,2,5,7],'before')


def test_arms_share_reproducible_example_order():
    rows=[dict(id=str(i)) for i in range(30)]
    assert epoch_order(rows,0)==epoch_order(rows,0)
    assert epoch_order(rows,0)!=epoch_order(rows,1)


def note():
    return dict(observations=[dict(id='o1',fact='A hand is visible.',evidence=[0],uncertain=False)],
        changes=[dict(fact='The hand rises.',before=[0],after=[7],cause='object_motion')],uncertainties=[])


def test_notes_reject_unseen_evidence_and_reversed_changes():
    n=note();assert parse_observations(json.dumps(n),[0,2,5,7])[1] is None
    n['observations'][0]['evidence']=[1];assert parse_observations(json.dumps(n),[0,2,5,7])[1]
    n=note();n['changes'][0]['before']=[7];assert parse_observations(json.dumps(n),[0,2,5,7])[1]


def test_secondary_format_audit_only_normalizes_literal_i_prefix():
    n=note();n['observations'][0]['evidence']=['i0']
    raw=json.dumps(n);assert parse_observations(raw,[0,2,5,7])[1]
    assert normalized_notes(raw,[0,2,5,7])[1] is None
    n['observations'][0]['evidence']=['i1'];assert normalized_notes(json.dumps(n),[0,2,5,7])[1]
    assert normalized_notes(raw[:-3],[0,2,5,7])[1]


def test_training_verification_does_not_open_test_files(tmp_path,monkeypatch):
    from semantic_transmission import qwen35_grounded_data as mod
    touched=[]
    docs={
      'input_freeze.json':dict(files={'train_inputs.json':'h','validation_inputs.json':'h','test_inputs.json':'h','teacher/packet_A.json':'h'},source_frames={},sheets={}),
      'data_freeze.json':dict(input_freeze_sha256='h',files={'train.json':'h','validation.json':'h','test.json':'h','teacher/labels_A.json':'h'}),
      'protocol.json':{},'train.json':{'records':[]},'validation.json':{'records':[]}}
    def read(p):
        name=str(Path(p).relative_to(tmp_path));touched.append(name);return docs[name]
    def hashed(p):touched.append(str(Path(p).relative_to(tmp_path)));return 'h'
    monkeypatch.setattr(mod,'read_json',read);monkeypatch.setattr(mod,'sha256',hashed)
    mod.verify_data(tmp_path,training_only=True)
    assert all('test' not in x and not x.startswith('teacher') for x in touched)
