import copy
import json
from pathlib import Path
import pytest
from semantic_transmission.qwen35_caption_repair import (
    assert_disjoint, fixed_prediction, legacy_target, order_for, supported_facts,
)
from semantic_transmission.qwen35_oracle_diagnostic import witnessed


def test_ordered_action_evidence_cannot_use_one_frame_twice():
    assert not witnessed([[2],[2]],[0,2,5,7])
    assert witnessed([[1,2],[2,3]],[0,2,3,7])


def test_target_drops_action_without_distinct_chronological_witnesses():
    row={'facts':[{'id':'s','text':'A dog.','kind':'state','evidence_groups':[[0]]},
                  {'id':'e','text':'The dog jumps.','kind':'transition','evidence_groups':[[2],[2]]}]}
    target=legacy_target(row)
    assert target['caption']=='A dog.'
    assert target['qa_answer']==''
    assert target['target_fact_ids']==['s']


@pytest.mark.parametrize('field',['source_id','source_sha256'])
def test_source_identity_and_hash_independently_block_leakage(field):
    a=dict(id='a',split='train',source_id='one',source_sha256='111')
    b=dict(id='b',split='test',source_id='two',source_sha256='222')
    assert_disjoint([a,b])
    b[field]=a[field]
    with pytest.raises(ValueError,match='crosses'):assert_disjoint([a,b])


def test_training_order_is_matched_reproducible_and_excludes_eval():
    rows=[dict(id='a',split='train',origin='legacy_replay'),dict(id='b',split='train',origin='new_action')]
    settings=dict(seed=42,new_train_repeats=2)
    order=order_for(rows,settings,0)
    assert order==order_for(rows,settings,0)
    assert [r['id'] for r in order].count('b')==2
    with pytest.raises(ValueError,match='Non-training'):order_for([dict(rows[0],split='test')],settings,0)


def test_fixed_keys_bypass_selector_and_preserve_real_tokens():
    import torch
    from semantic_transmission.qwen35_multitask import TemporalSelector
    head=TemporalSelector(width=6,hidden=4).requires_grad_(False)
    tokens=torch.randn(8,3,6)
    pred=fixed_prediction([1,3,5,7],'cpu')
    selected=head.select_tokens(tokens,pred)
    torch.testing.assert_close(selected,tokens[[1,3,5,7]])
    parameter=torch.nn.Parameter(torch.ones(6))
    (selected*parameter).sum().backward()
    assert parameter.grad is not None
    assert all(p.grad is None for p in head.parameters())
    assert not pred['matrix'].requires_grad


@pytest.mark.parametrize('keys',[[0,1,1,7],[0,1,7],[7,5,2,0],[0,2,5,8]])
def test_invalid_keys_rejected(keys):
    with pytest.raises(ValueError):fixed_prediction(keys,'cpu')


def test_source_reference_targets_all_have_fixed_key_evidence():
    settings=json.loads((Path(__file__).parents[1]/'configs/qwen35_caption_repair_v1.json').read_text())
    assert len(settings['records'])==22
    for row in settings['records']:
        assert len(supported_facts(row,row['keys']))==len(row['facts']),row['id']
        assert row['caption']==' '.join(f['text'] for f in row['facts'])
        assert row['qa_answer']==' '.join(f['text'] for f in row['facts'] if f['kind']=='event')
