from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('oracle_review', Path(__file__).parents[1]/'scripts/evaluate_qwen35_oracle_diagnostic.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def example():
    packet={'records':[{'label':'x','caption_sha256':'digest','caption':'The car leaves.',
                        'facts':[{'id':'f0'},{'id':'f1'}]}]}
    review={'records':[{'label':'x','caption_sha256':'digest','fact_hits':{'f0':True,'f1':False},
                       'partial_fact_ids':['f1'],'rationale':'The car stays in the frame.',
                       'errors':[{'quote':'leaves','reason':'It is still visible.','source_indices':[0,7]}]}]}
    return packet,review


def test_complete_quote_bound_review_is_valid():
    p,r=example()
    assert MODULE.validate_review(p,r)['x']['fact_hits']=={'f0':True,'f1':False}


@pytest.mark.parametrize('mutation',['missing_row','duplicate_row','stale_caption','missing_fact','fabricated_quote','overlap_partial'])
def test_unbound_or_inconsistent_review_rejected(mutation):
    p,r=example()
    if mutation=='missing_row':r['records']=[]
    elif mutation=='duplicate_row':r['records'].append(deepcopy(r['records'][0]))
    elif mutation=='stale_caption':r['records'][0]['caption_sha256']='changed'
    elif mutation=='missing_fact':del r['records'][0]['fact_hits']['f1']
    elif mutation=='fabricated_quote':r['records'][0]['errors'][0]['quote']='flies'
    else:r['records'][0]['partial_fact_ids']=['f0']
    with pytest.raises(ValueError):MODULE.validate_review(p,r)


@pytest.mark.parametrize('mutation',['keys','caption','missing'])
def test_masked_review_cannot_be_assigned_to_another_condition(mutation):
    p={'records':[{'label':'x','row_id':'row','caption_sha256':MODULE.digest('caption'), 'keys':[0,2,4,7]}]}
    mapping={'x':[{'row_id':'row','cell':'oracle_joint'}]}
    generated={('row','oracle_joint'):{'caption':'caption','keys':[0,2,4,7]}}
    MODULE.validate_mapping(p,mapping,generated)
    if mutation=='keys':generated[('row','oracle_joint')]['keys']=[0,1,2,7]
    elif mutation=='caption':generated[('row','oracle_joint')]['caption']='other'
    else:generated[('row','learned_joint')]={'caption':'caption','keys':[0,2,4,7]}
    with pytest.raises(ValueError):MODULE.validate_mapping(p,mapping,generated)
