import pytest
from semantic_transmission.qwen35_oracle_diagnostic import witnessed, coverage, oracle, reject_overlap


def test_temporal_fact_needs_distinct_ordered_witnesses():
    assert not witnessed([[1],[2]], [0,1,6,7])
    assert not witnessed([[2],[1]], [0,1,2,7])
    assert not witnessed([[1],[1]], [0,1,2,7])
    assert not witnessed([], [0,1,2,7])
    assert witnessed([[0,1],[2,3,4,5]], [0,4,6,7])
    assert not witnessed([[0],[],[7]], [0,1,2,7])


def test_oracle_preserves_transient_and_reversal_with_same_budget():
    row={'facts':[{'id':'transient','kind':'event','evidence_groups':[[0],[3],[7]]},
                  {'id':'reverse','kind':'event','evidence_groups':[[3],[5],[7]]}]}
    assert coverage(row,[0,1,6,7])['event_hits']==0
    assert oracle(row)['keys']==[0,3,5,7]
    assert oracle(row)['coverage']['event_hits']==2
    assert oracle(row)['subsets_checked']==15
    assert oracle(row,False)['subsets_checked']==70


def test_endpoint_budget_ceiling_can_be_lower_than_unrestricted():
    row={'facts':[{'id':'sequence','kind':'event','evidence_groups':[[1],[2],[3],[4]]}]}
    assert oracle(row)['coverage']['event_hits']==0
    assert oracle(row,False)['coverage']['event_hits']==1


@pytest.mark.parametrize('case', ['id','hash','protected_id','protected_hash'])
def test_source_leakage_rejected_even_for_renamed_duplicate(case):
    source={'source_id':'new','source_sha256':'newhash','split':'reserve_unreviewed'}
    prior=[{'source_id':'old','source_sha256':'oldhash','split':'train'}]
    if case=='id': source['source_id']='old'
    if case=='hash': source['source_sha256']='oldhash'
    ids={'new'} if case=='protected_id' else set()
    hashes={'newhash'} if case=='protected_hash' else set()
    with pytest.raises(ValueError): reject_overlap(source,prior,ids,hashes)


def test_disjoint_reserve_is_permitted():
    reject_overlap({'source_id':'new','source_sha256':'newhash','split':'reserve_unreviewed'},[],set(),set())
