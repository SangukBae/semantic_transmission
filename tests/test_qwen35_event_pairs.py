import itertools

import pytest

from semantic_transmission.qwen35_event_pairs import paired_keys,source_pairs,overlap_windows


def test_complete_pair_objective_matches_bruteforce():
    pairs=[dict(before=2,after=3,weight=3.),dict(before=5,after=7,weight=2.)]
    keys,stats=paired_keys(10,5,pairs,[0,2,4,7,9],[0.]*10,max_gap=4)
    def value(k):return sum(p['weight'] for p in pairs if p['before'] in k and p['after'] in k)
    possibilities=[[0,*mid,9] for mid in itertools.combinations(range(1,9),3)]
    possible=[k for k in possibilities if max(b-a for a,b in zip(k,k[1:]))<=4]
    assert value(keys)==max(map(value,possible))==stats['weighted_pair_coverage']
    optimal=[k for k in possible if value(k)==value(keys)]
    assert len(set(keys)&{0,2,4,7,9})==max(len(set(k)&{0,2,4,7,9}) for k in optimal)


def test_one_side_does_not_receive_pair_credit_and_short_pair_is_allowed():
    keys,stats=paired_keys(8,4,[dict(before=3,after=4,weight=3)], [0,2,5,7],[0.]*8,max_gap=4)
    assert keys==[0,3,4,7] and stats['covered_pairs']==[0]


def test_duplicates_are_capped_and_bad_source_pairs_rejected():
    row=dict(window=0,indices=[0,4,8],result=dict(events=[dict(before=0,after=4,kind='pose',description='a'),
        dict(before=0,after=4,kind='exit',description='b')]))
    pairs=source_pairs([row]);assert len(pairs)==1 and pairs[0]['weight']==3 and len(pairs[0]['origins'])==2
    row['result']['events'][0]['before']=1
    with pytest.raises(ValueError,match='source evidence'):source_pairs([row])


def test_infeasible_budget_does_not_silently_grow():
    with pytest.raises(ValueError,match='infeasible'):paired_keys(100,4,[],[],[0.]*100)


def test_overlap_probes_cross_every_original_internal_boundary():
    windows=overlap_windows(1440)
    assert len(windows)==29
    for center,ids in zip(range(48,1440,48),windows):
        assert ids[0]==center-24 and ids[-1]==center+24 and len(ids)==13
