import itertools
import math

import pytest

from semantic_transmission.sender_repairs import (
    calibrated_flow, capped_keys, distinct_samples, evidence_pairs, validate_event)


def test_event_review_protects_uncertainty_and_object_changes_without_text_regex():
    pairs=[dict(before=i,after=i+1,weight=1,origins=[dict(description='steady')]) for i in range(5)]
    reviews=[dict(classification=c,evidence='Visible source evidence.') for c in
             ('none','camera_only','object_change','scene_change','uncertain')]
    assert evidence_pairs(pairs,reviews)==pairs[2:]
    with pytest.raises(ValueError): validate_event(dict(classification='none',evidence=''))
    with pytest.raises(ValueError): evidence_pairs(pairs,reviews[:-1])


@pytest.mark.parametrize('frames',range(1,40))
def test_distinct_samples_cover_actual_clip_without_duplicates(frames):
    samples=distinct_samples(frames)
    assert len(samples)==min(4,frames)
    assert samples==sorted(set(samples))
    assert samples[0]==0 and samples[-1]==frames-1


def test_constant_speed_score_is_invariant_to_frame_sampling_and_fps():
    for fps in (12,24,30,60):
        for indices in ([0,1,2,3],[0,8,15,23],[0,10,20,30]):
            # A scene moving at six pixels/second must retain the old 10/24 s scale.
            displacements=[6*(b-a)/fps for a,b in zip(indices,indices[1:])]
            assert calibrated_flow(displacements,indices,fps)==pytest.approx(2.5)
    assert calibrated_flow([], [0],24) is None
    assert calibrated_flow([0,0],[0,1,2],24)==0
    with pytest.raises(ValueError): calibrated_flow([1],[0,0],24)
    with pytest.raises(ValueError): calibrated_flow([math.nan],[0,1],24)


def test_budget_ceiling_does_not_force_extra_static_keys():
    keys,audit=capped_keys(25,10,[],list(range(10)),[0.]*25,max_gap=12)
    assert keys==[0,12,24]
    assert audit['minimum_key_count']==3


def test_short_semantic_event_is_preserved_despite_minimum_count_objective():
    pair=dict(before=1,after=2,weight=3)
    keys,audit=capped_keys(25,6,[pair],[0,12,24],[0.]*25,max_gap=12)
    assert {0,1,2,24}.issubset(keys)
    assert max(b-a for a,b in zip(keys,keys[1:]))<=12
    assert audit['covered_pairs']==[0]


def test_infeasible_event_budget_matches_exhaustive_lexicographic_optimum():
    frames,cap,gap=9,4,4
    pairs=[dict(before=1,after=2,weight=3),dict(before=5,after=6,weight=2)]
    feasible=[]
    for count in range(2,cap+1):
        for middle in itertools.combinations(range(1,frames-1),count-2):
            keys=(0,*middle,frames-1)
            if max(b-a for a,b in zip(keys,keys[1:]))<=gap:
                score=sum(p['weight'] for p in pairs if p['before'] in keys and p['after'] in keys)
                feasible.append((score,-count))
    keys,audit=capped_keys(frames,cap,pairs,[0,4,8],[0.]*frames,max_gap=gap)
    assert (audit['weighted_pair_coverage'],-len(keys))==max(feasible)
    assert audit['omitted_pairs']
