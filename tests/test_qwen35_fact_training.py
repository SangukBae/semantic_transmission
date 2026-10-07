import copy
import pytest

from semantic_transmission.qwen35_fact_training import covered,validate_fact,balanced_order,supported_target,eligible


def test_temporal_fact_needs_all_sides_not_union():
    fact={'text':'A hand removes a cup.','evidence_groups':[[2,3],[5,6]],'kind':'transition'}
    validate_fact(fact)
    assert not covered(fact,[0,1,2,7])
    assert not covered(fact,[0,4,5,7])
    assert covered(fact,[0,2,5,7])
    bad=copy.deepcopy(fact);bad['evidence_groups'][1]=[]
    with pytest.raises(ValueError):validate_fact(bad)


def test_caption_supervision_does_not_invent_missing_evidence():
    row={'facts':[{'text':'A shelf is visible.','evidence_groups':[list(range(8))],'kind':'state'},
                  {'text':'A hand grasps the cup.','evidence_groups':[[3]],'kind':'state'},
                  {'text':'The cup is removed.','evidence_groups':[[3],[5,6]],'kind':'transition'}]}
    text,facts=supported_target(row,[0,1,5,7])
    assert text=='A shelf is visible.' and len(facts)==1
    text,facts=supported_target(row,[0,3,5,7])
    assert len(facts)==3 and 'removed' in text


def test_balanced_sampling_keeps_every_original_and_is_reproducible():
    rows=[{'id':str(i),'events':[{}] if i<2 else []} for i in range(8)]
    a=balanced_order(rows,7);b=balanced_order(rows,7)
    assert a==b and {r['id'] for r in a}=={r['id'] for r in rows}
    assert len(a)==12 and sum(bool(r['events']) for r in a)==6


def test_low_language_loss_cannot_override_lost_evidence():
    initial={'event_hits':9,'fact_evidence_hits':23}
    assert not eligible({'event_hits':8,'fact_evidence_hits':24,'fixed_evidence_caption_ce':0.01},initial)
    assert not eligible({'event_hits':10,'fact_evidence_hits':22},initial)
    assert eligible({'event_hits':9,'fact_evidence_hits':23},initial)


def test_fact_balancing_does_not_reward_long_easy_fact():
    torch=pytest.importorskip('torch')
    from semantic_transmission.qwen35_fact_training import balanced_token_loss
    facts=[{'text':'abc','kind':'state'},{'text':'x','kind':'transition'}]
    losses=torch.tensor([1.,1.,1.,4.,2.],requires_grad=True)
    loss=balanced_token_loss(losses,[(0,1),(1,2),(2,3),(4,5),(5,15)],facts)
    assert float(loss.detach())==pytest.approx((1+2*4)/3+0.05*2)
    loss.backward()
    assert losses.grad[3]>losses.grad[:3].sum()


def test_fact_evidence_loss_pushes_probability_towards_complete_pair():
    torch=pytest.importorskip('torch')
    from semantic_transmission.qwen35_multitask import TemporalSelector
    from semantic_transmission.qwen35_fact_training import evidence_penalty
    h=TemporalSelector(width=8,hidden=4)
    logits=torch.zeros(len(h.subsets),requires_grad=True)
    pred={'candidate_logits':torch.zeros(8,2,requires_grad=True),'probabilities':logits.softmax(0)}
    row={'frames':list(range(8)),'events':[], 'facts':[{'text':'A cup is removed.','evidence':[2,5],
           'evidence_groups':[[2],[5]],'kind':'transition'}]}
    evidence_penalty(h,pred,row).backward()
    correct=[i for i,s in enumerate(h.subsets.tolist()) if 2 in s and 5 in s][0]
    assert logits.grad[correct]<0
    assert all(logits.grad[i]>0 for i in range(len(logits)) if i!=correct)
