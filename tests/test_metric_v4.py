import numpy as np
import pytest

from semantic_transmission.event_duration_metric import evaluate_observations, sfr_observed
from semantic_transmission.metric_v4_cases import FAMILIES, scene,draw,variants,truth_query
from semantic_transmission.sta_selective import CLASSES,fit,calibrate,predict


def observation(times, event_times, count=20):
    mask=np.zeros((24,32),bool);mask[8:14,8:14]=True
    tracks=[{t:{'mask':mask,'feature':np.array([1.,0.]),'center':np.array([10.,10.])} for t in times}]
    events=[{'type':kind,'time_s':t/8.,'track_id':0,'direction':0,'cell':[1,1]} for kind,t in event_times]
    return {'tracks':tracks,'events':events,'frame_count':count,'shape':[24,32]}


def test_exit_duration_and_response_delay_are_different():
    a=observation(range(8),[('exit',8)])
    b=observation(range(10),[('exit',10)])
    r=evaluate_observations(a,b)['events'][0]
    assert r['ghost_auc']==.5 and r['ghost_observed_s']==.25
    assert r['delay_s']==.25


def test_ghost_reappearance_counts_after_first_absence():
    a=observation(range(8),[('exit',8)])
    b=observation(list(range(8))+[10,11],[('exit',8),('enter',10),('exit',12)])
    r=evaluate_observations(a,b)['events'][0]
    assert r['delay_s']==0 and r['ghost_auc']==.5 and r['ghost_returns']==1


def test_censoring_and_unobservable_are_not_zero_error():
    a=observation(range(18),[('exit',18)])
    b=observation(range(20),[])
    r=evaluate_observations(a,b)['events'][0]
    assert r['ghost_status']=='right_censored' and r['ghost_auc'] is None
    assert r['delay_status']=='right_censored' and r['delay_penalty'] is None
    b['tracks']=[]
    assert evaluate_observations(a,b)['events'][0]['delay_status']=='unobservable'


def test_sfr_does_not_detect_a_constant_ghost():
    a=observation(range(8),[('exit',8)])
    b=observation(range(20),[])
    r=evaluate_observations(a,b)
    assert r['ghost_max']==1 and sfr_observed(a,b,r['entity_groups'])==0


@pytest.mark.parametrize('family',FAMILIES)
def test_new_fault_truth_matches_its_intended_target(family):
    state=scene(260914001,family);_,a=draw(state)
    for name,frames,b,meta in variants(state):
        if meta['target']=='control':
            assert np.array_equal(a['visible'],b['visible'])
            assert np.array_equal(a['centers'],b['centers'])
            continue
        r=truth_query(a,b,meta['target_object'],meta['target_event'],meta['target_frame'])
        if meta['kind']=='event_delay':assert r['delay_s']==meta['severity']
        elif meta['kind']=='never_update':assert r['delay_status']=='deadline_miss'
        elif meta['kind']=='ghost_return':assert r['ghost_auc']==.5 and r['delay_s']==0
        else:assert r['ghost_auc']==min(meta['severity']/.5,1)


def test_normal_class_and_technical_rejection_are_explicit():
    rows=[{'source_id':f'source{s}','label':label,'scores':{'x':i*10.+s*.01}} for s in range(3) for i,label in enumerate(CLASSES)]
    model=fit(rows,['x']);cal=calibrate(rows,model)
    assert predict({'x':.01},model,cal)['prediction']=='normal'
    assert predict({'x':None},model,cal)['reason']=='unobservable'
    assert predict({'x':1000.},model,cal)['prediction']=='abstain'
    assert predict({'x':.01},model,cal,observable=False)['prediction']=='abstain'


def test_other_object_event_cannot_satisfy_response_latency():
    a=observation(range(8),[('exit',8)]);b=observation(range(20),[('exit',8)])
    other=observation(range(8),[])['tracks'][0]
    other={t:{**d,'feature':np.array([0.,1.])} for t,d in other.items()}
    a['tracks'].append(other);b['tracks'].append(other)
    b['events'][0]['track_id']=1
    r=evaluate_observations(a,b)['events'][0]
    assert r['delay_status']=='deadline_miss' and r['delay_penalty']==1


def test_source_pairing_conditions_on_absent_and_source_negative():
    from semantic_transmission.metric_v4_audit import source_paired_additions
    a=observation(range(9),[]);b=observation(range(10),[])
    truth={'visible':np.arange(20)[:,None]<8,'masks':np.zeros((20,24,32),np.uint8)}
    truth['masks'][:8,8:14,8:14]=1
    r=source_paired_additions(a,b,{'entity_groups':{0:[0]}},truth)
    assert r['absent_opportunities']==12 and r['source_positive_excluded']==1
    assert r['raw_additions']==2 and r['paired_additions']==1
    assert r['paired_opportunities']==11 and r['unpairable']==0


def test_source_bootstrap_keeps_missing_class_counts_explicit():
    import runpy
    report=runpy.run_path('scripts/report_metric_v4.py')['classification']
    rows=[{'source_id':f's{i}','label':label} for i,label in enumerate(CLASSES)]
    # Add normal observations to each source; one fault class is always wrong.
    rows += [{'source_id':f's{i}','label':'normal'} for i in range(1,5)]
    predictions=[{'prediction':'normal' if r['label']=='clr' else r['label'],'reason':None} for r in rows]
    result=report(rows,predictions)
    assert result['macro_accuracy']==.8 and result['fault_macro_accuracy']==.75
    assert result['coverage']==1 and result['class_recall']['clr']==0
