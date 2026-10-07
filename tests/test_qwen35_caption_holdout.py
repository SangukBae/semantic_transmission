import copy
import pytest
from semantic_transmission.qwen35_caption_holdout import input_row,validate_record
from evaluate_qwen35_caption_holdout import validate_source_labels,validate_cross_audit,gate,bootstrap


@pytest.mark.parametrize('field',['keys','source_hashes','input_tokens','grid','missing_condition'])
def test_fresh_comparison_rejects_changed_or_missing_inputs(field):
    row=dict(id='h',keys=[0,2,5,7],frames=[dict(sha256=str(i)) for i in range(8)])
    c=dict(keys=row['keys'],source_hashes=['0','2','5','7'],input_tokens=100,grid=[[1,12,12]]*4)
    rec=dict(id='h',protocol_sha256='p',cells={a:copy.deepcopy(c) for a in ['initial','caption']})
    validate_record(row,rec,'p')
    if field=='missing_condition':del rec['cells']['caption']
    elif field=='input_tokens':rec['cells']['caption'][field]+=1
    else:rec['cells']['caption'][field]=[]
    with pytest.raises(ValueError):validate_record(row,rec,'p')


def test_reference_and_audit_targets_never_enter_inference_row():
    row=dict(id='h',source_id='s',fps=24,frames=[],keys=[0,2,5,7],caption='secret',facts=['secret'],claims=['secret'])
    assert set(input_row(row))=={'id','source_id','fps','frames','keys'}


@pytest.mark.parametrize('mutation',['unseen_frame','unviewed_sheet','missing_fact','single_frame_motion','missing_row'])
def test_reference_requires_observable_evidence_before_generation(mutation):
    packet=dict(records=[dict(id='h',sheet='s.png',frames=[dict(index=i) for i in [0,24,60,84]])])
    facts=[dict(id=str(i),text='A visible fact.',kind='state',full_credit='Visible fact.',reason='Observed.',source_indices=[0]) for i in range(3)]
    row=dict(id='h',facts=facts,uncertainties='No additional identity established.',camera_review='Uncertain.')
    doc=dict(records=[row],viewed_sheet_paths=['s.png'])
    validate_source_labels(packet,doc)
    if mutation=='unseen_frame':facts[0]['source_indices']=[12]
    elif mutation=='unviewed_sheet':doc['viewed_sheet_paths']=[]
    elif mutation=='missing_fact':facts.pop()
    elif mutation=='single_frame_motion':facts[0]['kind']='action'
    else:doc['records']=[]
    with pytest.raises(ValueError):validate_source_labels(packet,doc)


def test_gate_cannot_trade_lost_actions_for_fewer_errors():
    initial=dict(fact_hits=10,action_hits=5,error_windows=3,generation_failures=0)
    caption=dict(fact_hits=10,action_hits=4,error_windows=1,generation_failures=0)
    assert gate(caption,initial)==dict(noninferior=False,strict_gain=True)
    caption['action_hits']=5
    assert all(gate(caption,initial).values())


def test_bootstrap_identical_paired_sources_have_zero_difference():
    counts=dict(fact_hits=3,facts=4,action_hits=1,actions=2,error_windows=0,windows=2)
    result=bootstrap({str(i):{a:counts.copy() for a in ['initial','caption']} for i in range(3)})
    assert all(x['delta_percentage_point_interval_95']==[0,0] for x in result.values())


def cross_review():
    packet=dict(records=[dict(id='r',sheet='x.png',frames=[dict(index=10),dict(index=30)])])
    ref=dict(annotator=dict(identity='author'),records=[dict(id='r',facts=[dict(id='f1'),dict(id='f2')])])
    doc=dict(annotator=dict(identity='auditor',prior_outputs_seen=False),source_only=True,viewed_sheet_paths=['x.png'],records=[dict(id='r',facts=[dict(id='f1',status='supported',reason='Visible.',source_indices=[10]),dict(id='f2',status='partial',reason='Ambiguous.',source_indices=[30])])])
    return packet,ref,doc


def test_primary_reference_uses_only_source_supported_facts():
    packet,ref,doc=cross_review()
    assert validate_cross_audit(packet,ref,doc)['r']['facts']==[dict(id='f1')]


@pytest.mark.parametrize('mutation',['same_author','seen_outputs','missing_fact','unseen_frame'])
def test_cross_reference_requires_separate_complete_source_audit(mutation):
    packet,ref,doc=cross_review()
    if mutation=='same_author':doc['annotator']['identity']='author'
    elif mutation=='seen_outputs':doc['annotator']['prior_outputs_seen']=True
    elif mutation=='missing_fact':doc['records'][0]['facts'].pop()
    else:doc['records'][0]['facts'][0]['source_indices']=[99]
    with pytest.raises(ValueError):validate_cross_audit(packet,ref,doc)
