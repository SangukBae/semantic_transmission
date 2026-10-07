import copy
from collections import Counter
import pytest
from semantic_transmission.qwen35_grounded_eval import cell_order,validate_record,ARMS
from evaluate_qwen35_grounded import comparison_gates,validate_reference_audit,validate_grounded_review,apply_adjudication
from evaluate_qwen35_astra_transfer import digest


def test_each_condition_has_each_order_position():
    for position in range(3):assert Counter(cell_order(i)[position] for i in range(3))==Counter(ARMS)


def generated():
    row=dict(id='r',keys=[0,2,5,7],frames=[dict(sha256=f'h{i}') for i in range(8)])
    cell=dict(keys=row['keys'],source_hashes=['h0','h2','h5','h7'],input_tokens=1200,grid=[[1,12,12]]*4)
    record=dict(id='r',protocol_sha256='p',cells={a:copy.deepcopy(cell) for a in ARMS})
    return row,record


@pytest.mark.parametrize('field,value',[('keys',[0,1,5,7]),('source_hashes',['h0']*4),('input_tokens',1201),('grid',[[1,12,12]]*3)])
def test_generation_rejects_unmatched_input(field,value):
    row,record=generated();validate_record(row,record,'p')
    record['cells']['grounded'][field]=value
    with pytest.raises(ValueError):validate_record(row,record,'p')


def test_generation_cannot_omit_a_condition():
    row,record=generated();del record['cells']['initial']
    with pytest.raises(ValueError):validate_record(row,record,'p')


def counts():
    return {s:{a:dict(fact_hits=5,action_hits=3,error_windows=1,generation_failures=0) for a in ARMS} for s in ['validation','test']}


def test_gate_requires_gain_against_both_controls_on_both_splits():
    c=counts();assert not comparison_gates(c)['grounded']['passed']
    for s in c:c[s]['grounded']['fact_hits']=6
    assert comparison_gates(c)['grounded']['passed']
    c['test']['caption']['fact_hits']=7
    assert not comparison_gates(c)['grounded']['passed']


@pytest.mark.parametrize('field,value',[('action_hits',2),('error_windows',2),('generation_failures',1)])
def test_fact_gain_does_not_hide_other_regression(field,value):
    c=counts()
    for s in c:c[s]['grounded']['fact_hits']=7
    c['test']['grounded'][field]=value
    assert not comparison_gates(c)['grounded']['passed']


def test_reference_audit_requires_each_fact_and_visible_source():
    packet=dict(records=[dict(id='r',sheet='a.png',frames=[dict(index=10)],facts=[dict(id='f')])])
    d=dict(viewed_sheet_paths=['a.png'],records=[dict(id='r',facts=[dict(id='f',status='supported',reason='Visible.',source_indices=[10])])])
    validate_reference_audit(packet,d)
    d['records'][0]['facts'][0]['source_indices']=[12]
    with pytest.raises(ValueError):validate_reference_audit(packet,d)


def test_positive_caption_hit_requires_exact_quote_and_error_category():
    caption='The dog sits.';r=dict(label='g0',caption=caption,caption_sha256=digest(caption),facts=[dict(id='f')],source_frames=[dict(index=10)],source_sheet='a.png')
    judged=dict(label='g0',caption_sha256=r['caption_sha256'],fact_hits={'f':True},partial_fact_ids=[],errors=[],rationale='Sitting visible.',fact_evidence={'f':dict(quote='dog sits',rationale='Action specified.')})
    doc=dict(records=[judged],viewed_sheet_paths=['a.png']);validate_grounded_review(dict(records=[r]),doc)
    judged['fact_evidence']['f']['quote']='dog runs'
    with pytest.raises(ValueError):validate_grounded_review(dict(records=[r]),doc)
    judged['fact_evidence']['f']['quote']='dog sits';judged['errors']=[dict(quote='sits',reason='Wrong action.',source_indices=[10],kind='maybe')]
    with pytest.raises(ValueError):validate_grounded_review(dict(records=[r]),doc)


def test_adjudication_preserves_original_and_requires_source_and_exact_quote():
    packet=dict(records=[dict(label='a',caption='A bicycle rack.',source_frames=[dict(index=85,path='85.png')])])
    scores={'a':dict(errors=[],rationale='Original.',fact_hits={'f':False})}
    d=dict(records=[dict(label='a',decision='add_error',quote='bicycle rack',kind='unsupported',reason='Identity not visible.',source_indices=[85])],viewed_frame_paths=['85.png'])
    updated=apply_adjudication(packet,d,scores)
    assert scores['a']['errors']==[] and len(updated['a']['errors'])==1
    assert updated['a']['fact_hits']==scores['a']['fact_hits']
    d['records'][0]['quote']='bicycle stand'
    with pytest.raises(ValueError):apply_adjudication(packet,d,scores)
    d['records'][0]['quote']='bicycle rack';d['viewed_frame_paths']=[]
    with pytest.raises(ValueError):apply_adjudication(packet,d,scores)
