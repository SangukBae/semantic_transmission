import copy
import pytest
from semantic_transmission.qwen35_astra_transfer import validate_annotation, candidate_supervision, arm_config, ARMS


def sample():
    return dict(id='row',caption='A person raises a hand.',
        facts=[dict(id='f0',text='A person raises a hand.',kind='event',evidence_groups=[[0],[5,7]])],
        events=[dict(description='Hand rises',before=[0],after=[5,7],cause='object_motion')],
        entities=[dict(id='e0',description='person',visibility='visible',evidence=[0,2,5,7])],
        uncertainties=[],event_annotation_status='reviewed_observed_frames')


def test_visible_input_reference_accepted():
    record=sample()
    assert validate_annotation(record,dict(id='row',keys=[0,2,5,7])) is record


@pytest.mark.parametrize('kind',['fact','event','entity'])
def test_teacher_cannot_use_hidden_frames(kind):
    record=sample()
    if kind=='fact':record['facts'][0]['evidence_groups']=[[1],[7]]
    if kind=='event':record['events'][0]['before']=[1]
    if kind=='entity':record['entities'][0]['evidence']=[1]
    with pytest.raises(ValueError):validate_annotation(record,dict(id='row',keys=[0,2,5,7]))


@pytest.mark.parametrize('groups',[[[5],[2]],[[0],[2],[2],[7]],[[0],[0,2],[0,2]]])
def test_temporal_evidence_requires_distinct_ordered_witnesses(groups):
    record=sample();record['facts'][0]['evidence_groups']=groups
    with pytest.raises(ValueError):validate_annotation(record,dict(id='row',keys=[0,2,5,7]))


def test_duplicate_fact_ids_rejected():
    record=sample();record['facts'].append(copy.deepcopy(record['facts'][0]))
    with pytest.raises(ValueError):validate_annotation(record,dict(id='row',keys=[0,2,5,7]))


def test_no_event_annotation_does_not_create_false_negatives():
    record=sample();record['events']=[]
    result=candidate_supervision(record,[0,2,5,7])
    assert sum(sum(row) for row in result['mask'])==0


def test_explicit_supervision_preserves_unseen_mask():
    record=sample();record['confirmed_no_transition_indices']=[2]
    result=candidate_supervision(record,[0,2,5,7])
    assert result['targets'][0]==[1.,0.] and result['mask'][0]==[1.,0.]
    assert result['mask'][2]==[1.,1.] and result['targets'][2]==[0.,0.]
    assert result['mask'][1]==[0.,0.] and result['mask'][6]==[0.,0.]


def test_rank_and_data_are_crossed_at_equal_scaling():
    configs={a:arm_config({'config':{'max_pixels':262144}},a) for a in ARMS}
    assert {(t,c['lora_rank']) for c,t in configs.values()}=={('control',8),('teacher',8),('control',32),('teacher',32)}
    assert {c['lora_alpha']/c['lora_rank'] for c,_ in configs.values()}=={2}


def gate_counts():
    return {split:{arm:dict(fact_hits=10,action_hits=5,error_windows=2,generation_failures=0)
        for arm in ['base',*ARMS]} for split in ['validation','test']}


def test_no_gain_or_split_tradeoff_cannot_pass_gate():
    from evaluate_qwen35_astra_transfer import extension_gates
    counts=gate_counts()
    assert not extension_gates(counts)['teacher_r8']['passed']
    counts['validation']['teacher_r8']['fact_hits']=20
    counts['test']['teacher_r8']['fact_hits']=9
    assert not extension_gates(counts)['teacher_r8']['passed']


def test_gate_requires_data_improvement_against_same_rank_control():
    from evaluate_qwen35_astra_transfer import extension_gates
    counts=gate_counts()
    for split in counts:
        counts[split]['teacher_r32']['fact_hits']=12
    assert extension_gates(counts)['teacher_r32']['passed']
    counts['test']['control_r32']['action_hits']=6
    assert not extension_gates(counts)['teacher_r32']['passed']


@pytest.mark.parametrize('field',['error_windows','generation_failures'])
def test_fact_gain_cannot_hide_new_errors_or_failures(field):
    from evaluate_qwen35_astra_transfer import extension_gates
    counts=gate_counts()
    for split in counts:counts[split]['teacher_r8']['fact_hits']=20
    counts['test']['teacher_r8'][field]+=1
    assert not extension_gates(counts)['teacher_r8']['passed']


def test_review_cannot_cite_an_unseen_source_frame():
    from evaluate_qwen35_astra_transfer import validate_source_review,digest
    row=dict(label='c0',caption='A dog runs.',caption_sha256=digest('A dog runs.'),facts=[dict(id='f0')],source_frames=[dict(index=100)])
    judged=dict(label='c0',caption_sha256=row['caption_sha256'],fact_hits={'f0':False},rationale='Incorrect action.',errors=[dict(quote='runs',reason='Dog sits.',source_indices=[99])])
    with pytest.raises(ValueError):validate_source_review({'records':[row]},{'records':[judged]})
    judged['errors'][0]['source_indices']=[100]
    assert validate_source_review({'records':[row]},{'records':[judged]})['c0'] is judged
