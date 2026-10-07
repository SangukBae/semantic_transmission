import pytest

from semantic_transmission.qwen35_feedback_eval import event_coverage, tolerant_matching, validate_judgments
from semantic_transmission.artifacts import write_json, sha256
from semantic_transmission.qwen35_feedback_eval import prepare_feedback_review
from semantic_transmission.auto_extraction import read


def test_event_requires_before_and_after_not_only_nearby_anchor():
    reference = dict(events=[dict(id="event", description="dog enters", before_range=[9, 11], after_range=[13, 15])])
    assert event_coverage([10, 12], reference)["covered"] == 0
    assert event_coverage([10, 14], reference)["covered"] == 1


def test_one_key_cannot_match_multiple_reference_keys():
    result = tolerant_matching([10], [8, 12], tolerance=3)
    assert result["precision"] == 1 and result["recall"] == .5


def test_one_reference_cannot_reward_dense_redundant_keys():
    result = tolerant_matching([8, 10, 12], [10], tolerance=3)
    assert result["precision"] == pytest.approx(1/3) and result["recall"] == 1


def test_matching_finds_maximum_assignment_not_greedy_nearest_only():
    result = tolerant_matching([2, 4], [0, 3], tolerance=2)
    assert len(result["matches"]) == 2


def judgment_fixture(root):
    dest=root/'evaluation'
    packet=dict(bins=[dict(reference=dict(id=0,facts=[dict(id='fact0')]),candidates=[
        dict(method='M',captions=[dict(caption_id='M_0',caption='A dog is visible.')])])])
    path=dest/'packets/packet.json'
    write_json(path,packet)
    write_json(dest/'protocol.json',dict(reference_sha256='reference'))
    write_json(dest/'input_manifest.json',{str(path):sha256(path)})
    data=dict(evaluator_model='gpt-6-astra',reference_sha256='reference',packet_sha256=sha256(path),
        judgments=[dict(bin_id=0,method='M',covered_fact_ids=['fact0'],unsupported_claims=[],uncertain_claims=[])])
    write_json(dest/'judgments/part.json',data)
    return data,dest/'judgments/part.json'


def test_judge_requires_exact_source_quote_not_a_paraphrase(tmp_path):
    data,path=judgment_fixture(tmp_path)
    data['judgments'][0]['unsupported_claims']=[dict(caption_id='M_0',quote='A cat is visible.',
        type='added',reason='No cat in source',evidence_frames=[0])]
    write_json(path,data)
    with pytest.raises(ValueError,match='exact candidate quote'):
        validate_judgments(tmp_path)


def test_judge_cannot_silently_skip_a_method_or_bin(tmp_path):
    data,path=judgment_fixture(tmp_path)
    data['judgments']=[]
    write_json(path,data)
    with pytest.raises(ValueError,match='incomplete Astra review'):
        validate_judgments(tmp_path)


def test_judge_cannot_count_unknown_or_duplicate_reference_facts(tmp_path):
    data,path=judgment_fixture(tmp_path)
    data['judgments'][0]['covered_fact_ids']=['fact0','made_up']
    write_json(path,data)
    with pytest.raises(ValueError,match='unknown reference fact'):
        validate_judgments(tmp_path)


def test_judge_ambiguities_are_not_automatically_errors(tmp_path):
    data,path=judgment_fixture(tmp_path)
    data['judgments'][0]['uncertain_claims']=['Animal subtype unclear']
    write_json(path,data)
    result=validate_judgments(tmp_path)
    assert not result[0]['unsupported_claims'] and not result[0]['missed_fact_ids']


def test_exact_caption_reuse_preserves_astra_labels_and_rejects_modified_labels(tmp_path):
    dest=tmp_path/'evaluation'
    bins=[]
    for i in range(2):
        candidates=[]
        for code in ('A','B'):
            caption='A dog is visible.' if i==0 or code=='A' else 'Snow covers the ground.'
            candidates.append(dict(method=code,captions=[dict(caption_id=f'{code}_{i}',start=i*24,end_exclusive=(i+1)*24,caption=caption)]))
        bins.append(dict(reference=dict(id=i,facts=[dict(id=f'fact{i}')]),candidates=candidates))
    packet=dest/'packets/all.json'
    write_json(packet,dict(reference_sha256='ref',bins=bins))
    old=dest/'judgments/00_01_phase2.json'
    rows=[dict(bin_id=i,method='A',covered_fact_ids=[f'fact{i}'],unsupported_claims=[],uncertain_claims=[]) for i in range(2)]
    write_json(old,dict(evaluator_model='gpt-6-astra',reference_sha256='ref',packet_sha256=sha256(packet),judgments=rows))
    prepare_feedback_review(tmp_path,[packet],{'A':'forward','B':'feedback'})
    delivery=read(dest/'feedback_review_delivery.json')
    assert delivery['reused_bins']==[0] and delivery['changed_bins']==[1]
    pending=dest/'early_packets/00_29_final.json'
    write_json(dest/'judgments/new.json',dict(evaluator_model='gpt-6-astra',reference_sha256='ref',packet_sha256=sha256(pending),
        judgments=[dict(bin_id=1,method='B',covered_fact_ids=[],unsupported_claims=[],uncertain_claims=[])]))
    write_json(dest/'protocol.json',dict(reference_sha256='ref'))
    write_json(dest/'input_manifest.json',{str(p):sha256(p) for p in (packet,pending,old)})
    assert len(validate_judgments(tmp_path))==4
    reused=dest/'judgments/exact_input_reuse.json'
    value=read(reused);value['judgments'][0]['covered_fact_ids']=[];write_json(reused,value)
    with pytest.raises(ValueError,match='labels were altered'):
        validate_judgments(tmp_path)
