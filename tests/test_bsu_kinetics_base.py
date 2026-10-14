import json

import pytest

from semantic_transmission import bsu_kinetics_base as current
from semantic_transmission import qwen35_semantic_selection as semantic
from test_qwen35_feedback import retry_session, ENTRY_RESPONSE


def test_word_count_feedback_preserves_first_request_and_semantic_checks(tmp_path):
    long=json.loads(ENTRY_RESPONSE)
    long.update(reference_state=' '.join(['visible']*27),candidate_state=' '.join(['changed']*22))
    short=json.loads(ENTRY_RESPONSE)
    session,prompts=retry_session(tmp_path,[json.dumps(long),json.dumps(short)])
    original=current.feedback.response_repair_prompt
    with current.current_pipeline():
        result=session.call('semantic_select',[0,16],'original request',
            lambda x:semantic.validate_decision(x,0,16),256)
        assert 'STATE LENGTH REPAIR' not in prompts[0]
        assert '27 words' in prompts[1] and '22 words' in prompts[1] and '49 total' in prompts[1]
        assert 'at most 12 words for EACH state' in prompts[1]
        assert result==short
    assert current.feedback.response_repair_prompt is original
    record=current.feedback.ae.unseal(next((tmp_path/'calls/semantic_select').glob('*.json')))
    assert json.loads(record['raw_text'])==short
    with pytest.raises(ValueError,match='45 words'):semantic.validate_decision(long,0,16)


def test_non_length_repair_and_valid_cached_calls_unchanged(tmp_path):
    session,prompts=retry_session(tmp_path,[ENTRY_RESPONSE])
    validator=lambda x:semantic.validate_decision(x,0,16)
    with current.current_pipeline():
        session.call('semantic_select',[0,16],'request',validator,256)
        session.call('semantic_select',[0,16],'request',validator,256)
    assert len(prompts)==1 and session.calls[-1]['reused']
    assert current.length_feedback([dict(raw_text=ENTRY_RESPONSE,error='wrong frame IDs')])==''


def test_three_length_failures_still_fail_and_retain_raw_records(tmp_path):
    long=json.loads(ENTRY_RESPONSE)
    long.update(reference_state=' '.join(['visible']*27),candidate_state=' '.join(['changed']*22))
    session,prompts=retry_session(tmp_path,[json.dumps(long)]*3)
    with current.current_pipeline(),pytest.raises(RuntimeError,match='three invalid responses'):
        session.call('semantic_select',[0,16],'request',lambda x:semantic.validate_decision(x,0,16),256)
    assert len(prompts)==3 and len(list((tmp_path/'failed_attempts').rglob('*.json')))==3


def test_dispatch_is_scoped_even_after_exception():
    original=current.base.MODULE
    with pytest.raises(RuntimeError),current.current_pipeline():
        assert current.base.MODULE==current.MODULE
        raise RuntimeError('failure')
    assert current.base.MODULE==original


def test_rebind_preserves_answer_and_rejects_changed_input(tmp_path):
    from semantic_transmission.artifacts import sha256,write_json
    session,_=retry_session(tmp_path,[ENTRY_RESPONSE])
    old=dict(source=dict(source_frame_hashes=session.plan['source_frame_hashes']),code={},template_sha256='old',
        selection_prompt='unchanged',frames=20)
    old['signature']=current.feedback.ae.digest(old)
    session.plan['signature']=old['signature']
    session.call('semantic_select',[0,16],'request',lambda x:semantic.validate_decision(x,0,16),256)
    path=next((tmp_path/'calls/semantic_select').glob('*.json'));original=path.read_bytes()
    record=current.feedback.ae.unseal(path)
    new={**old,'template_sha256':'new','code':{'src/semantic_transmission/bsu_kinetics_base.py':sha256(current.__file__)}}
    new['signature']=current.feedback.ae.digest({k:v for k,v in new.items() if k!='signature'})
    result=current.rebind_selection_record(record,old,new,path)
    assert result['raw_text']==record['raw_text'] and result['result']==record['result']
    assert result['reuse_provenance']['source_sha256']==sha256(path) and path.read_bytes()==original
    new['frames']=21;new['signature']=current.feedback.ae.digest({k:v for k,v in new.items() if k!='signature'})
    with pytest.raises(ValueError,match='inputs or policy'):current.rebind_selection_record(record,old,new,path)
