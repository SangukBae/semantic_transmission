import copy
import json
from pathlib import Path

import numpy as np
import pytest

from semantic_transmission import bsu_allocation_corpus as corpus
from semantic_transmission import bsu_allocation_labels as labels
from semantic_transmission.artifacts import write_json
from semantic_transmission.bsu_allocation_transport import validate_candidate


def test_six_distinct_source_only_proposals_are_reproducible():
    features=np.random.default_rng(42).random((49,71)).astype(np.float32)
    first=labels.candidates(features,'source-a',49,4,20261019)
    assert first==labels.candidates(features,'source-a',49,4,20261019)
    assert [p['id'] for p in first]==list(labels.ARMS)
    assert [len(p['keys']) for p in first]==[3,4,2,3,3,3]
    assert len({(tuple(p['keys']),tuple(p['weights'])) for p in first})==6
    for proposal in first:
        validate_candidate(dict(proposal,etas=[.2*w for w in proposal['weights']]),49,4)
    changed=features.copy(); changed[:,48:]=1e10
    assert labels.candidates(changed,'source-a',49,4,20261019)==first


def test_static_source_still_has_six_distinct_valid_candidates():
    result=labels.candidates(np.zeros((49,71)),'static',49,4,1)
    assert len({(tuple(p['keys']),tuple(p['weights'])) for p in result})==6


def test_split_is_exact_deterministic_and_order_independent():
    rows=[dict(id=str(i),source_sha256=f'{i:064x}') for i in range(100)]
    split=corpus.split_rows(rows,80,20261019)
    assert sum(v['split']=='train' for v in split)==80
    assert sum(v['split']=='validation' for v in split)==20
    reverse=corpus.split_rows(rows[::-1],80,20261019)
    assert {v['id']:v['split'] for v in split}=={v['id']:v['split'] for v in reverse}
    with pytest.raises(ValueError,match='duplicate'):
        corpus.split_rows(rows+[rows[0]],80,20261019)


def test_duplicate_screen_checks_temporal_samples_and_color():
    a=dict(phash=['0'*63]*5,rgb8=np.zeros((5,8,8,3)).tolist())
    assert corpus.similar(a,copy.deepcopy(a))
    b=dict(phash=['1'*63]*5,rgb8=a['rgb8'])
    assert not corpus.similar(a,b)
    c=dict(phash=a['phash'],rgb8=np.ones((5,8,8,3)).tolist())
    assert not corpus.similar(a,c)


def test_stage_reuses_only_hash_verified_outputs(tmp_path):
    calls=[]
    def produce(_):
        calls.append(1); (tmp_path/'product').write_text('real test artifact')
    labels.step(tmp_path,'encode','identity',['product'],produce)
    labels.step(tmp_path,'encode','identity',['product'],produce)
    assert len(calls)==1
    (tmp_path/'product').write_text('changed')
    with pytest.raises(ValueError,match='artifacts changed'):
        labels.step(tmp_path,'encode','identity',['product'],produce)


def test_failed_stage_cannot_count_as_completed_and_can_resume(tmp_path):
    def fail(_):
        (tmp_path/'product').write_text('partial')
        raise RuntimeError('intentional test failure')
    with pytest.raises(RuntimeError): labels.step(tmp_path,'encode','identity',['product'],fail)
    assert json.loads((tmp_path/'stages/encode.json').read_text())['status']=='FAILED'
    labels.step(tmp_path,'encode','identity',['product'],lambda _: (tmp_path/'product').write_text('complete'))
    assert (tmp_path/'product').read_text()=='complete'
    assert list((tmp_path/'failed_attempts').glob('*/product'))


def test_protocol_fails_closed_on_code_changes(tmp_path):
    source=tmp_path/'source.py'; source.write_text('a=1')
    manifest=tmp_path/'corpus.json'; write_json(manifest,dict(records=[]))
    p=dict(code={str(source):labels.sha256(source)},corpus=labels.receipt(manifest))
    p['signature']=labels.fingerprint(p); write_json(tmp_path/'protocol.json',p)
    assert labels.verify(tmp_path)==p
    source.write_text('a=2')
    with pytest.raises(ValueError,match='frozen code changed'): labels.verify(tmp_path)


def test_missing_labels_do_not_create_a_dataset(tmp_path):
    manifest=tmp_path/'corpus.json'
    write_json(manifest,dict(records=[dict(id='a')]))
    with pytest.raises(FileNotFoundError): labels.export(tmp_path,dict(corpus=labels.receipt(manifest)))
    assert not (tmp_path/'dataset.json').exists()


@pytest.mark.parametrize('invalid_low_rate',[False,True])
def test_rate_search_uses_real_packet_cost_and_reuses_completed_encodes(tmp_path,monkeypatch,invalid_low_rate):
    from semantic_transmission import bsu_allocation_transport as transport
    from semantic_transmission.wire import pack
    calls=[]
    def fake_encode(model,row,candidate,source,output,**kwargs):
        calls.append(candidate)
        if invalid_low_rate and min(candidate['etas'])<.015:
            raise ValueError('invalid NTSCC+ channel input')
        output.mkdir(parents=True)
        sizes=[max(1,round(eta*10000)) for eta in candidate['etas']]
        np.ones(sum(sizes),dtype='<c8').tofile(output/'visual.c64')
        cursor=0; keys=[]
        for index,eta,size in zip(candidate['keys'],candidate['etas'],sizes):
            keys.append(dict(index=index,allocation_eta=eta,complex_count=size,complex_offset=cursor)); cursor+=size
        (output/'metadata.bin').write_bytes(pack(dict(video=row['video'],visual_codec=dict(eta=.2),keyframes=keys)))
    monkeypatch.setattr(transport,'encode_candidate',fake_encode)
    source=tmp_path/'corpus/sources/1'; source.mkdir(parents=True)
    np.savez(source/'features.npz',features=np.zeros((49,71)))
    write_json(tmp_path/'labels/1/caption.json',dict(review=dict(caption='Test fixture only.')))
    policy=labels.read(labels.CONFIG)
    p=dict(signature='test-only',policy=policy)
    row=dict(id='1',source_id='test-1',normalized_sha256='0'*64,video=dict(frames=49))
    labels.encode_source(tmp_path,p,row,None)
    allocations=[labels.read(tmp_path/'labels/1'/arm/'allocation.json') for arm in labels.ARMS]
    budget=allocations[0]['base_cost']['total_uses']
    assert all(v['base_cost']['total_uses']<=budget for v in allocations)
    assert len(set(allocations[3]['etas']))>1
    assert all(v['quality_used_for_selection'] is False for v in allocations)
    if invalid_low_rate:
        assert (tmp_path/'labels/1/more_keys/invalid_rate_trials.json').exists()
    count=len(calls)
    labels.encode_source(tmp_path,p,row,None)
    assert len(calls)==count
