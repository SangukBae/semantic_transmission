"""Regression coverage for action-only dispatch and moved T5 cache integrity."""
import copy
import json

import pytest

from semantic_transmission import bsu_dcvc_validation as validation
from semantic_transmission import precision_text_cache, text_embedding_cache
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.webvid5 import fingerprint


def test_action_job_needs_no_reconstruction_run(monkeypatch):
    job={'runtime':{},'rows':[],'output':'action'}
    seen=[]
    monkeypatch.setattr(validation.legacy,'action',seen.append)
    validation.worker(job,'action')
    assert seen==[job]


def text_fixture(tmp_path,monkeypatch):
    model=tmp_path/'model';model.mkdir()
    (model/'config.json').write_text('{}')
    (model/'model.bin').write_bytes(b'pinned model bytes')
    old=tmp_path/'old';run=tmp_path/'new'
    inventory=text_embedding_cache.model_inventory(model)
    previous=copy.deepcopy(inventory)
    for value in previous.values():
        value['target']='/original-machine/weight-file'
        value['ctime_ns']=1
    contract=dict(model_path=str(model),model_inventory=previous,
        model_sha256={name:sha256(model/name) for name in inventory},code={})
    tensor=tmp_path/'embedding.pt';tensor.write_bytes(b'unchanged embedding')
    signature=fingerprint(contract)
    report=dict(contract=contract,signature=signature,
                entries=[dict(path=str(tensor),sha256=sha256(tensor),contract=signature,prompt='a caption')])
    write_json(old/precision_text_cache.REPORT,report)
    def validate(entry,contract,prompt):
        assert entry['contract']==fingerprint(contract)
        assert sha256(entry['path'])==entry['sha256']
    monkeypatch.setattr(precision_text_cache,'load',validate)
    return model,old,run,report,inventory,tensor


def test_rebind_preserves_original_report_and_embedding(tmp_path,monkeypatch):
    model,old,run,report,inventory,tensor=text_fixture(tmp_path,monkeypatch)
    original=sha256(old/precision_text_cache.REPORT);embedding=sha256(tensor)
    validation.portable_text_report(old,run)
    current=json.loads((run/precision_text_cache.REPORT).read_text())
    assert sha256(old/precision_text_cache.REPORT)==original
    assert sha256(tensor)==embedding
    assert current['contract']['model_inventory']==inventory
    assert current['signature']==fingerprint(current['contract'])
    assert current['inventory_rebinding']['original_signature']==report['signature']


def test_rebind_rejects_changed_model_bytes(tmp_path,monkeypatch):
    model,old,run,*_=text_fixture(tmp_path,monkeypatch)
    (model/'model.bin').write_bytes(b'different weights')
    with pytest.raises(ValueError,match='bytes changed'):
        validation.portable_text_report(old,run)
    assert not (run/precision_text_cache.REPORT).exists()
