"""Keep code-fix resume restricted to verified, unaffected completed products."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

from semantic_transmission import quality_validation as app
from semantic_transmission.artifacts import write_json


@pytest.fixture
def repair():
    path = Path(__file__).resolve().parents[1]/"scripts/repair_quality_validation_short_overlap.py"
    spec = importlib.util.spec_from_file_location("overlap_repair", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repair_accepts_only_exact_decoder_edit(repair):
    old = {"config":{"seed":42}, "execution":{"code_sha256":{
        repair.DECODER:hashlib.sha256(repair.BEFORE.encode()).hexdigest()}}}
    new = copy.deepcopy(old)
    new["execution"]["code_sha256"][repair.DECODER] = hashlib.sha256(repair.AFTER.encode()).hexdigest()
    assert repair.validate_patch(old,new,repair.AFTER) == repair.BEFORE
    with pytest.raises(ValueError,match="extend beyond"):
        repair.validate_patch(old,new,repair.AFTER+"\n# another change")
    new["config"]["seed"] = 43
    with pytest.raises(ValueError,match="other code"):
        repair.validate_patch(old,new,repair.AFTER)


def receipt_for(job):
    return dict(status="PASSED",seconds=5,artifacts=app.inventory(job["required"]),
                signature=app.fingerprint({"job":job,"previous":app.fingerprint("old")}))


def test_rebound_receipt_is_reused_and_tampered_products_are_rejected(repair,tmp_path,monkeypatch):
    output=tmp_path/"output.txt"
    output.write_text("verified output")
    job=app.task("completed",1,{"action":"fixture"},[output])
    original=receipt_for(job)
    updated,_=repair.rebind_prefix([job],{"completed":original},"old","new")
    assert updated["completed"]["artifacts"] == original["artifacts"]
    assert updated["completed"]["produced_with_execution_signature"] == "old"
    write_json(tmp_path/"tasks/completed/receipt.json",updated["completed"])
    monkeypatch.setattr(app,"launch",lambda *_:pytest.fail("verified output must not be rerun"))
    app.Runner(tmp_path,"new",tmp_path,{}).execute(job,lambda _:None)
    output.write_text("changed")
    with pytest.raises(ValueError,match="artifact changed"):
        repair.rebind_prefix([job],{"completed":original},"old","new")


def test_repair_rejects_completed_short_overlap(repair,tmp_path):
    inputs=tmp_path/"receiver/decoder_inputs.json"
    write_json(inputs,{"indices":[0,7,15]})
    config=tmp_path/"receiver/decoder_config.py"
    config.write_text("condition_frame_length = 5\n")
    job=app.task("short.reconstruct",2,{"action":"pipeline","stage":"reconstruct","run":str(tmp_path)},[inputs,config])
    with pytest.raises(ValueError,match="changed branch"):
        repair.rebind_prefix([job],{job["name"]:receipt_for(job)},"old","new")


def test_repair_rejects_hole_in_completed_dependencies(repair,tmp_path):
    output=tmp_path/"out";output.write_text("ok")
    first=app.task("failed",1,{},[output])
    second=app.task("completed",1,{},[output])
    with pytest.raises(ValueError,match="contiguous"):
        repair.rebind_prefix([first,second],{"completed":receipt_for(second)},"old","new")
