import copy
from pathlib import Path

import pytest

from semantic_transmission import assisted_captions as module
from semantic_transmission import hybrid_reconstruction, workers
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.webvid5 import fingerprint, read_json


def test_sampling_matches_existing_pllava_sampler_including_short_intervals(monkeypatch):
    monkeypatch.setattr(workers,"source_images",lambda _,indices: indices)
    for start in (0,113,1432):
        for length in range(1,25):
            actual,_=workers._caption_images({"paper_caption":True},None,None,start,start+length,None)
            assert module.sample_indices(start,start+length)==actual
            assert all(start <= i < start+length for i in actual)
    assert module.sample_indices(113,114)==[113]*4


@pytest.fixture
def bundle_fixture(tmp_path,monkeypatch):
    root=tmp_path / "selection"
    write_json(root / "selection_freeze.json",{"fixture":True})
    protocol=dict(input_sha256="input",source_frame_hashes={str(i):str(i) for i in range(8)})
    selection=dict(indices=[0,3,7])
    monkeypatch.setattr(module,"verify_selection",lambda *_:(protocol,selection))
    records=[dict(r,text="A snowy field with bare trees.") for r in module.expected_samples(protocol,selection)]
    bundle=dict(version=1,input_sha256="input",selection_freeze_sha256=sha256(root / "selection_freeze.json"),
        source_frame_hashes={str(i):str(i) for r in records for i in r["source_indices"]},
        records=records,scope="offline source captions")
    path=tmp_path / "captions.json"
    write_json(path,dict(bundle,checksum=fingerprint(bundle)))
    return root,path,bundle


@pytest.mark.parametrize("change",["checksum","source","selection","missing","sample","empty"])
def test_rejects_stale_incomplete_or_misaligned_captions(bundle_fixture,change):
    root,path,original=bundle_fixture
    bundle=copy.deepcopy(original)
    if change=="source": bundle["input_sha256"]="wrong"
    elif change=="selection": bundle["selection_freeze_sha256"]="wrong"
    elif change=="missing": bundle["records"].pop()
    elif change=="sample": bundle["records"][0]["source_indices"][0]+=1
    elif change=="empty": bundle["records"][0]["text"]=" "
    write_json(path,dict(bundle,checksum="bad" if change=="checksum" else fingerprint(bundle)))
    with pytest.raises(ValueError): module.validate_bundle(path,root)


def test_import_uses_verified_text_and_never_loads_pllava(bundle_fixture,tmp_path,monkeypatch):
    root,path,bundle=bundle_fixture
    run=tmp_path / "run"
    write_json(run / "run_config.json",dict(caption_bundle=str(path),hybrid_selection_root=str(root)))
    write_json(run / "keyframes.json",dict(indices=[0,3,7]))
    write_json(run / "semantic_clips_audit.json",dict(status="PASSED",policy="frame_exact",source={"keyframes":[0,3,7]}))
    monkeypatch.setattr(workers,"_load_caption_inference",lambda *_:pytest.fail("PLLaVA inference started"))
    module.import_captions(run)
    rows=read_json(run / "captions.json")
    assert [r["text"] for r in rows]==[r["text"] for r in bundle["records"]]
    assert [r["path"] for r in rows]==["clips/sample/00000.mp4","clips/sample/00001.mp4"]
    assert all(r["flow"]==0.0 for r in rows)
    assert read_json(run / "caption_provenance.json")["pllava_inference_executed"] is False


def test_caption_option_uses_separate_output_and_read_only_check(tmp_path,monkeypatch,capsys):
    seen=[]
    def check(root,output,bundle):
        seen.append((root,output,bundle))
        return (None,None,None,None,None,{"status":"READY_FOR_USER_EXECUTION"})
    monkeypatch.setattr(hybrid_reconstruction,"check_ready",check)
    monkeypatch.setattr(hybrid_reconstruction,"execute",lambda *_:pytest.fail("--check started reconstruction"))
    hybrid_reconstruction.main(["--selection-root",str(tmp_path),"--captions","--check"])
    assert seen==[(tmp_path,tmp_path / "reconstruction_assistant_captions",tmp_path / "assistant_captions/captions_bundle.json")]
    assert not list(tmp_path.iterdir())
    assert "READY_FOR_USER_EXECUTION" in capsys.readouterr().out


def test_assisted_plan_imports_captions_and_removes_pllava_checkpoint(tmp_path):
    bundle=tmp_path / "captions.json"
    cfg=hybrid_reconstruction.build_config({"config":{"selector_checkpoint":"old"}},tmp_path,tmp_path / "out",bundle)
    assert "caption_checkpoint" not in cfg and "selector_checkpoint" not in cfg
    caption=next(p for p in hybrid_reconstruction.job_plan(bundle) if p[0]=="caption")
    assert caption[1]==hybrid_reconstruction.MODULE
    assert "caption_provenance.json" in caption[2]
