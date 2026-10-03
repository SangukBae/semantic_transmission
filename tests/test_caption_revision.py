import csv
import shutil
import subprocess

import pytest

from semantic_transmission import assisted_captions, caption_revision as module
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.webvid5 import fingerprint, read_json


@pytest.fixture
def revision_inputs(tmp_path, monkeypatch):
    root = tmp_path / "selection"
    parent_path = root / "assistant_captions/captions_bundle.json"
    write_json(root / "selection_freeze.json", {"frozen": True})
    protocol = dict(input_sha256="source", source_frame_hashes={str(i):str(i) for i in range(8)})
    selection = dict(indices=[0,3,7])
    monkeypatch.setattr(assisted_captions,"verify_selection",lambda _: (protocol,selection))
    write_json(parent_path.parent / "samples.json", {"unchanged":True})
    sheet = parent_path.parent / "sheets/00.jpg"
    sheet.parent.mkdir()
    sheet.write_bytes(b"source observation fixture")
    records = [dict(r,text="Two people walk away on snow.")
               for r in assisted_captions.expected_samples(protocol,selection)]
    payload = dict(version=1, input_sha256="source", records=records,
        selection_freeze_sha256=sha256(root / "selection_freeze.json"),
        source_frame_hashes={str(i):str(i) for r in records for i in r["source_indices"]},
        evidence={"sheets/00.jpg":sha256(sheet), "samples.json":sha256(parent_path.parent / "samples.json")})
    write_json(parent_path,dict(payload,checksum=fingerprint(payload)))
    spec_path = tmp_path / "spec.json"
    spec = dict(version=1, revision="fixture_v2", author="test", instructions=["Describe visible facts."],
        sampling="same samples", parent_bundle_sha256=sha256(parent_path), input_sha256="source",
        selection_freeze_sha256=payload["selection_freeze_sha256"],
        reviewed_source_sheets={"sheets/00.jpg":sha256(sheet)},
        texts={"0":"Two cropped backs remain low in a snowy field.",
               "1":"The dog's head remains partly visible at the bottom edge."})
    write_json(spec_path,spec)
    return root,parent_path,spec_path,spec


def test_revision_freeze_preserves_parent_and_rejects_later_edits(revision_inputs):
    root,parent,spec_path,spec = revision_inputs
    parent_hash = sha256(parent)
    path = module.prepare(root,spec_path)
    frozen = assisted_captions.validate_bundle(path,root)
    assert [r["text"] for r in frozen["records"]] == list(spec["texts"].values())
    assert sha256(parent) == parent_hash
    before = sha256(path)
    assert module.prepare(root,spec_path) == path
    spec["texts"]["0"] = "A revised description would require a new frozen revision."
    write_json(spec_path,spec)
    with pytest.raises(ValueError,match="preserve frozen"):
        module.prepare(root,spec_path)
    assert sha256(path) == before and sha256(parent) == parent_hash


@pytest.mark.parametrize("change", ["source","parent","missing","evidence","truncated","sentences"])
def test_revision_rejects_wrong_source_or_incomplete_descriptions(revision_inputs,change):
    root,parent,spec_path,spec = revision_inputs
    if change == "source":spec["input_sha256"] = "wrong"
    elif change == "parent":spec["parent_bundle_sha256"] = "wrong"
    elif change == "missing":spec["texts"].pop("0")
    elif change == "evidence":spec["reviewed_source_sheets"] = {}
    elif change == "truncated":spec["texts"]["0"] = "The text ends in the middle of"
    elif change == "sentences":spec["texts"]["0"] = "A fact. "*7
    write_json(spec_path,spec)
    with pytest.raises(ValueError):module.prepare(root,spec_path)
    assert not module.paths(root)[0].exists()


def test_check_mode_is_read_only_and_never_runs_models(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(module,"preflight",lambda: ({},{"status":"READY_FOR_USER_EXECUTION"}))
    monkeypatch.setattr(module.hybrid,"execute",lambda *_:pytest.fail("check launched GPU work"))
    monkeypatch.setattr(module,"write_json",lambda *_:pytest.fail("check wrote a receipt"))
    module.main(["run","--check"])
    assert "READY_FOR_USER_EXECUTION" in capsys.readouterr().out


def test_only_caption_path_may_change_in_settings():
    old = dict(caption_bundle="v1", seed=2025, snr_db=10, steps=30)
    new = dict(old,caption_bundle="v2")
    module.assert_same_settings(old,new)
    for key,value in [("seed",42),("snr_db",8),("steps",10)]:
        with pytest.raises(ValueError,match="non-caption"):
            module.assert_same_settings(old,dict(new,**{key:value}))


@pytest.fixture
def paired_runs(tmp_path):
    roots = [tmp_path / "v1",tmp_path / "v2"]
    for i,root in enumerate(roots):
        run=root / "run"
        write_json(run / "run_config.json",dict(seed=2025,caption_bundle=f"v{i+1}"))
        write_json(run / "keyframes.json",dict(indices=[0,2]))
        write_json(run / "captions.json",[dict(text=f"Caption {i+1}.")])
        (run / "received").mkdir()
        (run / "received/visual.c64").write_bytes(b"identical channel symbols")
        keys=run / "receiver/frames/sample/key_frames_received"
        keys.mkdir(parents=True)
        for k in (0,2):(keys / f"{k}.png").write_bytes(bytes([k]))
        with (run / "receiver/metadata.csv").open("w") as f:
            w=csv.DictWriter(f,fieldnames=["path","text","flow"]);w.writeheader()
            w.writerow(dict(path="clips/sample/00000.mp4",text=f"Caption {i+1}.",flow=1.25))
    return roots


def test_pair_audit_allows_changed_captions_but_does_not_claim_exact_noise(paired_runs):
    report=module.verify_pair(*paired_runs)
    assert report["received_keyframes_identical"] == 2
    assert report["flow_and_segments_identical"] is True
    assert report["diffusion_noise_tensor_identity_verified"] is False


@pytest.mark.parametrize("change",["received_symbol","received_key","flow","caption"])
def test_pair_audit_rejects_uncontrolled_inputs(paired_runs,change):
    run=paired_runs[1] / "run"
    if change=="received_symbol":(run / "received/visual.c64").write_bytes(b"changed")
    elif change=="received_key":(run / "receiver/frames/sample/key_frames_received/2.png").write_bytes(b"changed")
    else:
        rows=module.csv_rows(run / "receiver/metadata.csv")
        rows[0]["flow" if change=="flow" else "text"] = "9" if change=="flow" else "Unintended corruption."
        with (run / "receiver/metadata.csv").open("w") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError):module.verify_pair(*paired_runs)


def test_comparison_video_and_completed_stage_reuse(paired_runs, tmp_path):
    # Exercise FFmpeg stacking and timestamp validation without any GPU models.
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg required for the comparison smoke test")
    source = tmp_path / "fixture.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
        "color=c=blue:s=16x16:r=3", "-frames:v", "3", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(source)], check=True)
    for root in paired_runs:
        run = root / "run"
        cfg = read_json(run / "run_config.json")
        write_json(run / "run_config.json", dict(cfg, frames=3, fps=3))
        for relative in ("data/normalized.mp4", "receiver/reconstruction/sample_0000.mp4"):
            target = run / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        # Synthetic metric values for report plumbing, not a quality evaluation.
        write_json(run / "quality.json", dict(status="PASSED", video_sha256=sha256(source),
            source_sha256=sha256(source), delivered_mp4=dict(psnr_db=30, ssim=1,
            lpips_vgg=0, clip=1, dists=0)))
        write_json(run / "channel_accounting.json", dict(total_complex_channel_uses=10))
    output = paired_runs[1]
    products = ["caption_comparison.mp4", "caption_comparison.html", "CAPTION_COMPARISON.json"]
    # An interrupted previous attempt is archived before regeneration.
    (output / products[0]).write_bytes(b"interrupted video")
    stage = module.Stages(output, "test")
    stage.step("caption-revision-comparison", products, products,
               lambda _: module.comparison(*paired_runs))
    assert list((output / "failed_attempts").rglob("caption_comparison.mp4"))
    result = read_json(output / "CAPTION_COMPARISON.json")
    assert result["hallucination_review"] == "PENDING"
    assert result["hallucination_mitigation_verified"] is False
    assert "수정 캡션 v2" in (output / "caption_comparison.html").read_text()
    module.Stages(output, "test").step("caption-revision-comparison", products, products,
        lambda _: pytest.fail("completed comparison should be reused"))
