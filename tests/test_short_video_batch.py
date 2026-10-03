from pathlib import Path
import shutil
import subprocess

import pytest

from semantic_transmission import short_video_batch as batch
from semantic_transmission.artifacts import write_json
from semantic_transmission.webvid5 import read_json


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    repo, root = tmp_path, tmp_path / "prepared"
    base = repo / "baseline"
    selection = root / "single_subject"
    for name in ("run_config.json", "keyframes.json", "data/normalized.mp4", "transmitter/visual.c64",
                 "received/visual.c64", "received/metadata.bin", "receiver/decoder_inputs.json",
                 "receiver/reconstruction/sample_0000.mp4"):
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    model = repo / "model"
    model.mkdir()
    for name in ("ntscc_hyperprior_quality_4_psnr.pth", "unimatch.pth"):
        p = repo / ".local/checkpoints" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(name)
    selection.mkdir(parents=True)
    write_json(selection / "selection_freeze.json", {"fixture": True})
    write_json(selection / "assistant_captions/captions_bundle.json", {"fixture": True})
    cfg = dict(frames=25, fps=24, seed=2025, snr_db=10, decoder_policy="official_release",
               models={k:str(model) for k in ("stdit", "vae", "vae2d", "t5")})
    protocol = dict(config=cfg, baseline=str(base), frames=25, fps=24)
    selected = dict(indices=[0, 12, 24])
    monkeypatch.setattr(batch, "ROOT", root)
    monkeypatch.setattr(batch, "REPO", repo)
    monkeypatch.setattr(batch, "CODE", ())
    monkeypatch.setattr(batch, "verify_selection", lambda _: (protocol, selected))
    monkeypatch.setattr(batch.captions, "validate_bundle", lambda *_: {"records":[{}, {}]})
    monkeypatch.setattr(batch, "settings", lambda _: {"python":__file__, "channel_python":__file__})
    monkeypatch.setattr(batch.shutil, "which", lambda _: "/bin/true")
    return selection, protocol, selected


def test_check_three_inputs_never_executes_or_creates_outputs(prepared, monkeypatch, capsys):
    monkeypatch.setattr(batch, "console_run", lambda *_: pytest.fail("check started a process"))
    monkeypatch.setattr(batch, "execute", lambda *_: pytest.fail("check reconstructed"))
    batch.main(["--video", "single_subject", "--check"])
    assert "READY_FOR_USER_EXECUTION" in capsys.readouterr().out
    assert not (prepared[0] / "reconstruction_fp32").exists()


def test_receipt_rejects_changed_caption_before_resuming(prepared):
    *_, identity, signature, report = batch.preflight("single_subject")
    output = Path(report["output"])
    write_json(output / "execution_protocol.json", dict(identity, signature=signature))
    batch.preflight("single_subject")
    write_json(prepared[0] / "assistant_captions/captions_bundle.json", {"changed": True})
    with pytest.raises(ValueError, match="frozen input/code changed"):
        batch.preflight("single_subject")


def test_no_baseline_quality_file_required_for_candle_style_input(prepared):
    assert not (Path(prepared[1]["baseline"]) / "quality.json").exists()
    (Path(prepared[1]["baseline"]) / "keyframes.json").unlink()
    report = batch.preflight("single_subject")[-1]
    assert report["keyframes"] == 3 and report["captions"] == 2
    assert report["t5"] == "CPU FP32 cache" and report["steps"] == 30


def test_legacy_seed_run_resolves_only_matching_declared_channel(tmp_path):
    base, source = tmp_path / "seed42", tmp_path / "source"
    for root in (base, source):
        write_json(root / "receiver/decoder_inputs.json", {"video":{"frames":3},"indices":[0,2]})
        (root / "receiver/metadata.csv").write_text("same received captions and flow")
        (root / "data").mkdir()
        (root / "data/normalized.mp4").write_bytes(b"same source")
    for filename in ("transmitter/visual.c64", "received/visual.c64", "received/metadata.bin"):
        path = source / filename
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"channel fixture")
    write_json(base / "seed_protocol.json", {"source":str(source),"channel_reused":True})
    assert batch.channel_source(base) == source
    (base / "receiver/metadata.csv").write_text("different received conditions")
    with pytest.raises(ValueError, match="received metadata differs"):
        batch.channel_source(base)


def test_channel_keeps_common_noise_and_repeats_new_frame_noise(tmp_path, monkeypatch):
    import numpy as np
    from semantic_transmission import metadata_channel, transmission_accounting
    from semantic_transmission.wire import pack
    base, run = tmp_path / "old", tmp_path / "new"
    (base / "received").mkdir(parents=True)
    (base / "transmitter").mkdir()
    (run / "transmitter").mkdir(parents=True)
    def item(index, offset):
        return dict(index=index,complex_offset=offset,complex_count=1,rate_offset=offset,rate_bytes=1)
    (base / "received/metadata.bin").write_bytes(pack({"keyframes":[item(0,0)]},b"a"))
    np.array([1+0j],dtype="<c8").tofile(base / "transmitter/visual.c64")
    np.array([1.1+0.2j],dtype="<c8").tofile(base / "received/visual.c64")
    packet = pack({"keyframes":[item(0,0),item(2,1)],"video":{}},b"ab")
    (run / "transmitter/metadata.bin").write_bytes(packet)
    np.array([1+0j,2+0j],dtype="<c8").tofile(run / "transmitter/visual.c64")
    write_json(run / "run_config.json",dict(hybrid_selection_root="fixture",snr_db=10,channel_seed=1024,
        width=32,height=32,frames=3))
    monkeypatch.setattr(batch,"verify_selection",lambda _ : ({"baseline":str(base)},{}))
    monkeypatch.setattr(metadata_channel,"transmit",lambda p,*_:(p,{"bit_errors":0,"complex_channel_uses":20}))
    monkeypatch.setattr(transmission_accounting,"channel_breakdown",lambda *_:{})
    batch.matched_channel(run)
    first = np.fromfile(run / "received/visual.c64",dtype="<c8")
    assert first[0] == np.complex64(1.1+0.2j) and first[1] != np.complex64(2)
    report = read_json(run / "channel_accounting.json")
    assert report["common_frames_exact"] == [0] and report["new_noise_frames"] == [2]
    shutil.rmtree(run / "received")
    batch.matched_channel(run)
    assert np.array_equal(first,np.fromfile(run / "received/visual.c64",dtype="<c8"))


def test_plan_imports_captions_and_preserves_sender_receiver_order():
    jobs = batch.job_plan()
    names = [row[0] for row in jobs]
    assert "select" not in names
    assert next(module for name,module,_ in jobs if name == "caption") == batch.hybrid.MODULE
    assert next(module for name,module,_ in jobs if name == "channel") == batch.MODULE
    assert names == ["input-audit", "semantic-clips", "caption", "flow", "send", "channel", "receive",
                     "receiver-policy", "prepare-text", "reconstruct", "output-audit", "evaluate"]


def test_old_duplicate_boundaries_removed_without_stretching(tmp_path):
    write_json(tmp_path / "run_config.json", {"frames":5})
    write_json(tmp_path / "receiver/decoder_inputs.json", {"indices":[0,2,4], "decoder":{"policy":"official_release"}})
    mapping, kept = batch.baseline_alignment(tmp_path)
    assert mapping == [0,1,2,2,3,4]
    assert kept == [0,1,2,4,5]
    assert [mapping[i] for i in kept] == list(range(5))


def test_workers_reject_check_flag(prepared):
    with pytest.raises(SystemExit):
        batch.main(["--worker", "reconstruct", "--run-dir", str(prepared[0]), "--check"])


def test_completed_pipeline_resumes_without_model_launch(prepared, monkeypatch):
    selection, protocol, selected = prepared
    output = selection / "reconstruction_fp32"
    calls = []
    monkeypatch.setattr(batch.hybrid, "initialization_products", lambda _: ["run/run_config.json"])
    monkeypatch.setattr(batch.hybrid, "initialize", lambda o,p,s,c:write_json(o / "run/run_config.json",c))
    monkeypatch.setattr(batch, "job_plan", lambda:[("reconstruct",batch.MODULE,["receiver/reconstruction/video.json"])])
    def fake_launch(command, log, env, resource):
        calls.append(command)
        write_json(output / "run/receiver/reconstruction/video.json", {"complete":True})
        write_json(resource, {"returncode":0})
    monkeypatch.setattr(batch.hybrid, "launch", fake_launch)
    def report(out):
        for name in ("baseline_aligned.mp4", "comparison.mp4", "RESULT.json", "review.html"):
            (out / name).write_text("fixture")
    monkeypatch.setattr(batch, "make_comparison", report)
    # lock() creates its parent in the real repository; give the fixture one.
    (batch.REPO / ".local").mkdir(exist_ok=True)
    batch.execute("single_subject")
    batch.execute("single_subject")
    assert len(calls) == 1
    assert read_json(output / "stages/reconstruct.json")["status"] == "PASSED"


def test_review_handles_shape_metadata_and_aligns_real_tiny_video(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg required")
    base, output = tmp_path / "base", tmp_path / "output"
    run = output / "run"
    (base / "receiver/reconstruction").mkdir(parents=True)
    (run / "receiver/reconstruction").mkdir(parents=True)
    (run / "data").mkdir()
    def make_video(path, frames):
        subprocess.run(["ffmpeg","-v","error","-f","lavfi","-i","testsrc2=size=32x32:rate=24",
                        "-frames:v",str(frames),"-c:v","libx264","-pix_fmt","yuv420p",str(path)],check=True)
    make_video(base / "receiver/reconstruction/sample_0000.mp4",6)
    make_video(run / "receiver/reconstruction/sample_0000.mp4",5)
    shutil.copyfile(run / "receiver/reconstruction/sample_0000.mp4",run / "data/normalized.mp4")
    write_json(base / "run_config.json", {"frames":5})
    write_json(base / "receiver/decoder_inputs.json", {"indices":[0,2,4],"decoder":{"policy":"official_release"}})
    cfg=dict(frames=5,fps=24,hybrid_selection_root=str(tmp_path / "selected"))
    write_json(run / "run_config.json",cfg)
    source_hash=batch.sha256(run / "data/normalized.mp4")
    protocol=dict(baseline=str(base),source_id="fixture",input_sha256=source_hash)
    monkeypatch.setattr(batch,"verify_selection",lambda _:(protocol,{"indices":[0,2,4]}))
    values={k:1.0 for k in batch.METRICS};values.update(shape=[5,32,32,3],frames=5)
    write_json(run / "quality.json",dict(status="PASSED",source_sha256=source_hash,
        video_sha256=batch.sha256(run / "receiver/reconstruction/sample_0000.mp4"),delivered_mp4=values))
    write_json(run / "output_audit.json",dict(full_input_reconstructed=True))
    write_json(run / "channel_accounting.json",dict(status="PASSED"))
    batch.make_comparison(output)
    assert read_json(output / "RESULT.json")["baseline_repeated_boundary_frames_removed"] == [3]
    assert (output / "review.html").exists()
    batch.hybrid.pts_audit(output / "comparison.mp4",5,24)
