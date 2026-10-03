"""Default adoption preserves prepared inputs, routes both fixes, and resumes."""
from pathlib import Path
import shutil
import subprocess

import pytest

from semantic_transmission import fc_lgvsc as run
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.webvid_ablation import snapshot, Stages


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "baseline"
    for name in run.INPUTS:
        path = source / "run" / name
        if name == "receiver/frames":
            path /= "sample/key_frames_received/0.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    write_json(source / "run/receiver_policy.json", dict(policy=run.base.tail.POLICY,
        t5_precision="fp32", noise_contract={"seed":2025}))
    write_json(source / "run" / run.base.noise.TRACE, {"status":"PASSED"})
    config = source / "run/receiver/decoder_config.py"
    config.write_text("align=5\ncondition_frame_length=5\n")
    return source


@pytest.mark.parametrize("video", [None, *run.base.VIDEOS])
def test_default_and_video_checks_are_read_only(tmp_path, monkeypatch, capsys, video):
    output = tmp_path / "untouched"
    sources = []
    def preflight(source, dest):
        sources.append(source)
        return dict(plan=[], schedule_plan=[], condition_collision_policy=run.base.fix.POLICY,
                    short_schedule_policy=run.paired.fix.POLICY), "signature"
    monkeypatch.setattr(run, "preflight", preflight)
    monkeypatch.setattr(run, "initialize_stage", lambda *_: pytest.fail("check initialized"))
    monkeypatch.setattr(run, "execute", lambda *_: pytest.fail("check ran model"))
    args = ["--check", "--output", str(output)] + (["--video", video] if video else [])
    assert run.main(args) == 0
    assert sources == [run.base.source_for(video or "tv_low_08").resolve()]
    assert not output.exists()
    assert '"model_inference_started": false' in capsys.readouterr().out


def test_initializer_enables_both_fixes_and_decoder_keeps_baseline_config(prepared, tmp_path, monkeypatch):
    output = tmp_path / "default"
    before = snapshot(prepared / "run", run.INPUTS)
    run.initialize(output, dict(source=str(prepared), inputs=before))
    policy = run.read_json(output / "run/receiver_policy.json")
    assert policy["default_method"] == run.POLICY
    assert policy["condition_collision_policy"] == run.base.fix.POLICY
    assert policy["short_schedule_policy"] == run.paired.fix.POLICY
    assert policy["t5_precision"] == "fp32"
    assert policy["noise_reference_sha256"] == sha256(prepared / "run" / run.base.noise.TRACE)
    from semantic_transmission import decoder_runner
    calls = []
    monkeypatch.setattr(decoder_runner, "run", lambda *args, **kwargs: calls.append(kwargs))
    run.reconstruct(output / "run")
    assert calls[0]["decoder"].name == "etri_schedule_fix_decoder.py"
    assert calls[0]["config"].read_bytes() == (prepared / "run/receiver/decoder_config.py").read_bytes()
    assert snapshot(output / "run", run.INPUTS) == snapshot(prepared / "run", run.INPUTS) == before
    assert run.read_json(output / "run" / run.base.fix.TRACE) == []
    assert run.read_json(output / "run" / run.paired.fix.TRACE) == []


def test_preflight_protects_source_paths(tmp_path):
    source = tmp_path / "source"
    for output in (source, source / "child", tmp_path):
        with pytest.raises(ValueError, match="separate"):
            run.preflight(source, output)


def test_protocol_resume_rejects_changed_artifacts_or_policy(prepared, tmp_path, monkeypatch):
    write_json(prepared / "run/run_config.json", dict(frames=9, fps=24))
    write_json(prepared / "run/keyframes.json", dict(indices=[0,8]))
    write_json(prepared / "execution_protocol.json", {})
    write_json(prepared / "RESULT.json", {})
    marker = prepared / "marker"
    marker.write_text("original")
    stages = Stages(prepared, "baseline")
    for name in ("initialize", "reconstruct", "audit", "evaluate", "comparison"):
        stages.step(name, [], ["marker"], lambda _: None)
    baseline = dict(inputs=snapshot(prepared / "run", run.INPUTS), baseline_noise_contract={"seed":2025},
                    plan=[dict(repaired=True)], baseline_code={}, repair_code={})
    monkeypatch.setattr(run.base, "preflight", lambda *_: (baseline,"signature"))
    monkeypatch.setattr(run, "CODE", ())
    monkeypatch.setattr(run.paired, "CODE", ())
    output = tmp_path / "default"
    identity, signature = run.preflight(prepared, output)
    write_json(output / "execution_protocol.json", dict(identity, signature=signature))
    assert run.preflight(prepared, output) == (identity, signature)
    write_json(output / "execution_protocol.json", dict(identity, signature="wrong"))
    with pytest.raises(ValueError, match="inputs/code changed"):
        run.preflight(prepared, output)
    marker.write_text("modified")
    with pytest.raises(ValueError, match="artifacts changed"):
        run.preflight(prepared, tmp_path / "other")


def test_stages_run_one_reconstruction_without_t5_preparation_and_resume(prepared, tmp_path, monkeypatch):
    output = tmp_path / "default"
    identity = dict(source=str(prepared), inputs=snapshot(prepared / "run", run.INPUTS),
                    baseline_noise_contract={"seed":2025})
    calls = []
    def worker(command, log, env, resource):
        name = command[4] if command[2] == run.MODULE else command[3]
        calls.append(name)
        required = {
            "reconstruct": ["receiver/reconstruction/sample_0000.mp4", "receiver/decoder_config.py",
                "receiver/reference_trace.json", "receiver/tail_reference_trace.json",
                run.base.text.TRACE, run.base.noise.TRACE, run.base.fix.TRACE, run.paired.fix.TRACE],
            "audit": ["output_audit.json"],
            "evaluate": ["quality.json", "quality_delivered_mp4.csv", "quality_lossless_frames.csv"],
        }[name]
        for p in required:
            path = output / "run" / p
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test-only artifact")
        write_json(resource, {"fixture":True})
    def review(dest):
        for name in ("RESULT.json", "review.html", "comparison.mp4"):
            (dest/name).write_text("test-only comparison")
    monkeypatch.setattr(run.base.tail.hybrid, "launch", worker)
    monkeypatch.setattr(run, "comparison", review)
    run.execute(output, identity, "test-signature")
    assert calls == ["reconstruct", "audit", "evaluate"]
    completed = snapshot(output, ["stages"])
    run.execute(output, identity, "test-signature")
    assert calls == ["reconstruct", "audit", "evaluate"]
    assert snapshot(output, ["stages"]) == completed
    (output / "run/keyframes.json").write_text("changed")
    with pytest.raises(ValueError, match="artifacts changed"):
        run.execute(output, identity, "test-signature")


def test_review_compares_both_repairs_without_claiming_mitigation(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg unavailable")
    source, output = tmp_path / "baseline", tmp_path / "default"
    video = output / "run/data/normalized.mp4"
    video.parent.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
        "color=c=blue:s=16x16:r=3", "-frames:v", "3", "-c:v", "libx264", "-pix_fmt",
        "yuv420p", str(video)], check=True)
    quality = dict(status="PASSED", source_sha256=sha256(video), video_sha256=sha256(video),
        delivered_mp4=dict(psnr_db=30, ssim=1, lpips_vgg=0, clip=1, dists=0, frames=3, shape=[3,16,16,3]))
    for parent in (source, output):
        target = parent / "run/receiver/reconstruction/sample_0000.mp4"
        target.parent.mkdir(parents=True)
        shutil.copyfile(video, target)
        write_json(parent / "run/quality.json", quality)
    write_json(output / "execution_protocol.json", dict(source=str(source), frames=3, fps=3))
    write_json(output / "run" / run.base.fix.TRACE, [[{"repaired":True}]])
    write_json(output / "run" / run.paired.fix.TRACE, [dict(loop=0, repaired=True)])
    monkeypatch.setattr(run, "audit", lambda _: None)
    run.comparison(output)
    result = run.read_json(output / "RESULT.json")
    assert result["condition_repaired_segments"] == result["schedule_repaired_segments"] == [0]
    assert result["additional_channel_uses"] == 0
    assert result["hallucination_mitigation_verified"] is False
    assert "원본 / 수정 전 FP32 / FC-LGVSC 기본" in (output / "review.html").read_text()
