import csv
import shutil
import subprocess
import sys

import pytest

from semantic_transmission import precision_reconstruction as run, precision_review as review
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.webvid_ablation import snapshot


@pytest.fixture
def source(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "fp32"
    monkeypatch.setattr(run, "SOURCE", source)
    monkeypatch.setattr(run, "PREVIOUS", source)
    monkeypatch.setattr(run, "REPO", tmp_path)
    monkeypatch.setattr(run, "CODE", ())
    monkeypatch.setattr(run.caption_revision, "preflight", lambda: None)
    monkeypatch.setattr(run.text.legacy, "model_inventory", lambda _: {})
    for name in run.tail.INPUTS:
        path = source / "run" / name
        if name == "receiver/frames":
            path /= "sample/key_frames_received/0.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    write_json(source / "run/run_config.json", dict(frames=3, fps=3, seed=2025, models={"t5": "fixture"}))
    write_json(source / "run/keyframes.json", {"indices": [0, 2]})
    write_json(source / "run/receiver/decoder_inputs.json", {"decoder": {"steps": 3}})
    write_json(source / "RESULT.json", {"status": "PASS_60S_HYBRID_RECONSTRUCTION"})
    write_json(source / "execution_protocol.json", {"code": {}})
    for stage in ("initialize", *[s for s, _, _ in run.tail.hybrid.job_plan(True)], "caption-revision-comparison"):
        write_json(source / f"stages/{stage}.json", dict(status="PASSED", required=["run/run_config.json"],
                   artifacts=snapshot(source, ["run/run_config.json"])))
    return source, output


def test_precision_contract_changes_cache_but_not_scoped_noise(source):
    _, output = source
    fp32, signature, _ = run.preflight(output, "fp32")
    bf16, _, _ = run.preflight(output, "bf16")
    assert fp32["noise_contract"] == bf16["noise_contract"]
    assert fp32["t5_device"] == "cpu" and bf16["t5_device"] == "cuda"
    assert not output.exists()
    write_json(output / "execution_protocol.json", dict(fp32, signature=signature))
    with pytest.raises(ValueError, match="inputs/code changed"):
        run.preflight(output, "bf16")


def test_check_does_not_initialize_models_or_outputs(source, monkeypatch, capsys):
    _, output = source
    monkeypatch.setattr(run, "execute", lambda *a, **k: pytest.fail("check launched inference"))
    run.main(["--check", "--output", str(output)])
    assert '"t5_compute_dtype": "fp32"' in capsys.readouterr().out
    assert not output.exists()


def test_unrecorded_legacy_run_is_not_accepted_as_noise_reference(source):
    source, output = source
    path = source / "run/receiver/generation_noise.json"
    write_json(path, {"status": "RUNNING"})
    with pytest.raises(ValueError, match="incomplete"):
        run.preflight(output, "fp32", path)


def test_decoder_keeps_video_settings_and_receives_noise_policy(source, monkeypatch):
    _, output = source
    identity, _, _ = run.preflight(output, "fp32")
    run.initialize(output, identity)
    target = output / "run"
    write_json(target / run.text.REPORT, dict(policy=run.text.POLICY, contract={"compute_dtype": "fp32"}))
    from semantic_transmission import decoder_runner
    calls = []
    monkeypatch.setattr(run.tail, "decoder_config", lambda _: "dtype='bf16'\nalign=5\nsteps=30\n")
    monkeypatch.setattr(decoder_runner, "run", lambda *a, **kw: calls.append(kw))
    run.reconstruct(target)
    assert calls[0]["decoder"].name == "etri_precision_decoder.py"
    assert calls[0]["environment"]["ETRI_PRECISION_RUN"] == str(target)
    assert "steps=30" in calls[0]["config"].read_text()
    assert run.read_json(target / "receiver_policy.json")["noise_contract"] == identity["noise_contract"]


def test_pipeline_assembles_review_and_resumes_completed_work_without_inference(source, monkeypatch):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg required")
    previous, output = source
    old = previous / "run"
    video = old / "data/normalized.mp4"
    video.unlink()
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
        "color=c=blue:s=16x16:r=3", "-frames:v", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True)
    target = old / "receiver/reconstruction/sample_0000.mp4"
    target.parent.mkdir()
    shutil.copyfile(video, target)
    write_json(old / "captions.json", [{"text": "A blue test frame."}])
    write_json(old / "channel_accounting.json", {"total_complex_channel_uses": 42})
    (old / "receiver/frames/sample/key_frames_received/2.png").write_bytes(b"key 2")
    with (old / "receiver/metadata.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "text", "flow"])
        writer.writeheader()
        writer.writerow(dict(path="clips/sample/00000.mp4", text="A blue test frame.", flow=0.0))
    quality = dict(status="PASSED", video_sha256=sha256(target), source_sha256=sha256(video),
                   delivered_mp4=dict(psnr_db=30, ssim=1, lpips_vgg=0, clip=1, dists=0,
                                       frames=3, shape=[3, 16, 16, 3]))
    write_json(old / "quality.json", quality)
    identity, signature, _ = run.preflight(output, "fp32")
    calls = []
    def worker(command, log, env, resource):
        stage = command[4] if command[2] == run.MODULE else command[3]
        calls.append(stage)
        new = output / "run"
        if stage == "prepare-text":
            write_json(new / run.text.REPORT, {"contract": {"device": "cpu"}})
        elif stage == "reconstruct":
            dest = new / "receiver/reconstruction/sample_0000.mp4"
            dest.parent.mkdir()
            shutil.copyfile(video, dest)
            (new / "receiver/decoder_config.py").write_text("steps=30\n")
            for trace in ("reference_trace", "tail_reference_trace", "text_embedding_trace", "generation_noise"):
                write_json(new / f"receiver/{trace}.json", {"fixture": True})
        elif stage == "audit":
            write_json(new / "output_audit.json", {"status": "PASSED"})
        elif stage == "evaluate":
            write_json(new / "quality.json", quality)
            for name in ("quality_delivered_mp4", "quality_lossless_frames"):
                (new / f"{name}.csv").write_text("fixture\n")
        else:
            pytest.fail(stage)
        write_json(resource, {"fixture": True})
    monkeypatch.setattr(run.tail.hybrid, "launch", worker)
    monkeypatch.setattr(run.tail.hybrid, "settings", lambda _: {"python": sys.executable})
    run.execute(output, identity, signature, prepare_only=True)
    assert calls == ["prepare-text"]
    # Reproduce the actual post-inference failure on list-valued metadata.
    with pytest.raises(TypeError, match="list.__format__"):
        run.execute(output, identity, signature)
    assert calls == ["prepare-text", "reconstruct", "audit", "evaluate"]
    completed = {p.name: sha256(p) for p in (output / "stages").glob("*.json") if p.stem != "comparison"}
    monkeypatch.setattr(run, "comparison", review.comparison)
    monkeypatch.setattr(run.tail.hybrid, "launch", lambda *a, **k: pytest.fail("repair reran model/evaluation worker"))
    run.execute(output, identity, signature)
    assert completed == {p.name: sha256(p) for p in (output / "stages").glob("*.json") if p.stem != "comparison"}
    assert list((output / "failed_attempts").glob("comparison_*/stages/comparison.json"))
    result = run.read_json(output / "RESULT.json")
    assert result["noise_reference_verified"] is False
    assert result["pairing"]["diffusion_noise_tensor_identity_verified"] is False
    assert result["hallucination_mitigation_verified"] is False
    assert result["quality_after"]["shape"] == [3, 16, 16, 3]
    assert result["report_builder"]["sha256"] == sha256(review.__file__)
    assert "잡음이 같은 비교가 아닙니다" in (output / "review.html").read_text()
    assert snapshot(old, run.tail.INPUTS) == identity["inputs"]
    monkeypatch.setattr(run.tail.hybrid, "launch", lambda *a, **k: pytest.fail("completed worker reran"))
    run.execute(output, identity, signature)


def test_report_only_renders_scores_and_retains_metadata():
    metrics = dict(psnr_db=15.2, ssim=.65, lpips_vgg=.38, clip=.94, dists=.18,
                   frames=1440, shape=[1440, 320, 576, 3])
    html = review.metric_rows(metrics, metrics)
    assert html.count('<tr>') == 5 and '15.2000' in html
    assert 'frames' not in html and 'shape' not in html
    assert metrics['shape'] == [1440, 320, 576, 3]
    for invalid in ([.38], None, float('nan'), float('inf'), True):
        with pytest.raises(ValueError, match='lpips_vgg'):
            review.metric_rows(metrics, dict(metrics, lpips_vgg=invalid))


def test_public_adapter_only_replaces_reporting(monkeypatch):
    original = run.comparison
    def main(args):
        assert args == ['--check'] and run.comparison is review.comparison
        return 0
    monkeypatch.setattr(run, 'main', main)
    assert review.main(['--check']) == 0
    assert run.comparison is original
