import importlib.util
from pathlib import Path
import sys
import types

import pytest

from semantic_transmission import condition_repair as run, condition_collision as fix
from semantic_transmission.artifacts import write_json, sha256
from semantic_transmission.webvid_ablation import snapshot


def test_check_does_not_initialize_or_generate(tmp_path, monkeypatch, capsys):
    output = tmp_path / "untouched"
    monkeypatch.setattr(run, "preflight", lambda *_: ({"plan": [{"repaired": True}]}, "signature"))
    monkeypatch.setattr(run, "initialize", lambda *_: pytest.fail("check initialized files"))
    monkeypatch.setattr(run, "execute", lambda *_: pytest.fail("check started reconstruction"))
    run.main(["--check", "--video", "person_walk", "--output", str(output)])
    assert not output.exists()
    assert '"model_inference_started": false' in capsys.readouterr().out


def test_existing_source_and_unrelated_outputs_are_protected(tmp_path):
    source = tmp_path / "source"
    for output in (source, source / "child", tmp_path):
        with pytest.raises(ValueError, match="separate"):
            run.preflight(source, output)


def test_initializer_reuses_exact_inputs_cache_and_baseline_noise(tmp_path):
    source, output = tmp_path / "baseline", tmp_path / "fixed"
    for name in run.INPUTS:
        path = source / "run" / name
        if name == "receiver/frames":
            path /= "sample/key_frames_received/0.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    write_json(source / "run/receiver_policy.json", {"policy": run.tail.POLICY, "noise_contract": {"seed": 2025}})
    write_json(source / "run" / run.noise.TRACE, {"status": "PASSED"})
    before = snapshot(source / "run", run.INPUTS)
    run.initialize(output, dict(source=str(source), inputs=before))
    assert snapshot(output / "run", run.INPUTS) == before
    assert snapshot(source / "run", run.INPUTS) == before
    policy = run.read_json(output / "run/receiver_policy.json")
    assert policy["condition_collision_policy"] == fix.POLICY
    assert policy["noise_contract"] == {"seed": 2025}
    assert policy["noise_reference_sha256"] == sha256(source / "run" / run.noise.TRACE)


def test_reconstruction_uses_new_entry_and_unchanged_config(tmp_path, monkeypatch):
    source, target = tmp_path / "baseline", tmp_path / "fixed/run"
    path = source / "run/receiver/decoder_config.py"
    path.parent.mkdir(parents=True)
    path.write_text("align=5\ncondition_frame_length=5\n")
    reference = source / "run" / run.noise.TRACE
    write_json(reference, {"status": "PASSED"})
    write_json(target / "receiver_policy.json", dict(paired_source=str(source), noise_reference=str(reference),
        noise_reference_sha256=sha256(reference)))
    from semantic_transmission import decoder_runner
    calls = []
    monkeypatch.setattr(decoder_runner, "run", lambda *args, **kwargs: calls.append(kwargs))
    run.reconstruct(target)
    assert calls[0]["decoder"].name == "etri_condition_fix_decoder.py"
    assert calls[0]["config"].read_bytes() == path.read_bytes()
    assert calls[0]["environment"]["ETRI_PRECISION_RUN"] == str(target)
    assert run.read_json(target / fix.TRACE) == []


def test_actual_decoder_entry_routes_mask_through_guard(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    functions = fix.upstream_functions(run.REPO)
    functions.apply_mask_strategy.__globals__["torch"] = torch
    utils = types.SimpleNamespace(**vars(functions))
    parent = types.ModuleType("opensora.utils")
    parent.inference_utils = utils
    monkeypatch.setitem(sys.modules, "opensora", types.ModuleType("opensora"))
    monkeypatch.setitem(sys.modules, "opensora.utils", parent)
    monkeypatch.setenv("ETRI_PRECISION_RUN", str(tmp_path))
    write_json(tmp_path / "receiver_policy.json", {"condition_collision_policy": fix.POLICY})
    write_json(tmp_path / "keyframes.json", {"indices": [0, 8]})
    path = run.REPO / "scripts/etri_condition_fix_decoder.py"
    spec = importlib.util.spec_from_file_location("collision_entry", path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    def decoder(path, run_name):
        assert Path(path).name == "etri_precision_decoder.py" and run_name == "__main__"
        z = torch.zeros(1, 1, 3, 1, 1)
        refs = [[torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1)*2]]
        utils.apply_mask_strategy(z, refs, ["0;0,1,0,-1,1"], 0, align=5)
        assert z.flatten().tolist() == [1, 0, 2]
    monkeypatch.setattr(entry.runpy, "run_path", decoder)
    entry.main()
    assert run.read_json(tmp_path / fix.TRACE)[0][0]["repaired"]

def test_stages_resume_and_never_prepare_text_again(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "output"
    for name in run.INPUTS:
        path = source / "run" / name
        if name == "receiver/frames":
            path /= "sample/key_frames_received/0.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    write_json(source / "run/receiver_policy.json", {"policy": run.tail.POLICY, "noise_contract": {"seed": 2025}})
    write_json(source / "run" / run.noise.TRACE, {"status": "PASSED"})
    identity = dict(source=str(source), inputs=snapshot(source / "run", run.INPUTS),
                    baseline_noise_contract={"seed": 2025})
    calls = []
    def worker(command, log, env, resource):
        name = command[4] if command[2] == run.MODULE else command[3]
        calls.append(name)
        required = {
            "reconstruct": ["receiver/reconstruction/sample_0000.mp4", "receiver/decoder_config.py",
                "receiver/reference_trace.json", "receiver/tail_reference_trace.json",
                run.text.TRACE, run.noise.TRACE, fix.TRACE],
            "audit": ["output_audit.json"],
            "evaluate": ["quality.json", "quality_delivered_mp4.csv", "quality_lossless_frames.csv"],
        }[name]
        for p in required:
            path = output / "run" / p
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test-only artifact")
        write_json(resource, {"fixture": True})
    def review(dest):
        for name in ("RESULT.json", "review.html", "comparison.mp4"):
            (dest/name).write_text("test-only comparison")
    monkeypatch.setattr(run.tail.hybrid, "launch", worker)
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


def test_review_handles_metric_metadata_and_keeps_quality_pending(tmp_path, monkeypatch):
    import shutil
    import subprocess
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg unavailable")
    source, output = tmp_path / "baseline", tmp_path / "fixed"
    video = output / "run/data/normalized.mp4"
    video.parent.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
        "color=c=blue:s=16x16:r=3", "-frames:v", "3", "-c:v", "libx264", "-pix_fmt",
        "yuv420p", str(video)], check=True)
    quality = dict(status="PASSED", source_sha256=sha256(video), video_sha256=sha256(video),
        delivered_mp4=dict(psnr_db=30, ssim=1, lpips_vgg=0, clip=1, dists=0, frames=3, shape=[3, 16, 16, 3]))
    for parent in (source, output):
        target = parent / "run/receiver/reconstruction/sample_0000.mp4"
        target.parent.mkdir(parents=True)
        shutil.copyfile(video, target)
        write_json(parent / "run/quality.json", quality)
    write_json(output / "run/receiver_policy.json", dict(paired_source=str(source)))
    write_json(output / "run/run_config.json", dict(frames=3, fps=3))
    write_json(output / "run" / fix.TRACE, [[{"repaired": True}]])
    monkeypatch.setattr(run, "audit", lambda _: None)
    run.comparison(output)
    result = run.read_json(output / "RESULT.json")
    assert result["repaired_segments"] == 1 and result["additional_channel_uses"] == 0
    assert result["noise_reference_verified"] is True
    assert result["hallucination_mitigation_verified"] is False
    assert (output / "comparison.mp4").is_file()
