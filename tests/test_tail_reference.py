import importlib.util
import csv
from pathlib import Path
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest

from semantic_transmission import tail_reference as module
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.webvid_ablation import Stages, snapshot


@pytest.mark.parametrize("frames", [1, 13, 17, 29, 42])
def test_every_reference_is_cropped_before_encoding_and_preserves_recent_order(frames):
    video = np.broadcast_to(np.arange(frames)[None, None, :, None, None], (2, 3, frames, 1, 1))
    observed = []
    def append(vae, pixels, refs, strategies, loop, length, edit):
        observed.append(pixels.copy())
        assert (loop, length, edit) == (4, 5, 0)
        for row in refs:
            row.append(np.zeros((4, 5, 1, 1)))
        return refs, strategies
    refs, strategies = [["key"], ["key"]], ["keep align=5", "keep align=5"]
    result, record = module.append_tail17(append, None, video, refs, strategies, 4, 5, 0)
    expected = [0] * max(0, 17 - frames) + list(range(max(0, frames - 17), frames))
    np.testing.assert_array_equal(observed[0][0, 0, :, 0, 0], expected)
    assert observed[0].shape == (2, 3, 17, 1, 1)
    assert record["source_indices"] == expected and record["encoded_frames"] == 17
    assert result[1] is strategies
    assert record["latent_shapes"] == [[4, 5, 1, 1], [4, 5, 1, 1]]


def test_reference_rejects_invalid_length_or_vae_output():
    video = np.zeros((1, 3, 29, 1, 1))
    with pytest.raises(ValueError, match="five reference"):
        module.append_tail17(None, None, video, [], [], 1, 4, 0)
    with pytest.raises(ValueError, match="positive"):
        module.append_tail17(None, None, video[:, :, :0], [], [], 1, 5, 0)
    def bad(vae, pixels, refs, strategies, *args):
        refs[0].append(np.zeros((4, 8, 1, 1)))
        return refs, strategies
    with pytest.raises(ValueError, match="five-latent"):
        module.append_tail17(bad, None, video, [[]], [""], 1, 5, 0)


def test_decoder_entry_installs_tail_crop_in_actual_append_path(tmp_path, monkeypatch):
    seen = []
    def original(vae, video, refs, strategies, *args):
        seen.append(video[0, 0, :, 0, 0].tolist())
        refs[0].append(np.zeros((4, 5, 1, 1)))
        return refs, strategies
    utils = types.SimpleNamespace(append_generated=original)
    monkeypatch.setitem(sys.modules, "opensora", types.ModuleType("opensora"))
    parent = types.ModuleType("opensora.utils")
    parent.inference_utils = utils
    monkeypatch.setitem(sys.modules, "opensora.utils", parent)
    trace = tmp_path / "trace.json"
    monkeypatch.setenv("ETRI_TAIL_REFERENCE_TRACE", str(trace))
    path = module.REPO / "scripts/etri_tail17_decoder.py"
    spec = importlib.util.spec_from_file_location("tail17_entry", path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    def decoder(path, run_name):
        assert Path(path).name == "etri_decoder_probe.py" and run_name == "__main__"
        video = np.arange(29)[None, None, :, None, None]
        utils.append_generated(None, video, [[]], [""], 1, 5, 0)
    monkeypatch.setattr(entry.runpy, "run_path", decoder)
    entry.main()
    assert seen == [list(range(12, 29))]
    module.validate_trace(module.read_json(trace), 1)


def test_trace_validation_rejects_stale_reference_and_missing_segments():
    row = dict(loop=1, input_frames=29, source_indices=list(range(12, 29)), encoded_frames=17,
               latent_shapes=[[4, 5, 1, 1]], policy=module.POLICY)
    module.validate_trace([row], 1)
    with pytest.raises(ValueError, match="last 17"):
        module.validate_trace([dict(row, source_indices=list(range(17)))], 1)
    with pytest.raises(ValueError, match="missing"):
        module.validate_trace([row], 2)


def test_check_is_read_only_and_never_launches_decoder(tmp_path, monkeypatch, capsys):
    output = tmp_path / "not_created"
    monkeypatch.setattr(module, "preflight", lambda _: ({}, "test", {"status": "READY_FOR_USER_EXECUTION"}))
    monkeypatch.setattr(module, "execute", lambda *_: pytest.fail("check started inference"))
    monkeypatch.setattr(module, "write_json", lambda *_: pytest.fail("check wrote files"))
    module.main(["--check", "--output", str(output)])
    assert "READY_FOR_USER_EXECUTION" in capsys.readouterr().out
    assert not output.exists()


@pytest.fixture
def prepared_source(tmp_path, monkeypatch):
    source = tmp_path / "v2"
    output = tmp_path / "tail17"
    monkeypatch.setattr(module, "SOURCE", source)
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "CODE", ())
    monkeypatch.setattr(module.caption_revision, "preflight", lambda: None)
    for name in module.INPUTS:
        path = source / "run" / name
        if name == "receiver/frames":
            path = path / "sample/key_frames_received/0.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    write_json(source / "run/run_config.json", dict(frames=1440, fps=24, seed=2025))
    write_json(source / "RESULT.json", dict(status="PASS_60S_HYBRID_RECONSTRUCTION", keyframes=79))
    write_json(source / "execution_protocol.json", dict(code={}))
    for name in ("initialize", *[s for s, _, _ in module.hybrid.job_plan(True)], "caption-revision-comparison"):
        write_json(source / f"stages/{name}.json", dict(status="PASSED", required=["run/run_config.json"],
                   artifacts=snapshot(source, ["run/run_config.json"])))
    return source, output


def test_preflight_rejects_modified_source_or_resume_identity(prepared_source):
    source, output = prepared_source
    identity, signature, report = module.preflight(output)
    assert not output.exists() and report["reruns_transmission"] is False
    assert report["diffusion_noise_tensor_identity_verified"] is False
    assert report["first_segment_collision_fixed"] is False
    write_json(output / "execution_protocol.json", dict(identity, signature=signature))
    module.preflight(output)
    (source / "run/received/metadata.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="inputs/code changed"):
        module.preflight(output)
    (source / "run/run_config.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="v2 artifacts changed"):
        module.preflight(output)


def test_initialize_reuses_received_inputs_and_resume_preserves_source(prepared_source):
    source, output = prepared_source
    original = snapshot(source / "run", module.INPUTS)
    identity, signature, _ = module.preflight(output)
    stages = Stages(output, signature)
    required = [f"run/{p}" for p in module.INPUTS] + ["run/receiver_policy.json"]
    stages.step("initialize", ["run"], required, lambda _: module.initialize(source, output, original))
    assert snapshot(output / "run", module.INPUTS) == original
    assert snapshot(source / "run", module.INPUTS) == original
    # Later outputs must not invalidate the initialization receipt.
    write_json(output / "run/quality.json", {"later": True})
    Stages(output, signature).step("initialize", ["run"], required,
                                  lambda _: pytest.fail("repeated initialization"))
    key = output / "run/receiver/frames/sample/key_frames_received/0.png"
    key.write_bytes(b"changed locally")
    assert snapshot(source / "run", module.INPUTS) == original
    with pytest.raises(ValueError, match="artifacts changed"):
        Stages(output, signature).step("initialize", ["run"], required, lambda _: None)


def test_reject_output_that_could_replace_existing_runs(tmp_path):
    source = tmp_path / "v2"
    for bad in (source, tmp_path, source / "run"):
        with pytest.raises(ValueError, match="separate sibling"):
            module.validate_destination(bad, source)
    occupied = tmp_path / "v1"
    occupied.mkdir()
    with pytest.raises(ValueError, match="existing output"):
        module.validate_destination(occupied, source)


def test_reconstruct_uses_tail_entry_and_retains_alignment(tmp_path, monkeypatch):
    from semantic_transmission import codec_transport, decoder_runner
    run = tmp_path / "run"
    write_json(run / "run_config.json", {"seed": 2025})
    write_json(run / "receiver_policy.json", {"policy": module.POLICY})
    write_json(run / "receiver/decoder_inputs.json", {})
    monkeypatch.setattr(codec_transport, "decoder_config_text", lambda *_: "condition_frame_length=5\n")
    calls = []
    monkeypatch.setattr(decoder_runner, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    module.reconstruct(run)
    kwargs = calls[0][1]
    assert kwargs["decoder"].name == "etri_tail17_decoder.py"
    text = kwargs["config"].read_text()
    assert "conditioning_alignment='official_release'" in text and "align=5" in text
    assert "last_17_frames_before_vae" in text
    assert kwargs["environment"]["ETRI_TAIL_REFERENCE_TRACE"].endswith("tail_reference_trace.json")


def test_audit_checks_full_timeline_and_tail_trace(tmp_path, monkeypatch):
    write_json(tmp_path / "keyframes.json", dict(indices=[0,12,32]))
    write_json(tmp_path / "receiver/tail_reference_trace.json", [])
    called = []
    monkeypatch.setattr(module.hybrid, "hybrid_output_audit", lambda run: called.append(run))
    with pytest.raises(ValueError, match="missing"):
        module.audit(tmp_path)
    assert called == [tmp_path]


@pytest.mark.parametrize("t5_cache", [False, True])
def test_pipeline_writes_honest_comparison_and_resumes_without_workers(prepared_source, monkeypatch, t5_cache):
    # Exercise orchestration/reporting with synthetic CPU media, never model inference.
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg required for comparison assembly")
    source, output = prepared_source
    run = source / "run"
    video = run / "data/normalized.mp4"
    video.unlink()
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
        "color=c=blue:s=16x16:r=3", "-frames:v", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True)
    target = run / "receiver/reconstruction/sample_0000.mp4"
    target.parent.mkdir()
    shutil.copyfile(video, target)
    write_json(run / "run_config.json", dict(frames=3, fps=3, seed=2025))
    write_json(run / "keyframes.json", dict(indices=[0,2]))
    write_json(run / "captions.json", [dict(text="A blue test frame.")])
    write_json(run / "channel_accounting.json", dict(total_complex_channel_uses=42))
    (run / "receiver/frames/sample/key_frames_received/2.png").write_bytes(b"second key fixture")
    with (run / "receiver/metadata.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "text", "flow"])
        writer.writeheader()
        writer.writerow(dict(path="clips/sample/00000.mp4", text="A blue test frame.", flow=0.0))
    quality = dict(status="PASSED", video_sha256=sha256(target), source_sha256=sha256(video),
                   delivered_mp4=dict(psnr_db=30, ssim=1, lpips_vgg=0, clip=1, dists=0))
    write_json(run / "quality.json", quality)
    identity = dict(inputs=snapshot(run, module.INPUTS))
    if t5_cache:
        identity["text_encoder"] = {"policy": "test"}
    calls = []
    def worker(command, log, env, resource):
        name = command[4] if command[2] == module.MODULE else command[3]
        calls.append(name)
        new = output / "run"
        if name == "prepare-text":
            from semantic_transmission.text_embedding_cache import POLICY, REPORT
            write_json(new / REPORT, dict(policy=POLICY, cache_hits=0, cache_misses=1))
        elif name == "reconstruct":
            dest = new / "receiver/reconstruction/sample_0000.mp4"
            dest.parent.mkdir()
            shutil.copyfile(video, dest)
            (new / "receiver/decoder_config.py").write_text("align=5\n")
            for trace in ("reference_trace", "tail_reference_trace"):
                write_json(new / f"receiver/{trace}.json", [])
            if t5_cache:
                write_json(new / "receiver/text_embedding_trace.json", {"fixture": True})
        elif name == "audit":
            write_json(new / "output_audit.json", dict(status="PASSED"))
        elif name == "evaluate":
            write_json(new / "quality.json", quality)
            for csv_name in ("quality_delivered_mp4", "quality_lossless_frames"):
                (new / f"{csv_name}.csv").write_text("synthetic fixture\n")
        else:
            pytest.fail(f"unexpected worker {command}")
        write_json(resource, {"fixture": True})
    monkeypatch.setattr(module.hybrid, "launch", worker)
    monkeypatch.setattr(module.hybrid, "settings", lambda _: {"python": sys.executable})
    if t5_cache:
        module.execute(output, identity, "fixture", prepare_only=True)
        assert calls == ["prepare-text"]
        assert not (output / "run/receiver/reconstruction").exists()
    module.execute(output, identity, "fixture")
    assert calls == (["prepare-text"] if t5_cache else []) + ["reconstruct", "audit", "evaluate"]
    result = module.read_json(output / "RESULT.json")
    assert result["additional_channel_uses"] == 0 and result["channel_uses"] == 42
    assert result["hallucination_mitigation_verified"] is False
    assert result["first_segment_collision_fixed"] is False
    assert result["pairing"]["diffusion_noise_tensor_identity_verified"] is False
    if t5_cache:
        assert result["legacy_cpu_fp32_equivalence_verified"] is False
        assert result["pairing"]["text_encoder_precision_changed"] is True
    assert "마지막 17프레임 참조" in (output / "review.html").read_text()
    assert snapshot(run, module.INPUTS) == identity["inputs"]
    monkeypatch.setattr(module.hybrid, "launch", lambda *_: pytest.fail("completed pipeline reran workers"))
    module.execute(output, identity, "fixture")


def test_t5_reconstruct_dispatch_keeps_existing_steps_and_uses_saved_conditions(tmp_path, monkeypatch):
    from semantic_transmission import codec_transport, decoder_runner, text_embedding_cache as cache
    run = tmp_path / "run"
    write_json(run / "run_config.json", {"seed": 2025})
    write_json(run / "receiver_policy.json", {"policy": module.POLICY, "text_encoder_policy": cache.POLICY})
    write_json(run / "receiver/decoder_inputs.json", {})
    write_json(run / cache.REPORT, {"policy": cache.POLICY})
    monkeypatch.setattr(codec_transport, "decoder_config_text", lambda *_: "scheduler=dict(num_sampling_steps=30)\n")
    calls = []
    monkeypatch.setattr(decoder_runner, "run", lambda *a, **k: calls.append(k))
    module.reconstruct(run)
    assert calls[0]["decoder"].name == "etri_t5_cached_decoder.py"
    assert "num_sampling_steps=30" in calls[0]["config"].read_text()
    assert Path(calls[0]["environment"]["ETRI_TEXT_EMBEDDINGS"]) == run / cache.REPORT
