import json
import os
from pathlib import Path
import sys

import pytest

from semantic_transmission import lgvsc_qwen_baseline as experiment
from semantic_transmission import qwen35_pairs_video as current
from semantic_transmission.artifacts import write_json


def test_source_window_requires_explicit_opt_in_and_correct_duration():
    source = dict(fps=24, frames=384, source_duration_seconds=20.6,
                  recipe=dict(duration_limit=16, source_start_seconds=0))
    with pytest.raises(ValueError, match="declared"):
        current.validate_source_window(source)
    current.validate_source_window(source, 16)
    with pytest.raises(ValueError, match="source window"):
        current.validate_source_window(dict(source, frames=360), 16)
    source["recipe"]["source_start_seconds"] = 2
    with pytest.raises(ValueError, match="start at zero"):
        current.validate_source_window(source, 16)


def test_baseline_is_sparse_release_policy_with_shared_channel_settings(tmp_path):
    shared = dict(models={"pllava": "real-weights"}, seed=2025, channel_seed=42,
                  snr_db=10, steps=30, flash_attn=True,
                  internvl_offload_layers=3, internvl_compact_kv_cache=True)
    cfg = experiment.baseline_config(dict(receiver=shared), tmp_path / "common", tmp_path)
    assert experiment.candidate_indices() == list(range(0, 384, 4))
    assert experiment.candidate_indices()[-1] == 380
    assert cfg["selector"] == "skem" and not cfg["internvl_8bit"]
    assert cfg["threshold"] == .35 and cfg["paper_caption"] and cfg["paper_flow"]
    assert cfg["semantic_clip_policy"] == cfg["decoder_policy"] == cfg["concatenation_policy"] == "official_release"
    assert "max_gap" not in cfg and "key_budget" not in cfg and "reference_policy" not in cfg
    assert all(cfg[k] == shared[k] for k in ("seed", "channel_seed", "snr_db", "steps"))


def test_no_placeholder_keys_remain_and_both_arms_share_source_bytes(tmp_path):
    source_dir = tmp_path / "common/source_frames"
    source_dir.mkdir(parents=True)
    for i in range(384):
        (source_dir / f"{i}.png").write_bytes(b"source-frame")
    video = tmp_path / "common/normalized.mp4"
    video.write_bytes(b"same-video")
    write_json(tmp_path / "common/source_prepared.json", dict(source_frames=str(source_dir),
        normalized_video=str(video), normalized_sha256="digest", frames=384))
    experiment.initialize_baseline(tmp_path, dict(baseline={}))
    run = tmp_path / "lgvsc/run"
    assert not (run / "keyframes.json").exists()
    assert (run / "data/normalized.mp4").resolve() == video
    lines = (run / "data/frames/sample/frames.csv").read_text().splitlines()
    assert len(lines) == 97 and lines[-1].endswith("/380.png")
    assert (run / "data/frames/sample/383.png").is_file()


def test_release_worker_runs_selector_without_forcing_key_count(tmp_path, monkeypatch):
    from semantic_transmission import workers, codec_transport
    write_json(tmp_path / "protocol.json", {})
    run = tmp_path / "lgvsc/run"
    write_json(run / "run_config.json", dict(frames=384, input="common.mp4"))
    received = []
    monkeypatch.setattr(workers, "select", lambda cfg, repo, r: received.append(("select", cfg)))
    monkeypatch.setattr(codec_transport, "send", lambda cfg, repo, r: received.append(("send", cfg)))
    experiment.worker(tmp_path, "select")
    experiment.worker(tmp_path, "send")
    assert [cfg["frames"] for _, cfg in received] == [381, 381]
    assert read_config(run)["frames"] == 384
    assert not (run / "keyframes.json").exists()  # The launcher never invents two keys.


def read_config(run):
    return json.loads((run / "run_config.json").read_text())


def test_baseline_decoder_does_not_install_current_receiver_repairs(tmp_path, monkeypatch):
    from semantic_transmission import codec_transport, decoder_runner
    run = tmp_path / "run"
    write_json(run / "keyframes.json", dict(indices=[0, 128, 380]))
    write_json(run / "receiver/decoder_inputs.json", {})
    monkeypatch.setattr(codec_transport, "decoder_config_text", lambda *_: "# test\n")
    calls = []
    monkeypatch.setattr(decoder_runner, "run", lambda *a, **kw: calls.append(kw))
    experiment.reconstruct_baseline(run, {})
    text = (run / "receiver/decoder_config.py").read_text()
    assert "conditioning_alignment='official_release'" in text
    assert "align=5" in text and "reuse_diffusion_model=False" in text
    assert calls[0]["decoder"].name == "etri_decoder_probe.py"
    assert "ETRI_TAIL_REFERENCE_TRACE" not in calls[0]["environment"]
    write_json(run / "keyframes.json", dict(indices=[0, 4, 380]))
    with pytest.raises(ValueError, match="without silently enabling"):
        experiment.reconstruct_baseline(run, {})
    assert len(calls) == 1


def test_audit_preserves_release_frame_mapping_and_rejects_tail17(tmp_path, monkeypatch):
    from semantic_transmission import etri_60s_check
    run = tmp_path / "lgvsc/run"
    write_json(run / "keyframes.json", dict(indices=[0, 128, 380]))
    video = run / "receiver/reconstruction/sample_0000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    write_json(run / "receiver/reference_trace.json", [dict(generated_reference_shape=[1, 3, 129, 320, 576])])
    write_json(run / "receiver/decoder_inputs.json", dict(indices=[0, 128, 380],
        decoder=dict(policy="official_release", concatenation_policy="official_release")))
    write_json(run / "channel_accounting.json", dict(metadata_exact_match=True))
    checked = []
    monkeypatch.setattr(etri_60s_check, "pts_audit", lambda path, n, fps: checked.append(n))
    monkeypatch.setattr(experiment, "probe", lambda p: dict(width=576, height=320))
    experiment.audit_baseline(tmp_path, {})
    audit = json.loads((run / "output_audit.json").read_text())
    assert checked == [382]  # 381 source positions + the published repeated boundary.
    assert audit["output_source_indices"].count(128) == 2
    assert audit["omitted_source_tail_frames"] == 3
    assert audit["quality_acceptance"] == "NOT_EVALUATED"
    write_json(run / "receiver/reference_trace.json", [dict(generated_reference_shape=[1, 3, 17, 320, 576])])
    with pytest.raises(ValueError, match="complete previous"):
        experiment.audit_baseline(tmp_path, {})


def test_sequence_stops_on_failure_and_resumes_verified_previous_phases(tmp_path, monkeypatch, capsys):
    p = dict(signature="test", baseline=dict(seed=2025), interpreters=dict(python=sys.executable))
    events = []
    fail = [True]
    def launch(command, log, env, resource, **kwargs):
        stage = command[-1]
        events.append(stage)
        if stage == "prepare":
            write_json(tmp_path / "common/source_prepared.json", {"input": "frozen"})
        elif fail[0]:
            write_json(tmp_path / "qwen/checkpoint.json", {"saved": True})
            raise RuntimeError("worker failed")
        else:
            assert (tmp_path / "qwen/checkpoint.json").is_file()
            write_json(tmp_path / "qwen/RESULT.json", {"complete": True})
        write_json(resource, {"returncode": 0})
    def baseline(root, protocol):
        events.append("lgvsc")
        write_json(root / "lgvsc/result.json", {"complete": True})
    monkeypatch.setattr(experiment, "launch", launch)
    monkeypatch.setattr(experiment, "baseline", baseline)
    monkeypatch.setattr(experiment, "finalize", lambda *_: dict(status="DONE", videos={}))
    with pytest.raises(RuntimeError, match="worker failed"):
        experiment.execute(tmp_path, p)
    assert events == ["prepare", "lgvsc", "current"]
    assert "전체 3/3 완료" not in capsys.readouterr().out
    fail[0] = False
    experiment.execute(tmp_path, p)
    assert events == ["prepare", "lgvsc", "current", "current"]
    assert "REUSE 02-lgvsc" in capsys.readouterr().out
    (tmp_path / "common/source_prepared.json").write_text("changed")
    with pytest.raises(ValueError, match="artifacts changed"):
        experiment.execute(tmp_path, p)


def test_real_small_worker_progress_log_and_failure_propagation(tmp_path, capsys):
    code = ("import os,json,time; "
            "open(os.environ['QUALITY_PROGRESS_FILE'],'w').write(json.dumps({'done':1,'total':4})); "
            "print('worker detail',flush=True); time.sleep(.25)")
    experiment.launch([sys.executable, "-c", code], tmp_path / "ok.log", dict(os.environ),
                      tmp_path / "ok.json", label="test", interval=.05)
    out = capsys.readouterr().out
    assert "1/4 (25.0%)" in out and "100% 완료" in out
    assert "worker detail" in (tmp_path / "ok.log").read_text()
    with pytest.raises(RuntimeError, match="exit 7"):
        experiment.launch([sys.executable, "-c", "print('failure evidence'); raise SystemExit(7)"],
                          tmp_path / "bad.log", dict(os.environ), tmp_path / "bad.json", interval=.05)
    out = capsys.readouterr().out
    assert "failure evidence" in out and "100% 완료" not in out
    assert json.loads((tmp_path / "bad.json").read_text())["returncode"] == 7


def test_unknown_progress_never_fabricates_a_percentage():
    assert "%" not in experiment.progress_message("loading", 60)
    assert "%" not in experiment.progress_message("loading", 60, dict(done=3, total=2))
