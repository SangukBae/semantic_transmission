"""Duration, scope, resume and failure checks; GPU integration is a separate command."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from semantic_transmission import etri_60s_check as app
from semantic_transmission.artifacts import sha256, write_json


@pytest.fixture
def canonical(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/webvid5.json").write_bytes((repo / "configs/webvid5.json").read_bytes())
    write_json(tmp_path / ".local/model_paths.json", {})
    source = tmp_path / "input.mp4"
    source.write_bytes(b"canonical input")
    row = dict(id="dev", split="development", frames=1440, clip_duration_sec=60,
               processed_path=str(source), processed_sha256=sha256(source))
    manifest = tmp_path / "manifest.json"
    write_json(manifest, [row])
    write_json(tmp_path / "freeze_manifest.json", {"files": [
        dict(path=str(p), sha256=sha256(p)) for p in (manifest, source)]})
    monkeypatch.setattr(app, "probe", lambda _: dict(frames=1440, fps=24, width=576, height=320, duration=60.))
    return tmp_path, manifest, row


def test_profile_preserves_full_input_and_does_not_change_legacy(canonical):
    repo, manifest, _ = canonical
    before = sha256(repo / "configs/webvid5.json")
    _, cfg = app.load_input(repo, manifest, "dev")
    assert cfg["frames"] == cfg["max_frames"] == 1440
    assert cfg["selection_stride"] == 1
    assert cfg["preserve_input"] and not cfg["official_preprocessing"]
    assert cfg["concatenation_policy"] == "endpoint_exact"
    assert cfg["decoder_policy"] == "official_release"
    assert cfg["internvl_offload_layers"] == 3 and cfg["internvl_compact_kv_cache"]
    assert sha256(repo / "configs/webvid5.json") == before


@pytest.mark.parametrize("change", ["test_split", "truncated", "tampered"])
def test_rejects_heldout_truncated_or_tampered_input(canonical, change):
    repo, manifest, row = canonical
    if change == "test_split":
        row["split"] = "test"
    elif change == "truncated":
        row["frames"] = 384
    else:
        Path(row["processed_path"]).write_bytes(b"changed")
    if change != "tampered":
        write_json(manifest, [row])
    with pytest.raises(ValueError):
        app.load_input(repo, manifest, "dev")


@pytest.mark.parametrize("times", [[0, .25, .5], [0, .5], [1, 1.25, 1.5], [0, .25, .25], [0, .25, float("nan")]])
def test_pts_requires_complete_zero_based_uniform_time(monkeypatch, times):
    monkeypatch.setattr(app.subprocess, "check_output", lambda *a, **k:
                        json.dumps({"frames": [{"best_effort_timestamp_time": v} for v in times]}))
    if times == [0, .25, .5]:
        assert app.pts_audit("unused", 3, 4)["all_pts_checked"]
    else:
        with pytest.raises(ValueError):
            app.pts_audit("unused", 3, 4)


def test_real_concatenation_preserves_all_1440_positions(tmp_path):
    app.temporal_check(tmp_path / "result.json")
    rows = app.read_json(tmp_path / "result.json")["cases"]
    assert all(r["endpoint_exact_frames"] == 1440 for r in rows)
    assert rows[-1]["official_release_frames"] == 2878


def test_injected_interruption_reuses_completed_and_rejects_tampering(tmp_path):
    app.resume_probe(tmp_path / "probe")
    result = app.read_json(tmp_path / "probe/result.json")
    assert result["completed_reused"] and result["partial_preserved"] and result["tampering_rejected"]


def test_subprocess_failure_propagates_and_keeps_resources(tmp_path):
    with pytest.raises(RuntimeError, match="exit 7"):
        app.run_command(tmp_path, [sys.executable, "-c", "print('failed'); raise SystemExit(7)"],
                        tmp_path / "log", {}, tmp_path / "resource.json", tmp_path / "progress.json")
    resource = app.read_json(tmp_path / "resource.json")
    assert resource["returncode"] == 7 and resource["error"]
    assert resource["max_rss_kib"] > 0
    assert "failed" in (tmp_path / "log").read_text()


def test_failed_parent_does_not_leave_running_model_descendant(tmp_path):
    script = ("import subprocess,sys; from pathlib import Path; "
              "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
              "Path('descendant.pid').write_text(str(p.pid)); raise SystemExit(9)")
    with pytest.raises(RuntimeError, match="exit 9"):
        app.run_command(tmp_path, [sys.executable, "-c", script], tmp_path / "log", {},
                        tmp_path / "resource.json", tmp_path / "progress.json")
    pid = int((tmp_path / "descendant.pid").read_text())
    state = Path(f"/proc/{pid}/stat")
    for _ in range(50):
        if not state.exists() or state.read_text().split()[2] == "Z":
            break
        time.sleep(.02)
    else:
        pytest.fail("orphaned model process is still running")


def test_cpu_only_result_cannot_certify_models(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "settings", lambda _: {"python": sys.executable})
    class FakeStages:
        completed = {}
        def __init__(self, *a):
            pass
        def step(self, *a):
            pass
    monkeypatch.setattr(app, "Stages", FakeStages)
    result = app.execute(tmp_path, tmp_path, {"seed": 42, "frames": 1440}, "test", True)
    assert result["status"] == "PASS_CPU_CHECKS"
    assert not result["short_model_probe_passed"]
    assert not result["full_60s_reconstructed"]
    assert not result["long_segment_gpu_capacity_verified"]


def test_smoke_missing_cross_segment_reference_fails(tmp_path, monkeypatch):
    cfg = dict(frames=25, width=576, height=320, fps=24)
    write_json(tmp_path / "run_config.json", cfg)
    folder = tmp_path / "receiver/reconstruction"
    folder.mkdir(parents=True)
    (folder / "sample.mp4").write_bytes(b"probe")
    frames = folder / "sample_frames"
    frames.mkdir()
    for i in range(25):
        (frames / f"{i}.png").touch()
    write_json(tmp_path / "receiver/reference_trace.json", [])
    monkeypatch.setattr(app, "probe", lambda _: cfg)
    monkeypatch.setattr(app, "pts_audit", lambda *a: {})
    with pytest.raises(ValueError, match="missing actual cross-segment"):
        app.audit_smoke(tmp_path)


def test_dry_run_has_no_writes(canonical, monkeypatch):
    repo, manifest, row = canonical
    monkeypatch.setattr(app, "doctor", lambda *a, **k: {"status": "PASSED"})
    monkeypatch.setattr(app, "execution_identity", lambda *a: {})
    for name in ["scripts/check_etri_60s.sh", "scripts/etri_decoder_probe.py", "configs/official_opensora.py"]:
        path = repo / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("fixture")
    before = {str(p): sha256(p) for p in repo.rglob("*") if p.is_file()}
    args = argparse.Namespace(manifest=manifest, video="dev", cpu_only=False, output=repo / "output", dry_run=True)
    app.run(repo, args)
    assert not args.output.exists()
    assert before == {str(p): sha256(p) for p in repo.rglob("*") if p.is_file()}
