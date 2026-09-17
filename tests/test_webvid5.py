"""CPU fixtures test orchestration/provenance; they are not model quality evidence."""
import argparse
import copy
import json
from pathlib import Path

import pytest

from semantic_transmission import webvid5
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.resume import STAGES, completed_runs, copy_completed


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    repo = tmp_path
    real_repo = Path(__file__).resolve().parents[1]
    cohort = webvid5.read_json(real_repo / "configs/webvid5_manifest.json")
    cfg = webvid5.read_json(real_repo / "configs/webvid5.json")
    data = repo / "dataset"
    infos = {}
    for index, video in enumerate(cohort["videos"]):
        name = f"video_{index}.mp4"
        video.update(filename=name, frames=3 + index)
        for folder in ("raw", "processed"):
            path = data / folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{folder}/{index}".encode())
            video["raw_sha256" if folder == "raw" else "processed_sha256"] = sha256(path)
        infos[name] = {k: video[k] for k in ("width", "height", "frames", "fps")}
        infos[name]["duration"] = video["frames"] / video["fps"]
    write_json(repo / "configs/webvid5.json", cfg)
    cohort["profile_sha256"] = sha256(repo / "configs/webvid5.json")
    write_json(repo / "configs/webvid5_manifest.json", cohort)
    write_json(repo / ".local/datasets.json", {"webvid55": str(data)})
    write_json(repo / ".local/model_paths.json", {})
    monkeypatch.setattr(webvid5, "probe", lambda p: infos[Path(p).name])
    monkeypatch.setattr(webvid5, "execution_identity", lambda *args: {"implementation": "fixture-v1"})
    monkeypatch.setattr(webvid5, "repository", lambda: repo)
    return repo


class FakeResearch:
    """Exercise real resume validation using small, explicitly synthetic artifacts."""
    def __init__(self, repo, fail_at=None):
        self.repo, self.fail_at = repo, fail_at
        self.commands, self.generated, self.reused = [], [], []

    def __call__(self, argv):
        self.commands.append(argv)
        parser = argparse.ArgumentParser()
        for key in ("input-dir", "output", "profile"):
            parser.add_argument("--" + key, type=Path, required=True)
        parser.add_argument("--video", action="append")
        parser.add_argument("--reuse-completed-from", action="append", type=Path, default=[])
        parser.add_argument("--dry-run", action="store_true")
        args = parser.parse_args(argv)
        cohort, profile, cfg, sources = webvid5.load_inputs(self.repo)
        assert args.profile == profile
        assert args.video == [s["id"] for s in sources]
        assert len(list(args.input_dir.glob("*.mp4"))) == 5
        assert cfg["selection_stride"] == 1
        assert cfg["evaluation_profile"] == "lgvsc_official_metrics_v1"
        reuse = completed_runs(args.reuse_completed_from, cfg, sources)
        if args.dry_run:
            return
        root = args.output
        root.mkdir(parents=True)
        write_json(root / "profile.json", cfg)
        batch = {"status": "RUNNING", "inputs": sources, "code": {"commit": "synthetic-fixture"}, "runs": []}
        for index, source in enumerate(sources):
            run = root / source["id"]
            if source["id"] in reuse:
                record = copy_completed(reuse[source["id"]], run)
                self.reused.append(source["id"])
            elif index == self.fail_at:
                batch["status"] = "FAILED"
                batch["runs"].append({"id": source["id"], "status": "FAILED", "error": "fixture stage failed"})
                write_json(root / "batch_manifest.json", batch)
                raise RuntimeError("fixture stage failed")
            else:
                self.generated.append(source["id"])
                record = {"id": source["id"], "status": "PASSED", "stages": [
                    {"stage": s, "status": "PASSED", "returncode": 0, "seconds": 1.0} for s in STAGES]}
                resolved = dict(cfg, frames=source["frames"], input=source["path"])
                write_json(run / "run_config.json", resolved)
                reference = run / "data/normalized.mp4"
                reference.parent.mkdir(parents=True)
                reference.write_bytes(Path(source["path"]).read_bytes())
                output = run / "receiver/reconstruction/sample_0000.mp4"
                output.parent.mkdir(parents=True)
                output.write_bytes(b"synthetic reconstruction")
                frames = output.parent / "sample_0000_frames"
                frames.mkdir()
                for frame in range(source["frames"]):
                    (frames / f"{frame:05d}.png").write_bytes(b"synthetic png")
                indices = [0, source["frames"] - 1]
                write_json(run / "keyframes.json", {"indices": indices})
                write_json(run / "receiver/decoder_inputs.json", {"indices": indices,
                    "decoder": {"policy": "official_release", "concatenation_policy": "official_release"}})
                quality = {"status": "PASSED", "source_sha256": source["sha256"],
                    "reference_sha256": sha256(reference), "video_sha256": sha256(output),
                    "output_source_indices": list(range(source["frames"])), "video": source,
                    "evaluation_profile": cfg["evaluation_profile"]}
                for boundary in webvid5.BOUNDARIES:
                    quality[boundary] = {metric: (index + 1) / 10 for metric in webvid5.METRICS}
                write_json(run / "quality.json", quality)
                files = {}
                for name in ("metadata.bin", "visual.c64"):
                    path = run / "transmitter" / name
                    path.parent.mkdir(exist_ok=True)
                    path.write_bytes(b"fixture payload")
                    files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
                write_json(run / "sender_accounting.json", {"transmitter_files": files, "serialized_model_input_bytes": 30})
                write_json(run / "receiver_accounting.json", {"status": "PASSED"})
                write_json(run / "channel_accounting.json", {"status": "PASSED", "transmission_breakdown": {"fixture": True},
                    "cbr_complex_uses_per_source_scalar": 0.1, "total_complex_channel_uses": 100})
            write_json(run / "run_manifest.json", record)
            batch["runs"].append(record)
        batch["status"] = "PASSED"
        write_json(root / "batch_manifest.json", batch)


def latest(repo):
    return Path(webvid5.read_json(repo / ".local/webvid5_history.json")["runs"][-1]["output"])


def test_dry_run_uses_five_full_sources_without_state_or_model_execution(pilot, monkeypatch):
    runner = FakeResearch(pilot)
    monkeypatch.setattr(webvid5, "research", runner)
    before = {str(p.relative_to(pilot)): sha256(p) for p in pilot.rglob("*") if p.is_file()}
    webvid5.main(["--dry-run"])
    after = {str(p.relative_to(pilot)): sha256(p) for p in pilot.rglob("*") if p.is_file()}
    assert after == before
    assert "--dry-run" in runner.commands[0] and runner.generated == []


@pytest.mark.parametrize("change,match", [("missing", ""), ("hash", "hash mismatch"),
    ("duplicate", "five unique"), ("profile", "profile changed"), ("length", "length changed")])
def test_invalid_cohort_fails_before_execution(pilot, monkeypatch, change, match):
    source = pilot / "dataset/processed/video_0.mp4"
    if change == "missing":
        source.unlink()
    elif change == "hash":
        source.write_bytes(b"truncated video")
    elif change == "profile":
        (pilot / "configs/webvid5.json").write_text("{}")
    else:
        path = pilot / "configs/webvid5_manifest.json"
        manifest = webvid5.read_json(path)
        if change == "duplicate":
            manifest["videos"][1] = copy.deepcopy(manifest["videos"][0])
        else:
            manifest["videos"][0]["frames"] += 1
        write_json(path, manifest)
    monkeypatch.setattr(webvid5, "research", lambda _: pytest.fail("must not start inference"))
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        webvid5.run(pilot)
    assert not (pilot / ".local/webvid5_history.json").exists()


def test_interrupt_resume_reuses_only_verified_completed_video_and_reports_all_five(pilot, monkeypatch):
    runner = FakeResearch(pilot, fail_at=1)
    monkeypatch.setattr(webvid5, "research", runner)
    with pytest.raises(RuntimeError, match="fixture stage failed"):
        webvid5.run(pilot)
    first = latest(pilot)
    partial = webvid5.read_json(first / "summary.json")
    assert partial["completed_videos"] == 1 and partial["status"] == "INCOMPLETE"
    assert partial["mean_over_five_videos"] is None
    snapshot = {str(p.relative_to(first)): sha256(p) for p in first.rglob("*") if p.is_file()}
    runner.fail_at = None
    webvid5.run(pilot)
    result = webvid5.read_json(latest(pilot) / "summary.json")
    assert runner.reused == ["video_0"] and runner.generated == [f"video_{i}" for i in range(5)]
    assert result["status"] == "PASSED" and result["completed_videos"] == 5
    assert result["manual_review_status"] == "PENDING" and result["paper_reproduction"] is False
    assert result["mean_over_five_videos"]["delivered_mp4_psnr_db"] == pytest.approx(0.3)
    assert result["transmission_totals"]["complex_channel_uses"] == 500
    assert (latest(pilot) / "review_media/low_motion_reconstruction.mp4").is_file()
    assert snapshot == {str(p.relative_to(first)): sha256(p) for p in first.rglob("*") if p.is_file()}


def test_changed_code_starts_new_cohort_run_without_mixing_old_results(pilot, monkeypatch):
    runner = FakeResearch(pilot)
    monkeypatch.setattr(webvid5, "research", runner)
    webvid5.run(pilot)
    monkeypatch.setattr(webvid5, "execution_identity", lambda *args: {"implementation": "fixture-v2"})
    webvid5.run(pilot)
    assert len(runner.generated) == 10 and not runner.reused


def test_missing_metric_cannot_appear_as_five_video_success(pilot, monkeypatch):
    runner = FakeResearch(pilot)
    monkeypatch.setattr(webvid5, "research", runner)
    webvid5.run(pilot)
    root = latest(pilot)
    quality_path = root / "video_2/quality.json"
    quality = webvid5.read_json(quality_path)
    del quality["delivered_mp4"]["dists"]
    write_json(quality_path, quality)
    cohort, _, cfg, sources = webvid5.load_inputs(pilot)
    review = root / "manual_review.csv"
    review.write_text("user review must remain intact\n")
    result = webvid5.make_report(root, cohort, cfg, sources)
    assert result["completed_videos"] == 4 and result["status"] == "INCOMPLETE"
    assert result["mean_over_five_videos"] is None
    assert review.read_text() == "user review must remain intact\n"


def test_duplicate_command_is_rejected_before_loading_models(pilot, monkeypatch):
    import fcntl
    monkeypatch.setattr(webvid5, "run", lambda *a, **kw: pytest.fail("must not start second run"))
    with (pilot / ".local/webvid5_run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit) as error:
            webvid5.main([])
    assert error.value.code == 1


@pytest.mark.parametrize("log,code,reason", [
    ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 112.00 MiB.\n", 1, "CUDA out of memory"),
    ("unrelated failure\n", 37, "exit code 37"),
])
def test_research_failure_reports_cause_and_preserves_failed_stage(pilot, monkeypatch, log, code, reason):
    import sys
    from types import SimpleNamespace
    from semantic_transmission import research as driver
    monkeypatch.setattr(driver, "repository", lambda: pilot)
    monkeypatch.setattr(driver, "settings", lambda _: {name: sys.executable
                        for name in ("python", "channel_python", "internvl_python")})
    monkeypatch.setattr(driver, "doctor", lambda _: {"status": "PASSED"})
    monkeypatch.setattr(driver, "probe", webvid5.probe)
    monkeypatch.setattr(driver, "runtime_state", lambda *a: {"scope": "synthetic test"})
    monkeypatch.setattr(driver, "git_state", lambda _: {"commit": "synthetic test"})
    def worker(command, **kwargs):
        failed = command[3] == "select"
        if failed:
            kwargs["stdout"].write(log)
        return SimpleNamespace(returncode=code if failed else 0)
    monkeypatch.setattr(driver.subprocess, "run", worker)
    output = pilot / "failed_batch"
    with pytest.raises(RuntimeError, match=reason) as error:
        driver.main(["--input-dir", str(pilot / "dataset/processed"), "--output", str(output),
                     "--profile", str(pilot / "configs/webvid5.json")])
    assert str(output / "video_0/logs/select.log") in str(error.value)
    batch = webvid5.read_json(output / "batch_manifest.json")
    assert batch["status"] == "FAILED" and batch["failed_videos"] == 1
    assert batch["completed_videos"] == 0 and len(batch["runs"]) == 1
    stages = batch["runs"][0]["stages"]
    assert stages[0]["status"] == "PASSED" and stages[1]["returncode"] == code
