import io
import json
from pathlib import Path
import re
import sys
import types

import pytest

from semantic_transmission import fc_lgvsc_batch as batch


def test_check_validates_all_current_default_policies_without_writes(tmp_path, monkeypatch, capsys):
    checked = []
    def preflight(source, output):
        checked.append((source, output))
        return dict(frames=24, fps=24, plan=[{}], baseline_noise_contract={"steps":30},
            t5_compute_dtype="fp32", condition_collision_policy=batch.default.base.fix.POLICY,
            short_schedule_policy=batch.default.paired.fix.POLICY), "signature"
    monkeypatch.setattr(batch.default, "preflight", preflight)
    monkeypatch.setattr(batch, "console_run", lambda *_: pytest.fail("check started workers"))
    dest = tmp_path / "new output"
    assert batch.main(["--check", "--output-root", str(dest)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [row["video"] for row in report["videos"]] == list(batch.VIDEOS)
    assert report["model_inference_started"] is False and not dest.exists()
    for video, (source, output) in zip(batch.VIDEOS, checked):
        assert source == batch.default.base.source_for(video).resolve()
        assert output == dest / video
    for job in report["videos"]:
        assert job["condition_collision_policy"] == batch.default.base.fix.POLICY
        assert job["short_schedule_policy"] == batch.default.paired.fix.POLICY
        assert job["t5_compute_dtype"] == "fp32"


def test_last_input_failure_prevents_starting_first_video(tmp_path, monkeypatch):
    checked = []
    def preflight(source, output):
        checked.append(output)
        if len(checked) == 4:
            raise ValueError("changed last-video inputs")
        return dict(frames=24, fps=24, plan=[{}], baseline_noise_contract={"steps":30},
            t5_compute_dtype="fp32", condition_collision_policy="guard", short_schedule_policy="repair"), "signature"
    monkeypatch.setattr(batch.default, "preflight", preflight)
    monkeypatch.setattr(batch, "console_run", lambda *_: pytest.fail("partial batch launched"))
    with pytest.raises(ValueError, match="changed last-video inputs"):
        batch.main(["--output-root", str(tmp_path / "untouched")])
    assert len(checked) == 4 and not (tmp_path / "untouched").exists()


def test_worker_routes_to_default_runner_with_resume_and_space_safe_output():
    job = dict(video="person_walk", output="/tmp/path with spaces/person_walk")
    command = batch.worker_command(job)
    assert command == [sys.executable, "-m", batch.default.MODULE, "--execute",
                       "--video", "person_walk", "--output", job["output"]]


@pytest.fixture
def fake_batch(tmp_path, monkeypatch):
    script = tmp_path / "worker.py"
    script.write_text('''import sys,time
from pathlib import Path
output=Path(sys.argv[1])
code=int(sys.argv[2])
review=output/'review.html'
if review.exists():
    print('REUSE reconstruct (fixture)',flush=True)
else:
    print('START reconstruct | fixture',flush=True)
    print('model warning',file=sys.stderr,flush=True)
    path=output/'run/receiver/reconstruction/00000.log'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("loop_i 0 batch_prompts_loop ['fixture']\\n1it [time]\\r")
    time.sleep(.08)
    print('DONE reconstruct: fixture',flush=True)
print('START evaluate | fixture',flush=True)
time.sleep(.08)
if not code: review.write_text('fixture')
sys.exit(code)
''')
    jobs = [dict(video=video, output=str(tmp_path/video), segments=2+i, steps=2)
            for i,video in enumerate(batch.VIDEOS)]
    calls, fail = [], [None]
    def command(job):
        calls.append(job["video"])
        return [sys.executable, str(script), job["output"], "7" if job["video"] == fail[0] else "0"]
    monkeypatch.setattr(batch, "worker_command", command)
    return jobs, calls, fail


def run_fixture(tmp_path, jobs):
    stream = io.StringIO()
    code = batch.console_run(jobs, stream=stream, interval=.02,
        gpu_monitor=types.SimpleNamespace(read=lambda: 87), logroot=tmp_path/"logs")
    return code, stream.getvalue()


def test_real_processes_run_in_order_keep_one_line_and_revalidate_completed_videos(tmp_path, fake_batch):
    jobs, calls, _ = fake_batch
    code, text = run_fixture(tmp_path, jobs)
    assert code == 0 and calls == list(batch.VIDEOS)
    assert "GPU 사용률:  87%" in text and "4/4 person_walk" in text
    assert "START" not in text and "model warning" not in text
    assert text.count("\n") == 5  # One completion line plus four result paths.
    percents = [float(x) for x in re.findall(r"복원 진행률:\s*([\d.]+)%", text)]
    assert percents == sorted(percents) and percents[-1] == 100 and percents.count(100) == 1
    sessions = list((tmp_path/"logs").iterdir())
    assert len(sessions) == 1
    assert "model warning" in (sessions[0]/"tv_low_08.log").read_text()
    calls.clear()
    code, text = run_fixture(tmp_path, jobs)
    assert code == 0 and calls == list(batch.VIDEOS)  # Worker verifies reuse; UI cannot skip it.
    assert "완료: 4개 영상" in text
    assert any("REUSE reconstruct" in p.read_text() for p in (tmp_path/"logs").glob("*/tv_low_08.log"))


def test_failed_video_stops_batch_without_false_completion(tmp_path, fake_batch):
    jobs, calls, fail = fake_batch
    fail[0] = "single_subject"
    code, text = run_fixture(tmp_path, jobs)
    assert code == 7 and calls == ["tv_low_08", "single_subject"]
    assert "100.00%" not in text and "복원 실패 (single_subject)" in text
    assert "상세 로그:" in text and "single_subject.log" in text
    assert (Path(jobs[0]["output"])/"review.html").exists()
    assert not Path(jobs[2]["output"]).exists()


def test_interruption_stops_child_and_does_not_start_next_video(tmp_path, fake_batch, monkeypatch):
    jobs, calls, _ = fake_batch
    stopped = []
    monkeypatch.setattr(batch, "stop", lambda process: stopped.append(process))
    class Process:
        def wait(self, timeout):
            raise KeyboardInterrupt
    process = Process()
    monkeypatch.setattr(batch.subprocess, "Popen", lambda *a,**kw: process)
    code, text = run_fixture(tmp_path, jobs)
    assert code == 130 and stopped == [process] and calls == ["tv_low_08"]
    assert "복원 중단" in text and "100.00%" not in text
