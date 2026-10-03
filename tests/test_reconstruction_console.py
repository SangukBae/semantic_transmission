import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import types

import pytest

from semantic_transmission import reconstruction_console as ui
from semantic_transmission.artifacts import write_json


def test_actual_enumerate_tqdm_steps_and_segments_are_monotonic():
    progress = ui.Progress(78, 30)
    progress.decoder_line("loop_i 0 batch_prompts_loop ['ignored before reconstruct']")
    assert progress.percent == 0
    progress.stage_line("START reconstruct | log")
    progress.decoder_line("Loading checkpoint shards: 100% 2/2")
    assert progress.percent == 0
    progress.decoder_line("loop_i 0 batch_prompts_loop ['text'] refs 1 ms []")
    progress.decoder_line("15it [00:04, 3.40it/s]")
    assert progress.percent == pytest.approx(100 / 156)
    progress.completed_segments({"done": 1, "total": 78})
    previous = progress.percent
    progress.decoder_line("0it [00:00, ?it/s]")
    assert progress.percent == previous
    progress.decoder_line("loop_i 1 batch_prompts_loop ['next'] refs 1 ms []")
    progress.decoder_line("6it [00:01, 4.00it/s]")
    assert progress.percent == pytest.approx(100 * 1.2 / 78)
    progress.stage_line("DONE reconstruct: 1분")
    assert progress.percent == 99.99
    progress.completed_segments({"done": 5, "total": 1440})
    assert progress.percent == 99.99


def test_verified_reused_reconstruction_waits_for_postprocessing():
    progress = ui.Progress(78, 30)
    progress.stage_line("REUSE reconstruct (완료 파일 검증됨)")
    assert progress.percent == 99.99 and not progress.active


def test_growing_log_handles_partial_utf8_cr_lines_and_replacement(tmp_path):
    path = tmp_path / "log"
    reader = ui.GrowingLog(path)
    assert reader.read() == []
    encoded = "복원\rloop_i 0 batch_prompts_loop\n3it [time]".encode()
    path.write_bytes(encoded[:2])
    assert reader.read() == []
    with path.open("ab") as stream:
        stream.write(encoded[2:])
    assert reader.read() == ["복원", "loop_i 0 batch_prompts_loop"]
    assert reader.read(final=True) == ["3it [time]"]
    path.unlink()
    path.write_text("new\n")
    assert reader.read() == ["new"]


@pytest.fixture
def fake_run(tmp_path):
    output = tmp_path / "run-output"
    write_json(output / "run/keyframes.json", {"indices": [0, 12, 32]})
    write_json(output / "run/receiver/decoder_inputs.json", {"decoder": {"steps": 30}})
    return output


@pytest.mark.parametrize("returncode", [0, 7])
def test_real_subprocess_only_updates_one_line_and_keeps_logs(fake_run, tmp_path, returncode):
    script = tmp_path / "worker.py"
    script.write_text('''import os,sys,time
from pathlib import Path
root=Path(sys.argv[1])
print("START reconstruct | test",flush=True)
print("model warning",file=sys.stderr,flush=True)
path=root/"run/receiver/reconstruction/00000.log"
path.parent.mkdir()
with path.open("w") as f:
    f.write("loop_i 0 batch_prompts_loop ['first']\\n"); f.flush()
    f.write("15it [00:01, 1.00it/s]\\r"); f.flush()
    time.sleep(.12)
    f.write("loop_i 1 batch_prompts_loop ['last']\\n30it [00:02, 1.00it/s]\\r"); f.flush()
print("DONE reconstruct: 0.1분",flush=True)
print("START evaluate | test",flush=True)
time.sleep(.12)
print("COMPLETE TAIL17: test",flush=True)
sys.exit(int(sys.argv[2]))
''')
    stream = io.StringIO()
    log = tmp_path / "console.log"
    code = ui.run_command([sys.executable, str(script), str(fake_run), str(returncode)], fake_run,
                          log, tmp_path / "progress.json", stream=stream, interval=.02,
                          gpu_monitor=types.SimpleNamespace(read=lambda: 87))
    text = stream.getvalue()
    assert code == returncode
    assert text.count("\n") == 1
    assert " 25.00%" in text and " 99.99%" in text
    assert "model warning" not in text and "START" not in text and "COMPLETE" not in text
    assert "model warning" in log.read_text() and "START evaluate" in log.read_text()
    if returncode:
        assert "100.00%" not in text and "복원 실패" in text and str(log) in text
    else:
        assert text.endswith("복원 진행률: 100.00% | GPU 사용률:  87%\n")


def test_check_forwards_options_without_creating_ui_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "REPO", tmp_path)
    monkeypatch.setattr(ui, "OUTPUT", tmp_path / "default")
    calls = []
    monkeypatch.setattr(ui.subprocess, "call", lambda cmd, **kwargs: calls.append(cmd) or 0)
    assert ui.main(["--check"]) == 0
    assert calls[0][-2:] == ["--output", str(tmp_path / "default")]
    assert "--t5-cache" in calls[0] and "--check" in calls[0]
    assert not (tmp_path / ".local").exists()
    explicit = str(tmp_path / "a space")
    ui.main(["--output=" + explicit, "--check"])
    assert calls[-1].count("--output=" + explicit) == 1 and "--output" not in calls[-1]


def test_sigterm_stops_child_and_preserves_interruption_exit_status(fake_run, tmp_path):
    if sys.platform != "linux":
        pytest.skip("process-group behavior is Linux specific")
    wrapper = tmp_path / "wrapper.py"
    pidfile = tmp_path / "pid"
    wrapper.write_text('''import signal,sys
from pathlib import Path
from semantic_transmission.reconstruction_console import run_command
def stop(*_): raise KeyboardInterrupt
signal.signal(signal.SIGTERM,stop)
worker="import os,time; from pathlib import Path; Path(%r).write_text(str(os.getpid())); time.sleep(60)" % sys.argv[2]
root=Path(sys.argv[1])
raise SystemExit(run_command([sys.executable,"-c",worker],root,root/"console.log",root/"progress.json",interval=.02))
''')
    import os
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    child = subprocess.Popen([sys.executable, str(wrapper), str(fake_run), str(pidfile)], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert pidfile.exists()
        pid = int(pidfile.read_text())
        child.send_signal(signal.SIGTERM)
        out, err = child.communicate(timeout=5)
        assert child.returncode == 130 and "복원 중단" in out and "100.00%" not in out
        assert not err
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_gpu_queries_selected_device_once_per_second_and_recovers_from_unavailable(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test,1")
    now, calls = [10.0], []
    monkeypatch.setattr(ui.time, "monotonic", lambda: now[0])
    def query(command, **kwargs):
        calls.append(command)
        return types.SimpleNamespace(stdout="73\n")
    monkeypatch.setattr(ui.subprocess, "run", query)
    monitor = ui.GpuUsage()
    assert monitor.read() == 73 and monitor.read() == 73
    assert len(calls) == 1 and calls[0][1:3] == ["-i", "GPU-test"]
    now[0] += 1.1
    monkeypatch.setattr(ui.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    assert monitor.read() is None
    monkeypatch.setattr(ui.subprocess, "run", query)
    now[0] += 1.1
    assert monitor.read() == 73


def test_gpu_changes_refresh_line_even_when_progress_is_stationary():
    stream = io.StringIO()
    line = ui.SingleLine(stream)
    line.update(12.34, 5)
    line.update(12.34, 95)
    line.update(12.34, 95)
    line.update(12.34, None)
    assert stream.getvalue().count("\r") == 3
    assert "GPU 사용률:  95%" in stream.getvalue() and "GPU 사용률:  --%" in stream.getvalue()
    assert "\n" not in stream.getvalue()


def test_fp32_check_routes_to_new_runner_without_forcing_old_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "REPO", tmp_path)
    calls = []
    monkeypatch.setattr(ui.subprocess, "call", lambda cmd, **kw: calls.append(cmd) or 0)
    assert ui.main(["--precision-run", "--check"]) == 0
    command = calls[0]
    assert command[:3] == [sys.executable, "-m", "semantic_transmission.precision_review"]
    assert "--t5-cache" not in command and "--precision-run" not in command
    assert command[-1].endswith("_t5fp32_fixednoise")
    ui.main(["--precision-run", "--t5-precision=bf16", "--noise-reference", "reference.json", "--check"])
    assert calls[-1][-1].endswith("_t5bf16_fixednoise")
    assert "reference.json" in calls[-1]
    assert not (tmp_path / ".local").exists()
