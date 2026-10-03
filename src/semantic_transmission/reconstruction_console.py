"""Single-line terminal view over the unchanged, resumable reconstruction runner.

Only reads existing decoder counters/logs; never changes sampling or tensors.
Progress counts completed diffusion iterations, not estimated remaining time.
"""
import codecs
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

from .artifacts import sha256, write_json
from .tail_reference import REPO, T5_OUTPUT

# The old launcher's hash belongs to the prepared run. Keep that receipt intact
# and use a fresh sibling; the persistent T5 cache remains shared and reusable.
OUTPUT = T5_OUTPUT.with_name(T5_OUTPUT.name + "_console")


class Progress:
    def __init__(self, segments, steps):
        if segments <= 0 or steps <= 0:
            raise ValueError("positive reconstruction segment/step counts required")
        self.segments, self.steps = segments, steps
        self.loop = None
        self.fraction = 0.0
        self.active = False

    def stage_line(self, line):
        if re.match(r"START reconstruct(?:\s|$)", line):
            self.active = True
        elif re.match(r"(?:DONE|REUSE) reconstruct(?:\s|:|$)", line):
            self.fraction = 1.0
            self.active = False

    def decoder_line(self, line):
        if not self.active:
            return
        # RFLOW wraps enumerate(timesteps) without a total: tqdm prints '7it'.
        events = re.finditer(r"loop_i (\d+) batch_prompts_loop|(?<!\w)(\d+)it\s*\[", line)
        for event in events:
            if event[1] is not None:
                loop = int(event[1])
                if 0 <= loop < self.segments:
                    self.loop = loop
            elif self.loop is not None:
                step = int(event[2])
                if 0 <= step <= self.steps:
                    self.fraction = max(self.fraction, (self.loop + step / self.steps) / self.segments)

    def completed_segments(self, record):
        if not self.active or record.get("total") != self.segments:
            return
        done = record.get("done")
        if isinstance(done, int) and 0 <= done <= self.segments:
            self.fraction = max(self.fraction, done / self.segments)

    @property
    def percent(self):
        # Encoding, audit, quality evaluation and comparison still follow the
        # final diffusion iteration. Only a successful command can show 100%.
        return min(99.99, 100 * self.fraction)


class GrowingLog:
    """Read appended bytes only, including tqdm carriage-return lines."""
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.inode = None
        self.pending = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def read(self, final=False):
        try:
            stat = self.path.stat()
            if self.inode != stat.st_ino or stat.st_size < self.offset:
                self.offset, self.pending, self.inode = 0, "", stat.st_ino
                self.decoder.reset()
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                data = stream.read()
                self.offset = stream.tell()
        except FileNotFoundError:
            return []
        pieces = re.split(r"[\r\n]", self.pending + self.decoder.decode(data))
        self.pending = pieces.pop()
        if final and self.pending:
            pieces.append(self.pending)
            self.pending = ""
        return pieces


class SingleLine:
    def __init__(self, stream):
        self.stream = stream
        self.last = None
        self.gpu = None

    def update(self, percent, gpu=None):
        self.gpu = gpu
        usage = f"{gpu:3d}%" if gpu is not None else " --%"
        value = f"복원 진행률: {percent:6.2f}% | GPU 사용률: {usage}"
        if value != self.last:
            self.stream.write("\r" + value)
            self.stream.flush()
            self.last = value

    def finish(self, code, log):
        if code == 0:
            self.update(100, self.gpu)
        else:
            status = "복원 중단" if code == 130 else "복원 실패"
            self.stream.write(f"\r{status} · 상세 로그: {log}")
        self.stream.write("\n")
        self.stream.flush()


class GpuUsage:
    """Query the selected physical GPU at most once per second; no CUDA init."""
    def __init__(self, interval=1.0):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        self.device = visible if visible and visible != "-1" else None
        self.interval, self.next_query, self.value = interval, 0.0, None

    def read(self):
        if time.monotonic() < self.next_query:
            return self.value
        self.next_query = time.monotonic() + self.interval
        self.value = None
        if self.device is None:
            return None
        try:
            result = subprocess.run(["nvidia-smi", "-i", self.device, "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=0.8, check=True)
            value = int(result.stdout.strip())
            if 0 <= value <= 100:
                self.value = value
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        return self.value


def stop(process):
    """Let the runner terminate its worker process groups and save receipts."""
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=35)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_command(command, output, log, progress_path, *, stream=sys.stdout, interval=0.2, gpu_monitor=None):
    from .tail_reference import SOURCE
    config_root = output / "run" if (output / "run/keyframes.json").exists() else SOURCE / "run"
    keys = json.loads((config_root / "keyframes.json").read_text())["indices"]
    inputs = json.loads((config_root / "receiver/decoder_inputs.json").read_text())
    progress = Progress(len(keys) - 1, inputs["decoder"]["steps"])
    console = SingleLine(stream)
    gpu_monitor = gpu_monitor or GpuUsage()
    console.update(0)
    root_log = GrowingLog(log)
    decoder_log = GrowingLog(output / "run/receiver/reconstruction/00000.log")
    env = dict(os.environ, QUALITY_PROGRESS_FILE=str(progress_path))
    process = None
    code = 1
    try:
        with log.open("x") as capture:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=capture,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                for line in root_log.read():
                    progress.stage_line(line)
                if progress.active:
                    for line in decoder_log.read():
                        progress.decoder_line(line)
                    try:
                        progress.completed_segments(json.loads(progress_path.read_text()))
                    except (FileNotFoundError, json.JSONDecodeError):
                        pass
                console.update(progress.percent, gpu_monitor.read())
                try:
                    code = process.wait(timeout=interval)
                    break
                except subprocess.TimeoutExpired:
                    pass
    except KeyboardInterrupt:
        if process is not None:
            stop(process)
        code = 130
    except BaseException:
        if process is not None:
            stop(process)
        import traceback
        with log.open("a") as capture:
            traceback.print_exc(file=capture)
        code = 1
    finally:
        console.finish(code, log)
    return code


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    output = OUTPUT
    precision_run = "--precision-run" in args
    if precision_run:
        from .precision_reconstruction import default_output
        args.remove("--precision-run")
        precision = "fp32"
        for index, arg in enumerate(args):
            if arg.startswith("--t5-precision="):
                precision = arg.split("=", 1)[1]
            elif arg == "--t5-precision" and index + 1 < len(args):
                precision = args[index + 1]
        output = default_output(precision)
    # Forward the runner's options, preserving both --output forms and spaces.
    for index, arg in enumerate(args):
        if arg.startswith("--output="):
            output = Path(arg.split("=", 1)[1]).resolve()
        elif arg == "--output" and index + 1 < len(args):
            output = Path(args[index + 1]).resolve()
    if not any(a == "--output" or a.startswith("--output=") for a in args):
        args.extend(["--output", str(output)])
    command = ([sys.executable, "-m", "semantic_transmission.precision_review", *args] if precision_run else
               ["bash", str(REPO / "scripts/reconstruct_tail17.sh"), "--t5-cache", *args])
    # Readiness/help keep their structured output and never create UI files.
    if any(a in args for a in ("--check", "--help", "-h", "--prepare-text-only")):
        return subprocess.call(command, cwd=REPO)
    folder = REPO / ".local/reconstruction_console"
    folder.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="run-", dir=folder))
    log = session / "console.log"
    write_json(session / "invocation.json", dict(command=command, output=str(output),
        console_sha256=sha256(Path(__file__)), started_unix=time.time()))
    def interrupted(*_):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        return run_command(command, output, log, session / "progress.json")
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
