"""Terminal-only progress display around an unchanged, resumable worker."""
import argparse
import codecs
import json
from collections import deque
from datetime import datetime
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import time
import unicodedata


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*(?:\x07|\x1b\\)")
FRACTION = re.compile(r"(?<!\d)([\d,]+)\s*/\s*([\d,]+)(?!\d)")
BATCH = re.compile(r"^BATCH (\d+)/(\d+) \([\d.]+%\)\s*(.*)")


def clean(text):
    text = ANSI.sub("", text)
    return "".join(c for c in text if c == "\t" or (ord(c) >= 32 and ord(c) != 127)).strip()


def fit(text, columns):
    """Clip by terminal cells, including Korean, before the autowrap column."""
    output, used = [], 0
    for c in clean(text).replace("\t", " "):
        width = 0 if unicodedata.combining(c) else (2 if unicodedata.east_asian_width(c) in "WF" else 1)
        if used + width > max(0, columns - 1):
            break
        output.append(c); used += width
    return "".join(output)


def duration(seconds):
    minutes, seconds = divmod(int(max(0, seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class Progress:
    def __init__(self):
        self.overall = "전체 건수 확인 중"
        self.current = "입력·환경 검사"
        self.stage = "시작 준비 중 (진행률 미확정)"
        self.state = "실행 중"
        self.tail = deque(maxlen=30)
        self.summary = []
        self.carry = ""
        self.reported = (0, 0)

    def feed(self, text, final=False):
        # Pipes can split UTF-8 and CR/LF independently. The caller decodes UTF-8
        # incrementally; this splitter also handles tqdm's carriage returns.
        pieces = re.split(r"[\r\n]", self.carry + text)
        self.carry = pieces.pop()
        for piece in pieces:
            self.line(piece)
        if final and self.carry:
            self.line(self.carry); self.carry = ""
        elif len(self.carry) > 16384:
            self.line(self.carry[:16384]); self.carry = ""

    def line(self, raw):
        line = clean(raw)
        if not line:
            return
        self.tail.append(line)
        if line.startswith('COMPARE_PROGRESS '):
            try:
                event = json.loads(line[len('COMPARE_PROGRESS '):])
                if isinstance(event.get('stage'), str) and 'state' not in event:
                    self.state = '실행 중'
                for name in ('overall', 'current', 'stage', 'state'):
                    if isinstance(event.get(name), str):
                        setattr(self, name, clean(event[name]))
            except (ValueError, TypeError):
                self.state = '진행 상태 해석 실패 — 상세 로그 확인'
            return
        if line.startswith(('CHECK:', 'REPORT:', 'STATUS:')):
            self.summary.append(line)
            return
        if line.startswith("BSU WebVid reconstruction"):
            self.summary = [line]
        elif self.summary and (line.startswith(("Status:", "Sources:", "SNR\t")) or "\t" in line):
            self.summary.append(line)
        completed = re.search(r"^Status: .*; complete video/SNR pairs (\d+)/(\d+)", line)
        if completed:
            self.reported = tuple(map(int, completed.group(1, 2)))
            self.count(*self.reported)
            return
        match = BATCH.match(line)
        if match:
            done, total = map(int, match.group(1, 2))
            # The initial report verifies cached results before the job loop.
            # BATCH counts visited jobs; it must not erase verified completion.
            if self.reported[1] == total:
                done = max(done, self.reported[0])
            self.count(done, total)
            self.current = re.split(r"\s+elapsed=", match[3])[0]
            self.stage = "완료 결과 확인 / 다음 단계 준비"
            self.state = "실행 중"
            return
        if line.startswith("RESUME:"):
            self.state = "완료 산출물 검증·재사용 (재추론 없음)"
            return
        if line.startswith("START "):
            self.stage = line[6:].split(";", 1)[0].split(" |", 1)[0] + " (진행률 미확정)"
            self.state = "실행 중"
            return
        if line.startswith("DONE "):
            self.state = "현재 단계 완료"
            return
        matches = list(FRACTION.finditer(line))
        if matches and ("%" in line or line.startswith(("QUALITY ", "METRICS ", "WINDOW ", "REFINEMENT ", "TEMPORAL "))):
            # WINDOW a/b ... n/total: show the inner counter as stage progress.
            counter = matches[-1]
            done, total = [int(s.replace(",", "")) for s in counter.group(1, 2)]
            if total > 0 and 0 <= done <= total:
                label = re.sub(r"[\d.]+%", "", line[:counter.start()]).rstrip(" :(|")
                eta = re.search(r"ETA\s*[=:]?\s*([^|;]+)", line)
                suffix = f" | ETA {eta[1].strip()}" if eta else ""
                self.stage = f"{label} {done}/{total} ({100*done/total:.1f}%)" + suffix
                self.state = "실행 중"
                return
        # Unknown-size loading is a status, never an invented percentage.
        if any(word in line.lower() for word in ("loading", "검증", "준비", "export", "warning", "error", "traceback", "failed")):
            self.state = line

    def count(self, done, total):
        percent = f"{100*done/total:.1f}%" if total else "미확정"
        self.overall = f"{done}/{total} ({percent}) 영상·SNR 조건 완료"

    def rows(self, elapsed, log):
        return [f"전체 진행 : {self.overall}", f"현재 작업 : {self.current}",
                f"단계 진행 : {self.stage}", f"경과 시간 : {duration(elapsed)} | 전체 ETA 미확정",
                f"실행 상태 : {self.state}", f"상세 로그 : {log}"]


class Display:
    def __init__(self, stream):
        self.stream, self.height = stream, 0

    def draw(self, rows):
        try:
            columns = os.get_terminal_size(self.stream.fileno()).columns
        except (AttributeError, OSError, ValueError):
            columns = 100
        if self.height:
            self.stream.write(f"\x1b[{self.height}F")
        for row in rows:
            self.stream.write("\r\x1b[2K" + fit(row, columns) + "\n")
        self.stream.flush()
        self.height = len(rows)


def run(command, log_dir, mode="auto", stream=None, stop_grace_seconds=30.):
    stream = sys.stdout if stream is None else stream
    live = mode == "live" or (mode == "auto" and stream.isatty() and os.environ.get("TERM") != "dumb")
    log_dir = Path(log_dir).resolve(); log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log = log_dir / f"bsu_webvid_{stamp}_{os.getpid()}.log"
    progress, display = Progress(), Display(stream)
    started = time.monotonic()
    print(f"상세 로그: {log}", file=stream, flush=True)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    interrupted = None
    last_draw = 0.
    handlers = {}
    proc = None
    try:
        with log.open("xb") as recording, selectors.DefaultSelector() as selector:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
            selector.register(proc.stdout, selectors.EVENT_READ)

            def stop(signum, _):
                nonlocal interrupted
                if interrupted is None:
                    interrupted = (signum, time.monotonic())
                    progress.state = "중단 요청 전달 중 — 종료·재개 상태 저장 대기"
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signum)
                    except ProcessLookupError:
                        pass

            for sig in (signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.signal(sig, stop)
            while selector.get_map() or proc.poll() is None:
                for key, _ in selector.select(timeout=.2):
                    data = os.read(key.fd, 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    recording.write(data); recording.flush()
                    text = decoder.decode(data)
                    progress.feed(text)
                    if not live:
                        stream.write(text); stream.flush()
                now = time.monotonic()
                if (interrupted and proc.poll() is None and stop_grace_seconds > 0
                        and now - interrupted[1] > stop_grace_seconds):
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if live and now - last_draw >= .2:
                    display.draw(progress.rows(now - started, log)); last_draw = now
            text = decoder.decode(b"", final=True)
            progress.feed(text, final=True)
            if text and not live:
                stream.write(text); stream.flush()
            code = proc.wait()
    except BaseException:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if proc is not None and proc.stdout is not None:
            proc.stdout.close()
    exit_code = 128 + interrupted[0] if interrupted else (128 - code if code < 0 else code)
    progress.state = f"중단됨 (종료 코드 {exit_code})" if interrupted else (
        "명령 정상 종료" if exit_code == 0 else f"실패 (종료 코드 {exit_code})")
    if live:
        display.draw(progress.rows(time.monotonic() - started, log))
        if exit_code:
            print("\n마지막 작업 출력:\n" + "\n".join(progress.tail), file=stream)
        elif progress.summary:
            print("\n" + "\n".join(progress.summary), file=stream)
    print(f"{progress.state} | 전체 로그: {log}", file=stream, flush=True)
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--display", choices=("auto", "live", "plain"), default=os.environ.get("BSU_PROGRESS", "auto"))
    parser.add_argument("--log-dir", type=Path, default=Path(".local/terminal_logs"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a worker command is required after --")
    if args.display not in ("auto", "live", "plain"):
        parser.error("BSU_PROGRESS must be auto, live or plain")
    raise SystemExit(run(command, args.log_dir, args.display))


if __name__ == "__main__":
    main()
