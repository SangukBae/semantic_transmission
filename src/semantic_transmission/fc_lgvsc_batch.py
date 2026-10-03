"""Run the four prepared FC-LGVSC videos sequentially with one progress line."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import traceback

from . import fc_lgvsc as default
from .artifacts import sha256, write_json
from .reconstruction_console import Progress, GrowingLog, GpuUsage, stop

REPO = default.REPO
VIDEOS = ("tv_low_08", "single_subject", "candle_flowers", "person_walk")


def preflight(output_root=None):
    """Validate all four before any worker or output directory is created."""
    jobs = []
    for video in VIDEOS:
        source = default.base.source_for(video).resolve()
        output = ((output_root / video) if output_root else
                  source.with_name(source.name + "_fc_lgvsc_default_v1")).resolve()
        identity, signature = default.preflight(source, output)
        jobs.append(dict(video=video, output=str(output), signature=signature,
            frames=identity["frames"], fps=identity["fps"], segments=len(identity["plan"]),
            steps=identity["baseline_noise_contract"]["steps"], t5_compute_dtype=identity["t5_compute_dtype"],
            condition_collision_policy=identity["condition_collision_policy"],
            short_schedule_policy=identity["short_schedule_policy"]))
    return jobs


def worker_command(job):
    # The default runner revalidates its protocol, locks the GPU job and resumes
    # verified stages. Do not skip a video merely because review.html exists.
    return [sys.executable, "-m", default.MODULE, "--execute",
            "--video", job["video"], "--output", job["output"]]


def console_run(jobs, *, stream=None, interval=0.2, gpu_monitor=None, logroot=None):
    stream = sys.stdout if stream is None else stream
    monitor = gpu_monitor or GpuUsage()
    total = sum(job["segments"] * job["steps"] for job in jobs)
    if not jobs or total <= 0:
        raise ValueError("positive batch work required")
    logroot = logroot or REPO / ".local/reconstruction_console"
    logroot.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="fc-lgvsc-all-", dir=logroot))
    write_json(folder / "invocation.json", dict(policy=default.POLICY, videos=jobs,
        code={name:sha256(REPO/name) for name in
              ("src/semantic_transmission/fc_lgvsc_batch.py", "scripts/reconstruct_fc_lgvsc_all.sh")}))
    completed, last = 0, ""
    for index, job in enumerate(jobs):
        video, output = job["video"], Path(job["output"])
        work = job["segments"] * job["steps"]
        progress = Progress(job["segments"], job["steps"])
        log, progress_path = folder / f"{video}.log", folder / f"{video}.progress.json"
        root_log = GrowingLog(log)
        decoder_log = GrowingLog(output / "run/receiver/reconstruction/00000.log")
        env = dict(os.environ, QUALITY_PROGRESS_FILE=str(progress_path))
        process, code = None, 1
        try:
            with log.open("x") as capture:
                process = subprocess.Popen(worker_command(job), cwd=REPO, env=env,
                    stdout=capture, stderr=subprocess.STDOUT, start_new_session=True)
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
                    percent = min(99.99, 100 * (completed + work * progress.percent / 100) / total)
                    gpu = monitor.read()
                    usage = f"{gpu:3d}%" if gpu is not None else " --%"
                    line = (f"복원 진행률: {percent:6.2f}% | GPU 사용률: {usage} | "
                            f"{index+1}/{len(jobs)} {video}")
                    if line != last:
                        stream.write("\r" + line.ljust(max(len(last), len(line))))
                        stream.flush()
                        last = line
                    try:
                        code = process.wait(timeout=interval)
                        break
                    except subprocess.TimeoutExpired:
                        pass
        except KeyboardInterrupt:
            if process is not None:
                stop(process)
            code = 130
        except Exception:
            if process is not None:
                stop(process)
            with log.open("a") as capture:
                traceback.print_exc(file=capture)
            code = 1
        if code:
            status = "중단" if code == 130 else "실패"
            stream.write(f"\n복원 {status} ({video}) · 상세 로그: {log}\n")
            stream.flush()
            return code
        completed += work
    line = f"복원 진행률: 100.00% | 완료: {len(jobs)}개 영상"
    stream.write("\r" + line.ljust(max(len(last), len(line))) + "\n")
    for job in jobs:
        stream.write(f"{job['video']}: {Path(job['output']) / 'review.html'}\n")
    stream.flush()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="read-only check of all four inputs")
    parser.add_argument("--output-root", type=Path, help="optional parent for four separate output folders")
    args = parser.parse_args(argv)
    jobs = preflight(args.output_root)
    if args.check:
        print(json.dumps(dict(status="READY", policy=default.POLICY, videos=jobs,
            model_inference_started=False), ensure_ascii=False, indent=2))
        return 0
    def interrupted(*_):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        return console_run(jobs)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
