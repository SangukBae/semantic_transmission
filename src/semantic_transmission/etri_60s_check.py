"""One-command 60-second input check and bounded real-model integration probe.

This does not run 60-second SKEM/generation or certify long-video quality.
Frozen benchmark inputs and legacy profiles are read-only.
"""
import argparse
import contextlib
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .artifacts import sha256, write_json
from .cli import doctor, repository, settings
from .input_contract import video_config
from .temporal import (concatenate_segments, conditioning_indices, output_source_indices,
                       unique_output_positions)
from .video_io import probe
from .webvid5 import execution_identity, fingerprint, read_json
from .webvid_ablation import Stages, environment, snapshot

VERSION = 1
PROBE_FRAMES = 25
PROBE_KEYS = [0, 8, 24]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_input(repo, manifest, video_id):
    rows = read_json(manifest)
    matches = [r for r in rows if r["id"] == video_id]
    require(len(matches) == 1, "input ID must occur exactly once")
    row = matches[0]
    require(row["split"] == "development", "only development inputs are allowed")
    require(row["frames"] == 1440 and row["clip_duration_sec"] == 60,
            "this check requires one canonical 60-second, 1440-frame input")
    frozen = {r["path"]: r["sha256"] for r in read_json(manifest.parent / "freeze_manifest.json")["files"]}
    for path in (manifest.resolve(), Path(row["processed_path"]).resolve()):
        require(frozen.get(str(path)) == sha256(path), f"frozen input changed: {path}")
    source = Path(row["processed_path"]).resolve(strict=True)
    require(sha256(source) == row["processed_sha256"], "source checksum mismatch")
    info = probe(source)
    cfg = read_json(repo / "configs/webvid5.json")
    cfg.update(profile="etri_60s_development_check_v1", frames=1440, max_frames=1440,
               seed=2025, channel_seed=42, concatenation_policy="endpoint_exact")
    cfg = video_config(cfg, info)
    require(math.isclose(info["duration"], 60, abs_tol=1e-5), "duration is not 60 seconds")
    cfg.update(input=str(source), models=read_json(repo / ".local/model_paths.json"))
    return row, cfg


def pts_audit(path, frames, fps):
    raw = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path)], text=True)
    times = [float(f["best_effort_timestamp_time"]) for f in json.loads(raw)["frames"]]
    require(len(times) == frames, "PTS frame count mismatch")
    require(all(math.isfinite(t) and abs(t - i / fps) <= 2e-6 for i, t in enumerate(times)),
            "PTS is discontinuous, shifted or not canonical CFR")
    return {"frames": frames, "fps": fps, "first_pts": times[0], "last_pts": times[-1],
            "duration_seconds": frames / fps, "all_pts_checked": True}


def audit_prepared(run):
    import cv2
    import numpy as np
    cfg = read_json(run / "run_config.json")
    video = run / "data/normalized.mp4"
    require(sha256(video) == sha256(cfg["input"]), "prepare changed source bytes")
    result = pts_audit(video, cfg["frames"], cfg["fps"])
    cap = cv2.VideoCapture(str(video))
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            saved = cv2.imread(str(run / f"data/frames/sample/{count}.png"))
            require(saved is not None and np.array_equal(frame, saved), f"PNG mismatch at {count}")
            count += 1
    finally:
        cap.release()
    require(count == cfg["frames"], "decode truncated input")
    require(len(list((run / "data/frames/sample").glob("*.png"))) == count, "extra prepared PNGs")
    result.update(all_png_pixels_checked=True, source_sha256=sha256(video))
    write_json(run / "input_audit.json", result)


def temporal_check(destination):
    import torch
    cases = [[0, 1439], [0, 8, 24, 383, 768, 1439], list(range(1440))]
    results = []
    for keys in cases:
        segments = []
        for i, (a, b) in enumerate(zip(keys, keys[1:])):
            body = torch.arange(a, b + 1)
            if i:
                previous = segments[-1]
                body = torch.cat([previous[conditioning_indices(len(previous))], body])
            segments.append(body)
        shaped = [s.reshape(1, -1, 1, 1) for s in segments]
        exact = concatenate_segments(shaped, policy="endpoint_exact").flatten().tolist()
        released = concatenate_segments(shaped, policy="official_release").flatten().tolist()
        require(exact == list(range(1440)), "concatenation lost or duplicated a source frame")
        require(released == output_source_indices(keys, "official_release"), "release accounting mismatch")
        require([released[i] for i in unique_output_positions(keys)] == exact, "alignment mismatch")
        results.append({"keyframes": len(keys), "endpoint_exact_frames": len(exact),
                        "official_release_frames": len(released), "identical_time_mapping": True})
    write_json(destination, {"status": "PASSED", "scope": "synthetic frame-index tensors, no generation",
                             "cases": results})


def run_command(repo, args, log, env, metrics, progress):
    """Isolated process group, live progress, measured wall/RSS and sampled GPU usage."""
    log.parent.mkdir(parents=True, exist_ok=True)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    timing = metrics.with_suffix(".time.txt")
    child_env = dict(env, QUALITY_PROGRESS_FILE=str(progress), LC_ALL="C")
    start, samples, code = time.monotonic(), [], None
    error = None
    with log.open("x") as stream:
        process = subprocess.Popen(["/usr/bin/time", "-v", "-o", str(timing), *map(str, args)],
            cwd=repo, env=child_env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        last_print = start
        try:
            while True:
                try:
                    code = process.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    try:
                        value = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"], text=True, timeout=2)
                        samples.append([int(v.strip()) for v in value.splitlines()])
                    except (OSError, ValueError, subprocess.SubprocessError):
                        pass
                    if time.monotonic() - last_print >= 30:
                        detail = read_json(progress) if progress.exists() else {}
                        print(f"  {log.stem}: {(time.monotonic()-start)/60:.1f}분 {detail}", flush=True)
                        last_print = time.monotonic()
            if code:
                raise RuntimeError(f"exit {code}; log: {log}")
        except BaseException as exc:
            error = str(exc) or type(exc).__name__
            # Always kill the group: a dead parent can still have live descendants.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            rss = None
            if timing.exists():
                for line in timing.read_text().splitlines():
                    if "Maximum resident set size (kbytes):" in line:
                        rss = int(line.rsplit(":", 1)[1])
            write_json(metrics, {"command": list(map(str, args)), "seconds": time.monotonic()-start,
                "returncode": process.returncode, "error": error, "max_rss_kib": rss,
                "gpu_sample_count": len(samples),
                "gpu_memory_peak_mib_per_device": [max(s[i] for s in samples) for i in range(len(samples[0]))] if samples else [],
                "gpu_measurement": "sampled whole-device usage every ~2 seconds; includes other GPU users"})


def resume_probe(root):
    """Exercise verified reuse, interrupted output retention, and tamper rejection."""
    root.mkdir(parents=True)
    calls = []
    def good(log):
        calls.append("good")
        (root / "good").write_text("complete")
    def fail(log):
        (root / "partial").write_text("interrupted")
        raise InterruptedError("deliberate resume test")
    first = Stages(root, "resume-selftest")
    first.step("a", ["good"], ["good"], good)
    try:
        first.step("b", ["partial"], ["partial"], fail)
    except InterruptedError:
        pass
    second = Stages(root, "resume-selftest")
    second.step("a", ["good"], ["good"], good)
    second.step("b", ["partial"], ["partial"], lambda _: (root / "partial").write_text("restarted"))
    require(calls == ["good"], "completed step was repeated")
    require(any((root / "failed_attempts").glob("b_*/partial")), "partial output not preserved")
    (root / "good").write_text("tampered")
    try:
        Stages(root, "resume-selftest").step("a", ["good"], ["good"], good)
    except ValueError:
        write_json(root / "result.json", {"status": "PASSED", "completed_reused": True,
                   "partial_preserved": True, "tampering_rejected": True,
                   "scope": "injected stage interruption; no mid-stage model checkpoint resume"})
        return
    raise ValueError("tampering was accepted")


def make_probe_config(cfg, root, selector=False):
    result = dict(cfg, input=str(root / "prefix.mp4"), frames=PROBE_FRAMES,
                  profile="etri_short_integration_probe_not_baseline",
                  selection_stride=12 if selector else 1)
    if not selector:
        result.update(selector="fixed_diagnostic", method="key_frames_probe")
    return result


def fixed_keys(run):
    cfg = read_json(run / "run_config.json")
    folder = run / "data/frames/sample" / cfg["method"]
    folder.mkdir()
    for i in PROBE_KEYS:
        shutil.copyfile(run / f"data/frames/sample/{i}.png", folder / f"{i}.png")
    write_json(run / "keyframes.json", {"indices": PROBE_KEYS, "selector": "fixed diagnostic, not SKEM",
               "reason": "force short first segment and transfer of generated reference into second segment"})


def audit_smoke(run):
    cfg = read_json(run / "run_config.json")
    files = list((run / "receiver/reconstruction").glob("*.mp4"))
    require(len(files) == 1, "expected exactly one generated video")
    info = probe(files[0])
    require(all(info[k] == cfg[k] for k in ("frames", "fps", "width", "height")), "smoke output dimensions/time mismatch")
    pts = pts_audit(files[0], cfg["frames"], cfg["fps"])
    pngs = list((run / "receiver/reconstruction").glob("*_frames/*.png"))
    require(len(pngs) == cfg["frames"], "lossless output frame count mismatch")
    trace = read_json(run / "receiver/reference_trace.json")
    require(len(trace) == 1 and trace[0]["loop"] == 1, "missing actual cross-segment reference trace")
    record = trace[0]
    require(record["generated_reference_shape"][2] == 17, "short reference was not padded to 17 frames")
    require(record["references_after"] == [x+1 for x in record["references_before"]], "reference not appended")
    require(all(s[1] == 5 for s in record["new_latent_shapes"]), "reference did not produce five latents")
    channel = read_json(run / "channel_accounting.json")
    require(channel["status"] == "PASSED" and channel["metadata_exact_match"], "AWGN packet not recovered")
    write_json(run / "audit.json", {"status": "PASSED_SHORT_INTEGRATION", "video": str(files[0]),
               "time_axis": pts, "generated_reference": trace, "full_60s_reconstructed": False})


def worker(name, run):
    repo = repository()
    if name == "audit-input":
        audit_prepared(run)
    elif name == "temporal":
        temporal_check(run / "temporal.json")
    elif name == "audit-smoke":
        audit_smoke(run)
    elif name == "reconstruct":
        from .codec_transport import decoder_config_text
        from .decoder_runner import run as decode
        cfg = read_json(run / "run_config.json")
        text = decoder_config_text(cfg, repo, read_json(run / "receiver/decoder_inputs.json"))
        path = run / "receiver/decoder_config.py"
        path.write_text(text)
        env = dict(os.environ, PYTHONPATH=str(repo / ".local/vendor/Open-Sora"),
                   ETRI_REFERENCE_TRACE=str(run / "receiver/reference_trace.json"))
        decode(run / "receiver/metadata.csv", run / "receiver/reconstruction", "key_frames_received",
               run / "receiver", decoder=repo / "scripts/etri_decoder_probe.py", config=path, environment=env)


def execute(repo, root, cfg, signature, cpu_only):
    local, env = settings(repo), environment(cfg["seed"])
    pipeline = Stages(root, signature)
    module = "semantic_transmission.etri_60s_check"
    def simple(name, paths, callback):
        pipeline.step(name, paths, paths, callback)
    def launch(name, owned, required, command, channel=False):
        metrics, progress = f"resources/{name}.json", f"progress/{name}.json"
        files = [metrics, f"resources/{name}.time.txt", progress]
        pipeline.step(name, [*owned, *files], [*required, metrics, files[1]],
            lambda log: run_command(repo, command, log, dict(env, CUDA_VISIBLE_DEVICES="-1") if channel else env,
                                    root / metrics, root / progress))
    def stage(case, name, owned, required=None, adapter="semantic_transmission.workers"):
        run = root / case
        args = [local["channel_python"] if name == "channel" else local["python"], "-m", adapter]
        args += ["--worker", name, "--run-dir", str(run)] if adapter == module else [name, str(run)]
        launch(f"{case}.{name}", [f"{case}/{p}" for p in owned],
               [f"{case}/{p}" for p in (required if required is not None else owned)], args, name == "channel")
    def prepare(case, count):
        stage(case, "prepare", ["data", "prepare.json"], ["prepare.json", "data/normalized.mp4",
              "data/16x24/videos.csv", "data/frames/sample/frames.csv",
              *[f"data/frames/sample/{i}.png" for i in range(count)]])
    simple("profile", ["full/run_config.json"], lambda _: write_json(root / "full/run_config.json", cfg))
    prepare("full", cfg["frames"])
    stage("full", "audit-input", ["input_audit.json"], adapter=module)
    launch("temporal", ["temporal.json"], ["temporal.json"],
           [local["python"], "-m", module, "--worker", "temporal", "--run-dir", str(root)])
    simple("resume", ["resume_probe"], lambda _: resume_probe(root / "resume_probe"))
    if not cpu_only:
        launch("prefix", ["prefix.mp4"], ["prefix.mp4"], ["ffmpeg", "-v", "error", "-nostdin",
            "-i", cfg["input"], "-frames:v", str(PROBE_FRAMES), "-an", "-c:v", "libx264",
            "-crf", "0", "-preset", "fast", "-pix_fmt", "yuv420p", str(root / "prefix.mp4")])
        for case in ("selector", "smoke"):
            config = make_probe_config(cfg, root, selector=case == "selector")
            simple(f"{case}.config", [f"{case}/run_config.json"],
                   lambda _, c=case, v=config: write_json(root / c / "run_config.json", v))
            prepare(case, PROBE_FRAMES)
        stage("selector", "select", ["keyframes.json", f"data/frames/sample/{cfg['method']}",
            "selector_runtime.json", "selector_resources.json"])
        simple("smoke.keys", ["smoke/keyframes.json", "smoke/data/frames/sample/key_frames_probe"],
               lambda _: fixed_keys(root / "smoke"))
        stage("smoke", "caption", ["captions.json", "caption_sampling.json", "data/clips"])
        stage("smoke", "flow", ["metadata_tx.json", "flow_sampling.json"])
        transport = "semantic_transmission.codec_transport"
        stage("smoke", "send", ["transmitter", "sender_accounting.json"], adapter=transport)
        stage("smoke", "channel", ["received", "channel_accounting.json"], adapter=transport)
        stage("smoke", "receive", ["receiver/frames", "receiver/metadata.csv", "receiver/decoder_inputs.json",
                                   "receiver_accounting.json"], adapter=transport)
        stage("smoke", "reconstruct", ["receiver/decoder_config.py", "receiver/reconstruction",
                                       "receiver/reference_trace.json"], adapter=module)
        stage("smoke", "audit-smoke", ["audit.json"], adapter=module)
    result = {"status": "PASS_CPU_CHECKS" if cpu_only else "PASS_60S_INPUT_AND_SHORT_MODEL_CHECK",
              "signature": signature, "full_input_frames_verified": cfg["frames"],
              "short_model_probe_frames": 0 if cpu_only else PROBE_FRAMES,
              "short_model_probe_passed": not cpu_only, "full_60s_reconstructed": False,
              "long_segment_gpu_capacity_verified": False, "hallucination_mitigation_verified": False,
              "resume_granularity": "completed stages only; interrupted stage restarts",
              "stage_seconds": {k: v["seconds"] for k, v in pipeline.completed.items()}}
    write_json(root / "RESULT.json", result)
    (root / "REPORT.md").write_text(
        f"# ETRI 60초 실행 준비 점검\n\n{result['status']}\n\n"
        "- 60초 입력 1,440프레임의 PTS·전처리 픽셀·합성 구간 연결을 확인했습니다.\n"
        f"- 실제 짧은 모델 실행: {'미실행(CPU 전용)' if cpu_only else '25프레임, 고정 키프레임 0/8/24, 두 생성 구간'}.\n"
        "- 60초 전체 SKEM·복원, 긴 생성 구간의 GPU 메모리, 완화 성능은 미검증입니다.\n"
        "- full/run_config.json: 60초 개발 설정. 기준선 확정이나 평가 조건 동결이 아닙니다.\n"
        "- logs/, resources/, progress/, stages/: 로그·자원 측정·진행률·재개 근거.\n"
        "- 같은 명령은 해시를 확인한 완료 단계를 재사용합니다. 실패한 단계는 처음부터 재실행합니다.\n")
    return result


@contextlib.contextmanager
def lock(path):
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("60초 점검 명령이 이미 실행 중입니다") from None
        yield


def run(repo, args):
    manifest = (args.manifest or repo / "data/etri_benchmark_v1_20260924/manifest.json").resolve()
    row, cfg = load_input(repo, manifest, args.video)
    if doctor(repo, check_models=not args.cpu_only)["status"] != "PASSED":
        raise RuntimeError("environment preflight failed")
    require(Path("/usr/bin/time").is_file(), "GNU time is required")
    identity = {"version": VERSION, "selection": row, "config": cfg, "cpu_only": args.cpu_only,
                "manifest_sha256": sha256(manifest), "execution": execution_identity(repo, cfg),
                "extra_code": {p: sha256(repo / p) for p in ["scripts/check_etri_60s.sh",
                    "scripts/etri_decoder_probe.py", "configs/webvid5.json", "configs/official_opensora.py"]}}
    signature = fingerprint(identity)
    root = (args.output or repo / "outputs" / f"etri_60s_check_{args.video}_{signature[:12]}").resolve()
    protocol = root / "protocol.json"
    if root.exists():
        require(protocol.exists() and read_json(protocol).get("signature") == signature,
                f"output identity changed or unknown directory; use a new --output: {root}")
    print(f"60초 입력 점검: {args.video} | 1440 frames / 24 fps\n"
          f"짧은 모델 실행: {'생략(CPU 전용)' if args.cpu_only else 'SKEM 2회 비교 + 25프레임/2구간 송수신·생성'}\n"
          f"결과: {root}", flush=True)
    if args.dry_run:
        for path in sorted((root / "stages").glob("*.json")):
            receipt = read_json(path)
            if receipt.get("status") == "PASSED":
                require(snapshot(root, receipt["required"]) == receipt["artifacts"], f"changed artifact: {path}")
        print("DRY_RUN_PASSED: no inference or output writes", flush=True)
        return root
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / ".lock"):
        if not protocol.exists():
            write_json(protocol, dict(identity, signature=signature))
        # A stale PASS must not survive an interrupted or rejected resume.
        (root / "RESULT.json").unlink(missing_ok=True)
        (root / "REPORT.md").unlink(missing_ok=True)
        state = {"status": "RUNNING", "started": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        write_json(root / "status.json", state)
        try:
            result = execute(repo, root, cfg, signature, args.cpu_only)
            state["status"] = result["status"]
        except BaseException as exc:
            state.update(status="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                         error=str(exc) or type(exc).__name__)
            raise
        finally:
            state["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            write_json(root / "status.json", state)
    print(f"완료: {root / 'REPORT.md'}", flush=True)
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="tv_low_08", help="development input ID")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, help="same command and output resumes completed stages")
    parser.add_argument("--cpu-only", action="store_true", help="input/temporal/resume checks only; not model PASS")
    parser.add_argument("--dry-run", action="store_true", help="read-only input/environment/resume validation")
    parser.add_argument("--worker", choices=["audit-input", "temporal", "reconstruct", "audit-smoke"], help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
    try:
        if args.worker:
            require(args.run_dir is not None, "worker requires --run-dir")
            worker(args.worker, args.run_dir.resolve())
        elif args.dry_run:
            run(repository(), args)
        else:
            with lock(repository() / ".local/etri_60s_check.lock"):
                run(repository(), args)
    except KeyboardInterrupt:
        parser.exit(130, "중단됨. 같은 명령으로 완료 단계 이후부터 재개합니다.\n")
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"점검 실패: {exc}\n")


if __name__ == "__main__":
    main()
