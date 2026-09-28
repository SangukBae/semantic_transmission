"""Full 60-second, all-frame SKEM/AWGN/Open-Sora development reconstruction."""
import argparse
import ast
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from .artifacts import sha256, write_json
from .cli import doctor, repository, settings
from .etri_60s_check import load_input, pts_audit, audit_prepared, require, run_command, lock
from .temporal import segment_lengths, output_source_indices
from .video_io import probe
from .webvid5 import execution_identity, fingerprint, read_json
from .webvid_ablation import Stages, snapshot, environment

MODULE = "semantic_transmission.etri_60s"
STAGES = ("prepare", "input-audit", "select", "selection-audit", "semantic-clips", "caption", "flow", "send",
          "channel", "receive", "reconstruct", "output-audit", "evaluate", "comparison")


def selection_audit(run):
    cfg = read_json(run / "run_config.json")
    import csv
    with (run / "data/frames/sample/frames.csv").open() as stream:
        candidates = [int(Path(r["frame_path"]).stem) for r in csv.DictReader(stream)]
    require(candidates == list(range(cfg["frames"])), "SKEM did not receive every input frame")
    keys = read_json(run / "keyframes.json")["indices"]
    lengths = segment_lengths(keys)
    require(keys[-1] == cfg["frames"]-1, "keyframes do not cover the entire input")
    state = read_json(cfg["selector_checkpoint"])
    checksum = state.pop("checksum")
    require(fingerprint(state) == checksum and state["done"] == cfg["frames"], "SKEM checkpoint incomplete")
    require(sorted(set([int(n) for n in state["selected"]] + [cfg["frames"]-1])) == keys,
            "selected keyframes differ from SKEM progress")
    result = {"frames": cfg["frames"], "comparisons": cfg["frames"]-1,
              "keyframes": keys, "segment_count": len(keys)-1,
              "generated_segment_frames_including_overlap": lengths,
              "maximum_segment_frames": max(lengths),
              "automatic_keyframe_insertion": False, "long_segment_capacity": "not pre-certified"}
    write_json(run / "selection_audit.json", result)
    print(f"SKEM 완료: {len(keys)}개 키프레임, {len(keys)-1}개 구간, 최대 생성 {max(lengths)}프레임", flush=True)


def output_audit(run):
    import cv2
    cfg = read_json(run / "run_config.json")
    require(sha256(cfg["input"]) == cfg["input_sha256"], "source changed during execution")
    require(read_json(run / "input_audit.json")["source_sha256"] == cfg["input_sha256"], "prepared source changed")
    files = list((run / "receiver/reconstruction").glob("*.mp4"))
    require(len(files) == 1, "expected exactly one reconstruction")
    video = files[0]
    info = probe(video)
    require(all(info[k] == cfg[k] for k in ("frames", "fps", "width", "height")), "output is truncated or stretched")
    require(math.isclose(info["duration"], cfg["frames"]/cfg["fps"], abs_tol=1e-5), "output duration mismatch")
    timing = pts_audit(video, cfg["frames"], cfg["fps"])
    directory = video.with_name(video.stem + "_frames")
    paths = sorted(directory.glob("*.png"))
    require([p.name for p in paths] == [f"{i:05d}.png" for i in range(cfg["frames"])], "lossless frames missing or duplicated")
    for path in paths:
        frame = cv2.imread(str(path))
        require(frame is not None and frame.shape == (cfg["height"], cfg["width"], 3), f"invalid frame: {path}")
    inputs = read_json(run / "receiver/decoder_inputs.json")
    keys = read_json(run / "keyframes.json")["indices"]
    require(inputs["indices"] == keys, "received keyframe sequence changed")
    mapping = output_source_indices(keys, inputs["decoder"]["concatenation_policy"])
    require(mapping == list(range(cfg["frames"])), "output/source alignment is not one-to-one")
    traces = read_json(run / "receiver/reference_trace.json") if len(keys) > 2 else []
    require([t["loop"] for t in traces] == list(range(1, len(keys)-1)), "missing cross-segment state transfer")
    for trace in traces:
        require(trace["references_after"] == [n+1 for n in trace["references_before"]], "reference was not appended")
    channel = read_json(run / "channel_accounting.json")
    require(channel["status"] == "PASSED" and channel["metadata_exact_match"], "channel metadata failed")
    write_json(run / "output_audit.json", {"status": "PASSED", "video": str(video),
        "video_sha256": sha256(video), "time_axis": timing, "all_lossless_frames_decoded": True,
        "source_indices": mapping, "cross_segment_transfers": len(traces),
        "full_60s_reconstructed": cfg["frames"] == 1440 and cfg["fps"] == 24,
        "hallucination_review": "PENDING", "automatic_keyframe_insertion": False})


def worker(stage, run):
    repo, cfg = repository(), read_json(run / "run_config.json")
    if stage == "input-audit":
        require(sha256(cfg["input"]) == cfg["input_sha256"], "input hash differs from frozen manifest")
        audit_prepared(run)
    elif stage == "selection-audit":
        selection_audit(run)
    elif stage == "semantic-clips":
        from .workers import prepare_semantic_clips
        prepare_semantic_clips(cfg, repo, run)
    elif stage == "output-audit":
        output_audit(run)
    elif stage == "reconstruct":
        from .codec_transport import decoder_config_text
        from .decoder_runner import run as decode
        text = decoder_config_text(cfg, repo, read_json(run / "receiver/decoder_inputs.json"))
        text += f"\ncpu_video_storage={cfg['cpu_video_storage']!r}\n"
        path = run / "receiver/decoder_config.py"
        path.write_text(text)
        trace = run / "receiver/reference_trace.json"
        write_json(trace, [])
        env = dict(os.environ, PYTHONPATH=str(repo / ".local/vendor/Open-Sora"), ETRI_REFERENCE_TRACE=str(trace))
        decode(run / "receiver/metadata.csv", run / "receiver/reconstruction", "key_frames_received",
               run / "receiver", decoder=repo / "scripts/etri_decoder_probe.py", config=path, environment=env)
    elif stage == "comparison":
        video = read_json(run / "output_audit.json")["video"]
        dest = run.parent / "comparison.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(run / "data/normalized.mp4"),
            "-i", video, "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]", "-an",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p", str(dest)], check=True)
        pts_audit(dest, cfg["frames"], cfg["fps"])


def jobs(cfg):
    frames = [f"data/frames/sample/{i}.png" for i in range(cfg["frames"])]
    yield "prepare", "semantic_transmission.workers", ["data", "prepare.json"], [
        "prepare.json", "data/normalized.mp4", "data/16x24/videos.csv", "data/frames/sample/frames.csv", *frames]
    yield "input-audit", MODULE, ["input_audit.json"], None
    yield "select", "semantic_transmission.workers", ["keyframes.json", f"data/frames/sample/{cfg['method']}",
        "selector_resources.json", "selector_runtime.json"], None
    yield "selection-audit", MODULE, ["selection_audit.json"], None
    yield "semantic-clips", MODULE, ["semantic_clips_audit.json", "data/clips"], None
    yield "caption", "semantic_transmission.workers", ["captions.json", "caption_sampling.json"], None
    yield "flow", "semantic_transmission.workers", ["metadata_tx.json", "flow_sampling.json"], None
    transport = "semantic_transmission.codec_transport"
    yield "send", transport, ["transmitter", "sender_accounting.json"], None
    yield "channel", transport, ["received", "channel_accounting.json"], None
    yield "receive", transport, ["receiver/frames", "receiver/metadata.csv", "receiver/decoder_inputs.json", "receiver_accounting.json"], None
    yield "reconstruct", MODULE, ["receiver/decoder_config.py", "receiver/reconstruction", "receiver/reference_trace.json"], None
    yield "output-audit", MODULE, ["output_audit.json"], None
    yield "evaluate", "semantic_transmission.research_quality", ["quality.json", "quality_lossless_frames.csv", "quality_delivered_mp4.csv"], None
    yield "comparison", MODULE, [], []


def finalize(root, pipeline):
    run = root / "baseline"
    audit, quality = read_json(run / "output_audit.json"), read_json(run / "quality.json")
    require(audit["full_60s_reconstructed"] and quality["status"] == "PASSED", "60-second reconstruction/evaluation incomplete")
    require(quality["video_sha256"] == audit["video_sha256"] == sha256(audit["video"]), "evaluated video changed")
    require(quality["video"]["frames"] == 1440, "metrics do not cover 1440 frames")
    pts_audit(root / "comparison.mp4", 1440, 24)
    result = {"status": "PASS_60S_RECONSTRUCTION", "full_60s_reconstructed": True,
        "frames": 1440, "fps": 24, "duration_seconds": 60,
        "video": audit["video"], "comparison": str(root / "comparison.mp4"),
        "metrics": quality["delivered_mp4"], "keyframes": read_json(run / "keyframes.json")["indices"],
        "channel": read_json(run / "channel_accounting.json"),
        "hallucination_review": "PENDING", "hallucination_mitigation_verified": False,
        "semantic_clip_policy": read_json(run / "run_config.json").get("semantic_clip_policy"),
        "selection_reuse": read_json(run / "selection_reuse.json") if (run / "selection_reuse.json").exists() else None,
        "resume": "SKEM resumes after completed comparisons; captions resume after saved segments; other interrupted stages restart",
        "stage_seconds_latest_successful_attempt": {k: v["seconds"] for k, v in pipeline.completed.items()}}
    write_json(root / "RESULT.json", result)
    (root / "REPORT.md").write_text(
        "# ETRI 60초 기준선 복원\n\nPASS_60S_RECONSTRUCTION\n\n"
        f"- 1,440프레임 / 24fps / 60초. 키프레임 {len(result['keyframes'])}개.\n"
        f"- [복원 영상]({Path(audit['video']).relative_to(root)})\n"
        "- [동기 비교 영상](comparison.mp4): 왼쪽 원본, 오른쪽 복원.\n"
        "- [결과·지표·전송량](RESULT.json), [시간축 검증](baseline/output_audit.json).\n"
        "- 실행·파일·시간축 통과이며 Added/Missing/Distorted 검수와 완화 성능은 미완료입니다.\n"
        "- 결과는 개발 입력 한 편의 기준선이며 ETRI 전체 성능평가 완료를 의미하지 않습니다.\n")
    return result


def execute(repo, root, cfg, signature, stop_after=None, selection_reuse=None):
    local = settings(repo)
    env = dict(environment(cfg["seed"]), ETRI_RUN_SIGNATURE=signature)
    pipeline = Stages(root, signature)
    run = root / "baseline"
    actual_cfg = dict(cfg, selector_checkpoint=str(root / "checkpoints/selector.json"),
                      caption_checkpoint=str(root / "checkpoints/caption.json"))
    pipeline.step("config", ["baseline/run_config.json"], ["baseline/run_config.json"],
                  lambda _: write_json(run / "run_config.json", actual_cfg))
    for name, module, owned, required in jobs(actual_cfg):
        required = owned if required is None else required
        owned = [f"baseline/{p}" for p in owned]
        required = [f"baseline/{p}" for p in required]
        if name == "comparison":
            owned = required = ["comparison.mp4"]
        # Selector state survives failed stage archiving, but a successful receipt binds its hash.
        if name == "select":
            required = [*required, "checkpoints/selector.json"]
            if selection_reuse:
                owned.append("baseline/selection_reuse.json")
                required.append("baseline/selection_reuse.json")
        if name == "caption":
            required.append("checkpoints/caption.json")
        args = [local["channel_python"] if name == "channel" else local["python"], "-m", module]
        args += ["--worker", name, "--run-dir", str(run)] if module == MODULE else [name, str(run)]
        metrics, timing, progress = f"resources/{name}.json", f"resources/{name}.time.txt", f"progress/{name}.json"
        child_env = dict(env, CUDA_VISIBLE_DEVICES="-1") if name == "channel" else env
        if name == "select" and selection_reuse:
            def launch(log):
                from .etri_selection_reuse import import_selection
                started = time.monotonic()
                source = Path(selection_reuse["source_root"])
                import_selection(source, root, selection_reuse)
                seconds = time.monotonic() - started
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(f"Verified completed SKEM imported from {source}; no model inference.\n")
                write_json(root / metrics, {"mode": "verified_selection_import", "seconds": seconds,
                    "source": str(source), "returncode": 0})
                (root / timing).write_text(f"verified import elapsed seconds: {seconds}\n")
                write_json(root / progress, {"done": cfg["frames"], "total": cfg["frames"], "reused": True})
                print("기존 SKEM 키프레임 재사용 완료 (모델 재실행 없음)", flush=True)
        else:
            launch = lambda log, a=args, e=child_env, m=metrics, p=progress: run_command(repo, a, log, e, root / m, root / p)
        pipeline.step(name, [*owned, metrics, timing, progress], [*required, metrics, timing], launch)
        if stop_after == name:
            return {"status": "STOPPED_AFTER_STAGE", "stage": name, "full_60s_reconstructed": False}
    return finalize(root, pipeline)


def status(root):
    state = read_json(root / "status.json") if (root / "status.json").exists() else {"status": "NOT_STARTED"}
    receipts = [(p.stem, read_json(p)) for p in (root / "stages").glob("*.json")]
    state["completed_stages"] = [n for n, r in receipts if r.get("status") == "PASSED"]
    state["active_or_failed"] = {n: r for n, r in receipts if r.get("status") != "PASSED"}
    state["output"] = str(root)
    if (root / "checkpoints/selector.json").exists():
        saved = read_json(root / "checkpoints/selector.json")
        state["selector_progress"] = {"done": saved["done"], "total": 1440}
    if (root / "checkpoints/caption.json").exists():
        saved = read_json(root / "checkpoints/caption.json")
        state["caption_checkpoint"] = str(root / "checkpoints/caption.json")
        state["caption_progress"] = {"done": len(saved["records"]), "total": saved["total"]}
    print(json.dumps(state, indent=2, ensure_ascii=False))


def run(repo, args):
    manifest = (args.manifest or repo / "data/etri_benchmark_v1_20260924/manifest.json").resolve()
    row, cfg = load_input(repo, manifest, args.video)
    cfg.update(profile="etri_60s_baseline_v2", cpu_video_storage=True, input_sha256=row["processed_sha256"],
               semantic_clip_policy="frame_exact")
    require(cfg["frames"] == 1440 and cfg["selection_stride"] == 1, "full reconstruction requires all 1440 frames")
    if doctor(repo)["status"] != "PASSED":
        raise RuntimeError("environment/model preflight failed")
    execution = execution_identity(repo, cfg)
    tree = ast.parse((repo / "src/semantic_transmission/workers.py").read_text())
    selector = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "select")
    execution["selector_entrypoint_sha256"] = hashlib.sha256(ast.dump(selector, include_attributes=False).encode()).hexdigest()
    selection_reuse = None
    if args.reuse_selection_from:
        from .etri_selection_reuse import inspect_source
        selection_reuse = inspect_source(repo, args.reuse_selection_from.resolve(), cfg, execution)
    identity = {"version": 2, "selection": row, "config": cfg, "execution": execution,
        "selection_reuse": selection_reuse,
        "extra_code": {p: sha256(repo / p) for p in ["scripts/run_etri_60s.sh", "scripts/etri_decoder_probe.py",
                                                    "configs/webvid5.json", "configs/official_opensora.py"]}}
    signature = fingerprint(identity)
    root = (args.output or repo / "outputs" / f"etri_60s_{args.video}_{signature[:12]}").resolve()
    protocol = root / "protocol.json"
    if root.exists():
        require(protocol.exists() and read_json(protocol)["signature"] == signature,
                f"output configuration/code differs; choose a new --output: {root}")
    estimate = ("검증된 SKEM 재사용 / 남은 자동 실행 약 3~6시간(추정)." if selection_reuse else
                "SKEM: 1439회 비교 / 첫 실행 예상 15~35시간(영상별 차이).")
    print(f"60초 전체 복원: {args.video} | 1440 frames / 24fps / 576×320\n"
          f"{estimate}\nAWGN 10dB / Open-Sora 30 steps / 프레임 기준 클립 사전 검증\n"
          f"결과: {root}", flush=True)
    if args.status:
        status(root)
        return root
    if args.dry_run:
        for path in (root / "stages").glob("*.json"):
            saved = read_json(path)
            if saved.get("status") == "PASSED":
                require(snapshot(root, saved["required"]) == saved["artifacts"], f"changed artifacts: {path}")
        print("DRY_RUN_PASSED: 입력·환경 확인. 실제 복원은 실행하지 않았습니다.", flush=True)
        return root
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / ".lock"):
        if not protocol.exists():
            write_json(protocol, dict(identity, signature=signature))
        for name in ("RESULT.json", "REPORT.md"):
            (root / name).unlink(missing_ok=True)
        state = {"status": "RUNNING", "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "full_60s_reconstructed": False, "signature": signature}
        write_json(root / "status.json", state)
        try:
            result = execute(repo, root, cfg, signature, args.stop_after, selection_reuse)
            state.update(status=result["status"], full_60s_reconstructed=result["full_60s_reconstructed"])
            if args.stop_after:
                state["stopped_after"] = args.stop_after
        except BaseException as error:
            state.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=str(error))
            raise
        finally:
            state["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            write_json(root / "status.json", state)
    print(f"{state['status']}: {root}", flush=True)
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="tv_low_08", help="development input ID")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse-selection-from", type=Path, help="verify and import completed SKEM from a previous run into a new run")
    parser.add_argument("--dry-run", action="store_true", help="read-only preflight; no inference")
    parser.add_argument("--status", action="store_true", help="read current progress without starting workers")
    parser.add_argument("--stop-after", choices=STAGES, help="stop after this completed stage; default runs everything")
    parser.add_argument("--worker", choices=["input-audit", "selection-audit", "semantic-clips", "reconstruct", "output-audit", "comparison"], help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
    try:
        if args.worker:
            require(args.run_dir is not None, "worker requires run-dir")
            worker(args.worker, args.run_dir.resolve())
        elif args.status and args.output:
            status(args.output.resolve())
        elif args.status or args.dry_run:
            run(repository(), args)
        else:
            # Shares the integration check's GPU lock; other campaigns still require coordination.
            with lock(repository() / ".local/etri_60s_check.lock"):
                run(repository(), args)
    except KeyboardInterrupt:
        parser.exit(130, "중단됨. 같은 명령으로 SKEM·캡션 진행 상태와 완료 단계를 재사용합니다.\n")
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f"60초 복원 실패: {error}\n")


if __name__ == "__main__":
    main()
