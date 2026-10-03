"""Reconstruct the three frozen hybrid/v2 inputs, sequentially, on user command.

This runner does not select keyframes or generate captions. Source-only preparation
is separate. All existing experiment runners and their pinned code remain intact.
"""
import argparse
import html
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile

from . import assisted_captions as captions, hybrid_reconstruction as hybrid
from . import precision_reconstruction as precision, precision_text_cache as text
from . import generation_noise as noise, tail_reference as tail
from .artifacts import sha256, write_json
from .cli import settings
from .hybrid_selection import verify_selection
from .temporal import output_source_indices, unique_output_positions
from .webvid5 import fingerprint, read_json
from .webvid_ablation import Stages, environment, snapshot

REPO = hybrid.REPO
ROOT = REPO / "outputs/etri_latest_short3_20261001"
NAMES = ("single_subject", "candle_flowers", "person_walk")
MODULE = "semantic_transmission.short_video_batch"
METRICS = ("psnr_db", "ssim", "lpips_vgg", "clip", "dists")
CODE = tuple(sorted(set(precision.CODE) | {
    "src/semantic_transmission/short_video_batch.py", "scripts/reconstruct_short3.sh",
    "src/semantic_transmission/hybrid_reconstruction.py", "src/semantic_transmission/assisted_captions.py",
    "src/semantic_transmission/hybrid_selection.py", "src/semantic_transmission/reconstruction_console.py",
    "src/semantic_transmission/etri_60s.py", "src/semantic_transmission/etri_60s_check.py",
    "src/semantic_transmission/workers.py", "src/semantic_transmission/semantic_clips.py",
    "src/semantic_transmission/codec_transport.py", "src/semantic_transmission/wire.py",
    "src/semantic_transmission/metadata_channel.py", "src/semantic_transmission/transmission_accounting.py",
    "src/semantic_transmission/temporal.py", "src/semantic_transmission/decoder_runner.py",
    "src/semantic_transmission/research_quality.py", "src/semantic_transmission/official_quality.py",
    "src/semantic_transmission/webvid_ablation.py", "src/semantic_transmission/artifacts.py",
    "scripts/etri_decoder_probe.py", "04_semantic_decoder/scripts/mydemo_new_align_sh.py",
}))


def paths(name):
    if name not in NAMES:
        raise ValueError("unknown prepared video")
    selection = ROOT / name
    return selection, selection / "assistant_captions/captions_bundle.json", selection / "reconstruction_fp32"


def job_plan():
    jobs = []
    for stage, module, products in hybrid.job_plan(True):
        if stage == "reconstruct":
            break
        jobs.append((stage, MODULE if stage == "channel" else module, products))
    return jobs + [
        ("receiver-policy", MODULE, ["receiver_policy.json"]),
        ("prepare-text", MODULE, [text.REPORT]),
        ("reconstruct", MODULE, ["receiver/decoder_config.py", "receiver/reconstruction",
            "receiver/reference_trace.json", "receiver/tail_reference_trace.json", text.TRACE, noise.TRACE]),
        ("output-audit", MODULE, ["output_audit.json"]),
        ("evaluate", "semantic_transmission.research_quality",
            ["quality.json", "quality_delivered_mp4.csv", "quality_lossless_frames.csv"]),
    ]


def channel_source(base):
    """Resolve an explicitly recorded legacy receive replay, without modifying it."""
    files = ("transmitter/visual.c64", "received/visual.c64", "received/metadata.bin")
    if all((base / p).is_file() for p in files):
        return base
    receipt = read_json(base / "seed_protocol.json")
    source = Path(receipt["source"])
    if not receipt.get("channel_reused") or not all((source / p).is_file() for p in files):
        raise ValueError("legacy channel replay is incomplete")
    if sha256(base / "data/normalized.mp4") != sha256(source / "data/normalized.mp4"):
        raise ValueError("legacy channel replay source video differs")
    original = read_json(base / "receiver/decoder_inputs.json")
    reused = read_json(source / "receiver/decoder_inputs.json")
    if any(original[k] != reused[k] for k in ("video", "indices")):
        raise ValueError("legacy channel replay decoder inputs differ")
    if sha256(base / "receiver/metadata.csv") != sha256(source / "receiver/metadata.csv"):
        raise ValueError("legacy channel replay received metadata differs")
    return source


def preflight(name):
    selection_root, bundle_path, output = paths(name)
    protocol, selection = verify_selection(selection_root)
    bundle = captions.validate_bundle(bundle_path, selection_root)
    cfg = hybrid.build_config(protocol, selection_root, output, bundle_path)
    cfg.update(profile="short3_hybrid_faithful_v2_tail17_fp32", cpu_video_storage=True,
               semantic_clip_policy="frame_exact", concatenation_policy="endpoint_exact", steps=30)
    base = Path(protocol["baseline"])
    source = channel_source(base)
    files = [base / p for p in ("run_config.json", "data/normalized.mp4",
        "receiver/decoder_inputs.json", "receiver/reconstruction/sample_0000.mp4")]
    files += [source / p for p in ("transmitter/visual.c64", "received/visual.c64", "received/metadata.bin")]
    if source != base:
        files += [base / "seed_protocol.json", base / "receiver/metadata.csv", source / "receiver/metadata.csv",
                  source / "receiver/decoder_inputs.json", source / "data/normalized.mp4"]
    identity = dict(version=1, name=name, policy="hybrid_v2_tail17_fp32_scoped_noise",
        config=cfg, selection_freeze_sha256=sha256(selection_root / "selection_freeze.json"),
        caption_bundle_sha256=sha256(bundle_path), baseline_files={str(p):sha256(p) for p in files},
        code={p:sha256(REPO / p) for p in CODE})
    signature = fingerprint(identity)
    existing = output / "execution_protocol.json"
    if existing.exists() and read_json(existing) != dict(identity, signature=signature):
        raise ValueError(f"{name}: frozen input/code changed; preserve the existing run")
    if output.exists() and not existing.exists():
        raise ValueError(f"{name}: unrecognized output directory: {output}")
    local = settings(REPO)
    for field in ("python", "channel_python"):
        if not Path(local[field]).is_file():
            raise FileNotFoundError(local[field])
    for model in ("stdit", "vae", "vae2d", "t5"):
        if not Path(cfg["models"][model]).is_dir():
            raise FileNotFoundError(cfg["models"][model])
    for checkpoint in ("ntscc_hyperprior_quality_4_psnr.pth", "unimatch.pth"):
        if not (REPO / ".local/checkpoints" / checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise FileNotFoundError("FFmpeg/FFprobe required")
    if cfg["decoder_policy"] != "official_release" or cfg["fps"] != 24:
        raise ValueError("unsupported prepared decoder configuration")
    report = dict(video=name, status="READY_FOR_USER_EXECUTION", frames=cfg["frames"], fps=cfg["fps"],
        seconds=cfg["frames"]/cfg["fps"], keyframes=len(selection["indices"]), captions=len(bundle["records"]),
        t5="CPU FP32 cache", reference=tail.POLICY, steps=30, snr_db=cfg["snr_db"],
        output=str(output), selection_rerun=False, caption_model_run=False, reconstruction_started=False)
    return protocol, selection, cfg, identity, signature, report


def matched_channel(run):
    """Replay common keys from the verified legacy channel, seed new keys by frame."""
    import numpy as np
    from .metadata_channel import transmit
    from .transmission_accounting import channel_breakdown
    from .wire import unpack
    cfg = read_json(run / "run_config.json")
    protocol, _ = verify_selection(Path(cfg["hybrid_selection_root"]))
    base = channel_source(Path(protocol["baseline"]))
    packet = (run / "transmitter/metadata.bin").read_bytes()
    restored, report = transmit(packet, cfg["snr_db"], cfg["channel_seed"])
    if restored != packet or report["bit_errors"]:
        raise ValueError("metadata channel failed")
    header, payload = unpack(restored)
    old_header, old_payload = unpack((base / "received/metadata.bin").read_bytes())
    sent = np.fromfile(run / "transmitter/visual.c64", dtype="<c8")
    old_sent = np.fromfile(base / "transmitter/visual.c64", dtype="<c8")
    old_received = np.fromfile(base / "received/visual.c64", dtype="<c8")
    old_items = {item["index"]: item for item in old_header["keyframes"]}
    if len(sent) != sum(k["complex_count"] for k in header["keyframes"]) or not np.isfinite(sent).all():
        raise ValueError("invalid encoded visual stream")
    if len(old_sent) != len(old_received) or not np.isfinite(old_received).all():
        raise ValueError("invalid legacy received stream")
    values = sent.copy()
    common, fresh = [], []
    sigma = math.sqrt(1/(2*10**(cfg["snr_db"]/10)))
    for item in header["keyframes"]:
        i, a, n = item["index"], item["complex_offset"], item["complex_count"]
        if i in old_items:
            old = old_items[i]
            b = old["complex_offset"]
            if n != old["complex_count"] or not np.array_equal(sent[a:a+n], old_sent[b:b+n]):
                raise ValueError(f"common encoded keyframe changed: {i}")
            x, y = item["rate_offset"], old["rate_offset"]
            if item["rate_bytes"] != old["rate_bytes"] or payload[x:x+item["rate_bytes"]] != old_payload[y:y+old["rate_bytes"]]:
                raise ValueError(f"common rate indices changed: {i}")
            values[a:a+n] = old_received[b:b+n]
            common.append(i)
        else:
            rng = np.random.default_rng(cfg["channel_seed"] + 100000 + i)
            values[a:a+n] += (rng.normal(0,sigma,n) + 1j*rng.normal(0,sigma,n)).astype("<c8")
            fresh.append(i)
    received = run / "received"
    received.mkdir()
    (received / "metadata.bin").write_bytes(restored)
    values.tofile(received / "visual.c64")
    digital = report["complex_channel_uses"]
    report.update(status="PASSED", metadata_exact_match=True, common_frames_exact=common,
        new_noise_frames=fresh, visual_complex_channel_uses=len(sent), digital_complex_channel_uses=digital,
        total_complex_channel_uses=len(sent)+digital,
        cbr_complex_uses_per_source_scalar=(len(sent)+digital)/(3*cfg["width"]*cfg["height"]*cfg["frames"]),
        complete_sample_dependent_model_input_accounting=True, physical_link_overhead_included=False,
        visual_awgn_rng="Exact baseline receive for common keys; PCG64(channel_seed+100000+source_frame) for others",
        channel_seed=cfg["channel_seed"], visual_tx_mean_power=float(np.mean(np.abs(sent)**2)),
        baseline_channel_source=str(base), transmission_breakdown=channel_breakdown(packet,len(sent),header["video"],report),
        received_files={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in received.iterdir()})
    write_json(run / "channel_accounting.json", report)


def receiver_policy(run):
    cfg = read_json(run / "run_config.json")
    received = read_json(run / "receiver/decoder_inputs.json")
    code = read_json(run.parent / "execution_protocol.json")["code"]
    contract = dict(policy=noise.POLICY, seed=received["decoder"]["seed"],
        segments=len(received["indices"])-1, steps=received["decoder"]["steps"],
        inputs=fingerprint(snapshot(run, tail.INPUTS)), code=fingerprint(code), reference_policy=tail.POLICY)
    write_json(run / "receiver_policy.json", dict(policy=tail.POLICY, reference_frames=17,
        reference_latents=5, alignment=5, text_encoder_policy=text.POLICY, t5_precision="fp32",
        noise_contract=contract, noise_reference=None, source=cfg["hybrid_selection_root"]))


def audit(run):
    precision.audit(run)
    result = read_json(run / "output_audit.json")
    cfg = read_json(run / "run_config.json")
    result.update(full_input_reconstructed=True, duration_seconds=cfg["frames"]/cfg["fps"],
                  is_60_second_experiment=False)
    write_json(run / "output_audit.json", result)


def baseline_alignment(base):
    inputs = read_json(base / "receiver/decoder_inputs.json")
    policy = inputs["decoder"].get("concatenation_policy", inputs["decoder"]["policy"])
    mapping = output_source_indices(inputs["indices"], policy)
    kept = unique_output_positions(inputs["indices"], policy)
    frames = read_json(base / "run_config.json")["frames"]
    if [mapping[i] for i in kept] != list(range(frames)):
        raise ValueError("baseline source-frame correspondence incomplete")
    return mapping, kept


def make_comparison(output):
    run = output / "run"
    cfg = read_json(run / "run_config.json")
    selection_root = Path(cfg["hybrid_selection_root"])
    protocol, selection = verify_selection(selection_root)
    base = Path(protocol["baseline"])
    q = read_json(run / "quality.json")
    output_audit = read_json(run / "output_audit.json")
    video = run / "receiver/reconstruction/sample_0000.mp4"
    if q["status"] != "PASSED" or not output_audit["full_input_reconstructed"]:
        raise ValueError("reconstruction/evaluation incomplete")
    if q["video_sha256"] != sha256(video) or q["source_sha256"] != protocol["input_sha256"]:
        raise ValueError("quality does not belong to this source/video")
    mapping, kept = baseline_alignment(base)
    # The old decoder repeats segment boundaries. Remove those repeats in the
    # viewing copy only; never stretch the original or modify frozen outputs.
    discarded = sorted(set(range(len(mapping))) - set(kept))
    expression = '+'.join(f'eq(n\\,{i})' for i in discarded)
    vf = (f"select=not({expression})," if expression else "") + f"setpts=N/({cfg['fps']}*TB)"
    old = output / "baseline_aligned.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(base / "receiver/reconstruction/sample_0000.mp4"),
        "-vf", vf, "-an", "-r", str(cfg["fps"]), "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", str(old)], check=True)
    hybrid.pts_audit(old, cfg["frames"], cfg["fps"])
    target = output / "comparison.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin",
        *[a for p in (run / "data/normalized.mp4", old, video) for a in ("-i", str(p))],
        "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]", "-map", "[v]", "-an",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p", str(target)], check=True)
    hybrid.pts_audit(target, cfg["frames"], cfg["fps"])
    result = dict(status="PASS_SHORT_VIDEO_RECONSTRUCTION_REVIEW_PENDING", name=protocol["source_id"],
        frames=cfg["frames"], fps=cfg["fps"], duration_seconds=cfg["frames"]/cfg["fps"],
        keyframes=len(selection["indices"]), captions=len(selection["indices"])-1,
        video=str(video), video_sha256=sha256(video), quality=q["delivered_mp4"],
        channel=read_json(run / "channel_accounting.json"),
        t5_precision="fp32", reference_policy=tail.POLICY, noise_policy=noise.POLICY,
        baseline_repeated_boundary_frames_removed=discarded, baseline_view_is_reencoded=True,
        matched_legacy_generation_noise=False, hallucination_review="PENDING", hallucination_mitigation_verified=False)
    write_json(output / "RESULT.json", result)
    rows = ''.join(f'<tr><td>{k}</td><td>{q["delivered_mp4"][k]:.4f}</td></tr>' for k in METRICS)
    (output / "review.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<title>최신 방식 복원 결과</title><style>body{font:16px system-ui;margin:24px}video{width:100%}td{padding:8px}</style>'
        f'<h1>{html.escape(protocol["source_id"])} · {cfg["frames"]/cfg["fps"]:.2f}초</h1>'
        '<p>왼쪽 원본 / 가운데 기존 LGVSC / 오른쪽 혼합 키프레임·캡션 v2·마지막 17프레임 참조·T5 FP32.</p>'
        '<video controls preload="metadata" src="comparison.mp4"></video>'
        '<p>기존 영상의 중복 경계 프레임을 제거해 같은 원본 시각으로 비교합니다. 비교용 영상은 재인코딩했으며, 아래 지표는 이번 원래 출력 MP4 기준입니다.</p>'
        '<table><tr><th>이번 복원의 지표</th><th>값</th></tr>'+rows+'</table>'
        '<p>과거 결과와 생성 잡음은 같지 않습니다. 화질·할루시네이션 개선 여부는 별도 검토가 필요합니다.</p>'
        '<a href="RESULT.json">결과 기록</a> · <a href="run/receiver/reconstruction/sample_0000.mp4">복원 영상</a></html>')


def execute(name):
    protocol, selection, cfg, identity, signature, _ = preflight(name)
    selection_root, _, output = paths(name)
    output.mkdir(parents=True, exist_ok=True)
    with hybrid.lock(REPO / ".local/etri_60s_check.lock"), hybrid.lock(output / ".execution.lock"):
        write_json(output / "execution_protocol.json", dict(identity, signature=signature))
        stages = Stages(output, signature)
        stages.step("initialize", ["run"], hybrid.initialization_products(cfg["frames"]),
            lambda _: hybrid.initialize(output, protocol, selection, cfg))
        local = settings(REPO)
        env = dict(environment(cfg["seed"]), PYTHONPATH=str(REPO / "src"), ETRI_RUN_SIGNATURE=signature)
        for stage, module, products in job_plan():
            python = local["channel_python"] if stage == "channel" else local["python"]
            command = [python, "-m", module]
            if module in (MODULE, hybrid.MODULE, hybrid.ETRI):
                command += ["--worker", stage, "--run-dir", str(output / "run")]
            else:
                command += [stage, str(output / "run")]
            child_env = dict(env, CUDA_VISIBLE_DEVICES="-1") if stage == "channel" else env
            resource = f"resources/{stage}.json"
            products = [f"run/{p}" for p in products] + [resource]
            stages.step(stage, products, products,
                lambda log, cmd=command, e=child_env, res=resource: hybrid.launch(cmd, log, e, output / res))
        products = ["baseline_aligned.mp4", "comparison.mp4", "RESULT.json", "review.html"]
        stages.step("comparison", products, products, lambda _: make_comparison(output))
        print(f"COMPLETE SHORT VIDEO: {output / 'review.html'}", flush=True)


def console_run(names):
    from .reconstruction_console import Progress, GrowingLog, GpuUsage, stop
    # Validate every input before starting the first expensive worker.
    reports = {name:preflight(name)[-1] for name in names}
    total = sum(reports[name]["captions"] for name in names)
    completed = 0
    monitor = GpuUsage()
    logroot = REPO / ".local/reconstruction_console"
    logroot.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="short3-", dir=logroot))
    for index,name in enumerate(names):
        _, _, output = paths(name)
        progress = Progress(reports[name]["captions"], 30)
        log, progress_path = folder / f"{name}.log", folder / f"{name}.progress.json"
        rootlog = GrowingLog(log)
        decoderlog = GrowingLog(output / "run/receiver/reconstruction/00000.log")
        command = [settings(REPO)["python"], "-m", MODULE, "--execute-one", name]
        env = dict(os.environ, QUALITY_PROGRESS_FILE=str(progress_path))
        process, code = None, 1
        try:
            with log.open("x") as stream:
                process = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
                while True:
                    for line in rootlog.read():
                        progress.stage_line(line)
                    if progress.active:
                        for line in decoderlog.read():
                            progress.decoder_line(line)
                        try:
                            progress.completed_segments(read_json(progress_path))
                        except (FileNotFoundError, json.JSONDecodeError):
                            pass
                    fraction = (completed + reports[name]["captions"] * progress.percent/100) / total
                    gpu = monitor.read()
                    usage = f"{gpu:3d}%" if gpu is not None else " --%"
                    sys.stdout.write(f"\r복원 진행률: {min(99.99,fraction*100):6.2f}% | GPU 사용률: {usage} | {index+1}/{len(names)} {name:<16}")
                    sys.stdout.flush()
                    try:
                        code = process.wait(timeout=0.5)
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
            raise
        if code:
            print(f"\n복원 {'중단' if code==130 else '실패'} · 상세 로그: {log}")
            return code
        completed += reports[name]["captions"]
    print(f"\r복원 진행률: 100.00% | 완료: {len(names)}개 영상                                      ")
    for name in names:
        print(f"{name}: {paths(name)[2] / 'review.html'}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", choices=NAMES, help="omit to run all three in order")
    p.add_argument("--check", action="store_true", help="read-only readiness check, no model execution")
    p.add_argument("--execute-one", choices=NAMES, help=argparse.SUPPRESS)
    p.add_argument("--worker", choices=("channel", "receiver-policy", "prepare-text", "reconstruct", "output-audit"), help=argparse.SUPPRESS)
    p.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.worker:
        if args.run_dir is None or args.check or args.execute_one or args.video:
            p.error("worker requires only --run-dir")
        functions = {"channel":matched_channel, "receiver-policy":receiver_policy, "prepare-text":precision.prepare,
                     "reconstruct":precision.reconstruct, "output-audit":audit}
        return functions[args.worker](args.run_dir.resolve())
    if args.run_dir is not None:
        p.error("--run-dir requires --worker")
    if args.execute_one:
        if args.check or args.video:
            p.error("--execute-one is an internal execution option")
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
        return execute(args.execute_one)
    names = (args.video,) if args.video else NAMES
    if args.check:
        print(json.dumps([preflight(name)[-1] for name in names], ensure_ascii=False, indent=2))
        return 0
    return console_run(names)


if __name__ == "__main__":
    raise SystemExit(main())
