"""One-command reconstruction from a completed, frozen hybrid selection.

--check is read-only and starts no worker/model. Default execution starts with
captions/flow preparation, never with SKEM. Successful stages are hash-reused;
partial failed stages are archived before restarting them.
"""
import argparse
import csv
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from .artifacts import sha256, write_json
from .cli import repository, settings
from .etri_60s_check import lock, pts_audit
from .hybrid_selection import DEFAULT_ROOT, verify_selection
from .webvid5 import fingerprint, read_json
from .webvid_ablation import Stages, environment

REPO = repository()
MODULE = "semantic_transmission.hybrid_reconstruction"
ETRI = "semantic_transmission.etri_60s"


def job_plan(caption_bundle=None):
    return [
        ("input-audit", ETRI, ["input_audit.json"]),
        ("semantic-clips", ETRI, ["semantic_clips_audit.json", "data/clips"]),
        ("caption", MODULE if caption_bundle else "semantic_transmission.workers",
         ["captions.json", "caption_sampling.json"] + (["caption_provenance.json"] if caption_bundle else [])),
        ("flow", "semantic_transmission.workers", ["metadata_tx.json", "flow_sampling.json"]),
        ("send", "semantic_transmission.codec_transport", ["transmitter", "sender_accounting.json"]),
        ("channel", MODULE, ["received", "channel_accounting.json"]),
        ("receive", "semantic_transmission.codec_transport", ["receiver/frames", "receiver/metadata.csv", "receiver/decoder_inputs.json", "receiver_accounting.json"]),
        ("reconstruct", ETRI, ["receiver/decoder_config.py", "receiver/reconstruction", "receiver/reference_trace.json"]),
        ("output-audit", MODULE, ["output_audit.json"]),
        ("evaluate", "semantic_transmission.research_quality", ["quality.json", "quality_delivered_mp4.csv", "quality_lossless_frames.csv"]),
        ("comparison", MODULE, [])]


def build_config(protocol, selection_root, output, caption_bundle=None):
    cfg = dict(protocol["config"])
    cfg.pop("selector_checkpoint", None)
    cfg.update(profile="etri_hybrid_selection_v1", selector="ai_skem_hybrid",
        visual_channel="baseline_common_replay_new_frame_awgn",
        caption_checkpoint=str(output / "checkpoints/caption.json"),
        hybrid_selection_root=str(selection_root))
    if caption_bundle:
        cfg.pop("caption_checkpoint", None)
        cfg.update(caption_provider="assistant_provided", caption_bundle=str(caption_bundle),
                   profile="etri_hybrid_assistant_captions_v1")
    return cfg


def initialization_products(frames):
    # Only immutable initialization products: later data/clips additions must
    # not make a completed initialization receipt fail when resuming.
    return ["run/run_config.json", "run/keyframes.json", "run/data/normalized.mp4",
            "run/data/frames/sample/frames.csv",
            *[f"run/data/frames/sample/{i}.png" for i in range(frames)]]


def initialize(output, protocol, selection, cfg):
    run = output / "run"
    frames = run / "data/frames/sample"
    frames.mkdir(parents=True)
    for i in range(protocol["frames"]):
        (frames / f"{i}.png").symlink_to(Path(protocol["source_frames"]) / f"{i}.png")
    with (frames / "frames.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=["frame_path"])
        writer.writeheader()
        writer.writerows({"frame_path": str(frames / f"{i}.png")} for i in range(protocol["frames"]))
    (run / "data/normalized.mp4").symlink_to(protocol["normalized_video"])
    write_json(run / "run_config.json", cfg)
    write_json(run / "keyframes.json", {"indices": selection["indices"], "selector": selection["selector"]})


def check_ready(selection_root, output, caption_bundle=None):
    protocol, selection = verify_selection(selection_root)
    if caption_bundle:
        from .assisted_captions import validate_bundle
        validate_bundle(caption_bundle,selection_root)
    base = Path(protocol["baseline"])
    protected = [base, Path(protocol["source_frames"]), Path(protocol["input"]).parent,
                 *[selection_root / name for name in ("selected_frames", "scores", "logs", "assistant_captions")]]
    if (output == selection_root or output in selection_root.parents
            or any(output == p or output in p.parents or p in output.parents for p in protected)):
        raise ValueError("reconstruction output must not replace selection/source directories")
    local = settings(REPO)
    for field in ("python", "channel_python"):
        if not Path(local[field]).is_file():
            raise FileNotFoundError(local[field])
    cfg = build_config(protocol, selection_root, output, caption_bundle)
    for model in ("stdit", "vae", "t5", "vae2d", *([] if caption_bundle else ["pllava"])):
        if not Path(cfg["models"][model]).is_dir():
            raise FileNotFoundError(cfg["models"][model])
    for name in ("ntscc_hyperprior_quality_4_psnr.pth", "unimatch.pth"):
        if not (REPO / ".local/checkpoints" / name).is_file():
            raise FileNotFoundError(name)
    if not shutil_which_ffmpeg():
        raise FileNotFoundError("ffmpeg/ffprobe")
    code_paths = [Path(__file__), REPO / "scripts/reconstruct_hybrid.sh",
        REPO / "src/semantic_transmission/etri_60s.py", REPO / "src/semantic_transmission/workers.py",
        REPO / "src/semantic_transmission/codec_transport.py", REPO / "src/semantic_transmission/research_quality.py",
        REPO / "src/semantic_transmission/official_quality.py", REPO / "src/semantic_transmission/temporal.py",
        REPO / "src/semantic_transmission/decoder_runner.py", REPO / "src/semantic_transmission/webvid_ablation.py",
        REPO / "src/semantic_transmission/semantic_clips.py", REPO / "src/semantic_transmission/caption_checkpoint.py",
        REPO / "src/semantic_transmission/metadata_channel.py", REPO / "src/semantic_transmission/wire.py",
        REPO / "src/semantic_transmission/transmission_accounting.py",
        REPO / "scripts/etri_decoder_probe.py", REPO / "04_semantic_decoder/scripts/mydemo_new_align_sh.py"]
    if caption_bundle:
        code_paths.append(REPO / "src/semantic_transmission/assisted_captions.py")
    baseline_files = [base / p for p in ("run_config.json", "transmitter/visual.c64",
        "received/visual.c64", "received/metadata.bin", "receiver/reconstruction/sample_0000.mp4", "quality.json")]
    identity = dict(selection_freeze_sha256=sha256(selection_root / "selection_freeze.json"),
        config=cfg, code={str(p.relative_to(REPO)): sha256(p) for p in code_paths},
        baseline_files={str(p): sha256(p) for p in baseline_files})
    if caption_bundle:
        identity["caption_bundle_sha256"] = sha256(caption_bundle)
    signature = fingerprint(identity)
    receipt = output / "execution_protocol.json"
    if receipt.exists() and read_json(receipt)["signature"] != signature:
        raise ValueError("reconstruction inputs/code changed; preserve this run and choose a new --output")
    if not receipt.exists() and (output / "run").exists():
        raise ValueError("unrecognized existing run directory; choose a new --output")
    report = dict(status="READY_FOR_USER_EXECUTION", keyframes=len(selection["indices"]),
        frames=protocol["frames"], fps=protocol["fps"], duration_seconds=protocol["frames"]/protocol["fps"],
        stages=["initialize", *[s for s,_,_ in job_plan(caption_bundle)]],
        caption_provider="assistant_provided" if caption_bundle else "pllava",
        runs_pllava=not bool(caption_bundle),
        reruns_keyframe_selection=False, output=str(output),
        expected_video=str(output / "run/receiver/reconstruction/sample_0000.mp4"),
        check_scope="Read-only input/code/model-path checks; reconstruction has not been launched or validated.")
    return protocol, selection, cfg, identity, signature, report


def shutil_which_ffmpeg():
    import shutil
    return shutil.which("ffmpeg") and shutil.which("ffprobe")


def launch(command, log, env, resource):
    start = time.monotonic()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as stream:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait()  # completion driven; no progress polling
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            raise
        finally:
            write_json(resource, dict(command=command, seconds=time.monotonic()-start,
                                      returncode=process.returncode))
    if code:
        raise RuntimeError(f"stage failed ({code}); log: {log}. Re-run the same command after fixing the cause.")


def execute(selection_root, output, caption_bundle=None):
    protocol, selection, cfg, identity, signature, _ = check_ready(selection_root, output, caption_bundle)
    output.mkdir(parents=True, exist_ok=True)
    with lock(REPO / ".local/etri_60s_check.lock"), lock(output / ".execution.lock"):
        receipt = output / "execution_protocol.json"
        if not receipt.exists():
            write_json(receipt, dict(identity, signature=signature))
        stages = Stages(output, signature)
        stages.step("initialize", ["run"], initialization_products(protocol["frames"]),
                    lambda _: initialize(output, protocol, selection, cfg))
        local = settings(REPO)
        run = output / "run"
        env = environment(cfg["seed"])
        env["PYTHONPATH"] = str(REPO / "src")
        env["ETRI_RUN_SIGNATURE"] = signature
        for stage, module, products in job_plan(caption_bundle):
            python = local["channel_python"] if stage == "channel" else local["python"]
            command = [python, "-m", module]
            if module in (ETRI, MODULE):
                command += ["--worker", stage, "--run-dir", str(run)]
            else:
                command += [stage, str(run)]
            worker_env = dict(env, CUDA_VISIBLE_DEVICES="-1") if stage == "channel" else env
            artifacts = [f"run/{p}" for p in products] if stage != "comparison" else ["comparison.mp4"]
            owned = [*artifacts, f"resources/{stage}.json"]
            required = list(owned)
            if stage == "caption" and not caption_bundle:
                required.append("checkpoints/caption.json")
            stages.step(stage, owned, required,
                lambda log, cmd=command, e=worker_env, s=stage: launch(cmd, log, e, output / f"resources/{s}.json"))
        finalize(output, selection_root, protocol, selection, stages)


def matched_channel(run):
    import numpy as np
    from .metadata_channel import transmit
    from .transmission_accounting import channel_breakdown
    from .wire import unpack
    cfg = read_json(run / "run_config.json")
    protocol, _ = verify_selection(Path(cfg["hybrid_selection_root"]))
    base = Path(protocol["baseline"])
    packet = (run / "transmitter/metadata.bin").read_bytes()
    restored, report = transmit(packet, cfg["snr_db"], cfg["channel_seed"])
    if restored != packet or report["bit_errors"]:
        raise ValueError("metadata channel failed; no reconstruction inputs produced")
    header, payload = unpack(restored)
    old_header, old_payload = unpack((base / "received/metadata.bin").read_bytes())
    sent = np.fromfile(run / "transmitter/visual.c64", dtype="<c8")
    old_sent = np.fromfile(base / "transmitter/visual.c64", dtype="<c8")
    old_received = np.fromfile(base / "received/visual.c64", dtype="<c8")
    old_items = {item["index"]: item for item in old_header["keyframes"]}
    if len(sent) != sum(k["complex_count"] for k in header["keyframes"]) or not np.isfinite(sent).all():
        raise ValueError("invalid encoded visual stream")
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
        transmission_breakdown=channel_breakdown(packet, len(sent), header["video"], report),
        received_files={p.name:dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in received.iterdir()})
    write_json(run / "channel_accounting.json", report)


def comparison(run):
    cfg = read_json(run / "run_config.json")
    protocol = read_json(Path(cfg["hybrid_selection_root"]) / "protocol.json")
    video = run / "receiver/reconstruction/sample_0000.mp4"
    base = Path(protocol["baseline"]) / "receiver/reconstruction/sample_0000.mp4"
    output = run.parent / "comparison.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(run / "data/normalized.mp4"),
        "-i", str(base), "-i", str(video), "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]",
        "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p", str(output)], check=True)
    pts_audit(output, cfg["frames"], cfg["fps"])


def hybrid_output_audit(run):
    from .etri_60s import output_audit
    output_audit(run)
    cfg = read_json(run / "run_config.json")
    _, selection = verify_selection(Path(cfg["hybrid_selection_root"]))
    audit = read_json(run / "output_audit.json")
    anchors = [r["frame"] for r in selection["records"] if "max_gap" in r["reason"]]
    audit.update(automatic_keyframe_insertion=bool(anchors),
        automatic_keyframe_insertion_stage="offline_hybrid_selection",
        time_anchor_frames=anchors, decoder_inserted_keyframes=False,
        selection_sha256=sha256(Path(cfg["hybrid_selection_root"]) / "selection.json"))
    write_json(run / "output_audit.json", audit)


def finalize(output, selection_root, protocol, selection, stages):
    run = output / "run"
    audit = read_json(run / "output_audit.json")
    quality = read_json(run / "quality.json")
    if not audit["full_60s_reconstructed"] or quality["status"] != "PASSED":
        raise ValueError("full reconstruction/evaluation incomplete")
    if quality["video_sha256"] != audit["video_sha256"] or sha256(audit["video"]) != quality["video_sha256"]:
        raise ValueError("reconstructed/evaluated video changed")
    verify_selection(selection_root)
    pts_audit(output / "comparison.mp4", protocol["frames"], protocol["fps"])
    metrics = quality["delivered_mp4"]
    baseline = read_json(Path(protocol["baseline"]) / "quality.json")["delivered_mp4"]
    result = dict(status="PASS_60S_HYBRID_RECONSTRUCTION", video=audit["video"],
        comparison=str(output / "comparison.mp4"), selection_root=str(selection_root),
        selection_sha256=sha256(selection_root / "selection.json"), keyframes=len(selection["indices"]),
        frames=1440, fps=24, duration_seconds=60, quality=metrics, baseline_quality=baseline,
        channel=read_json(run / "channel_accounting.json"),
        stage_seconds={name:r["seconds"] for name,r in stages.completed.items()},
        hallucination_review="PENDING", hallucination_mitigation_verified=False,
        comparison_limit="One development source/seed; only common received keys are exact. Different segments change captions and diffusion noise.")
    cfg = read_json(run / "run_config.json")
    result["caption_provider"] = cfg.get("caption_provider", "pllava")
    if cfg.get("caption_bundle"):
        result["caption_provenance"] = read_json(run / "caption_provenance.json")
    write_json(output / "RESULT.json", result)
    rows = ''.join(f'<tr><td>{name}</td><td>{baseline[name]:.4f}</td><td>{metrics[name]:.4f}</td></tr>'
                   for name in ("psnr_db", "ssim", "lpips_vgg", "clip", "dists"))
    (output / "review.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<title>혼합 선택 복원</title><style>body{font:16px system-ui;max-width:1740px;margin:24px auto}video{width:100%}td,th{padding:8px}</style>'
        f'<h1>혼합 키프레임 {len(selection["indices"])}개 · 60초 복원 완료</h1>'
        f'<p>왼쪽 원본 / 가운데 기존 SKEM·PLLaVA / 오른쪽 혼합 선택·{result["caption_provider"]}. 의미 오류 검수는 아직 미완료입니다.</p>'
        '<video controls preload="metadata" src="comparison.mp4"></video>'
        '<table><tr><th>전달 MP4 전체 평균</th><th>기존 SKEM</th><th>혼합 선택</th></tr>'+rows+'</table>'
        '<p>PSNR·SSIM·CLIP은 높을수록, LPIPS·DISTS는 낮을수록 양호. 화질 수치만으로 할루시네이션 완화를 판정하지 않습니다.</p>'
        '<p><a href="RESULT.json">전체 결과·전송량</a> · <a href="run/receiver/reconstruction/sample_0000.mp4">복원 영상</a></p></html>')
    print(f"COMPLETE: {output / 'review.html'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--captions", type=Path, nargs="?", const="__prepared_assistant__",
                        help="use a frozen assistant caption bundle; omit path to use the prepared tv_low_08 bundle")
    parser.add_argument("--check", action="store_true", help="read-only readiness check; no reconstruction")
    parser.add_argument("--worker", choices=["channel", "comparison", "output-audit", "caption"], help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.check or args.run_dir is None:
            parser.error("worker requires --run-dir and cannot be combined with --check")
        from .assisted_captions import import_captions
        return {"channel": matched_channel, "comparison": comparison,
                "output-audit": hybrid_output_audit, "caption": import_captions}[args.worker](args.run_dir)
    root = args.selection_root.resolve()
    captions = (root / "assistant_captions/captions_bundle.json" if str(args.captions) == "__prepared_assistant__"
                else args.captions.resolve() if args.captions else None)
    output = args.output.resolve() if args.output else root / ("reconstruction_assistant_captions" if captions else "reconstruction")
    if args.check:
        import json
        print(json.dumps(check_ready(root, output, captions)[-1], ensure_ascii=False, indent=2))
        return
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
    execute(root, output, captions)


if __name__ == "__main__":
    main()
