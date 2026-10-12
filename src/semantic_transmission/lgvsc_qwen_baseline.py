"""Prepare one 16 s input, run released LGVSC, then variable-count Qwen semantics.

This is a matched local baseline, not a claim to reproduce a published GIF.
Sampling, captioning and receiver policies are recorded separately from the
shared local normalization/NTSCC transport and memory-placement adaptations.
"""
import argparse
import html
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from . import qwen35_pairs_video as current
from .artifacts import sha256, write_json
from .cli import settings
from .reconstruction_console import GrowingLog, Progress
from .temporal import output_source_indices, segment_lengths
from .validation_progress import emit
from .video_io import probe
from .webvid_ablation import Stages, environment

REPO = current.REPO
MODULE = "semantic_transmission.lgvsc_qwen_baseline"
POLICY = "zea_16s_released_lgvsc_then_semantic_qwen_v2"
DEFAULT_INPUT = REPO / "assets/sample/stock-footage-aerial-drone-panoramic-video-of-famous-port-and-marina-of-zea-with-luxury-yachts-docked-as-seen.mp4"
DEFAULT_OUTPUT = REPO / "outputs/lgvsc_qwen_semantic_zea_16s_20261009"
FRAMES, FPS, SECONDS = 384, 24, 16
SIZE = (576, 320)


def read(path):
    return current.ae.read(path)


def candidate_indices(frames=FRAMES):
    # Released get_frame.py: int(0.2 * fps), without an extra EOF candidate.
    return list(range(0, frames, int(.2 * FPS)))


def baseline_config(template, common, root):
    cfg = read(REPO / "configs/etri_official.json")
    cfg.pop("source_video", None)
    shared = template["receiver"]
    cfg.update(profile=POLICY + ":lgvsc", input=str(common / "normalized.mp4"),
        frames=FRAMES, max_frames=FRAMES, width=SIZE[0], height=SIZE[1], fps=FPS,
        models=shared["models"], preserve_input=True, official_preprocessing=False,
        selection_stride=4, semantic_clip_policy="official_release",
        conditioning_alignment="official_release", concatenation_policy="official_release",
        cpu_video_storage=True, reuse_diffusion_model=False,
        visual_channel="source_frame_seeded_numpy_awgn_v1",
        selector_checkpoint=str(root / "lgvsc/checkpoints/selector.json"),
        caption_checkpoint=str(root / "lgvsc/checkpoints/caption.json"))
    for key in ("seed", "channel_seed", "snr_db", "steps", "flash_attn"):
        cfg[key] = shared[key]
    # BF16 weights remain unchanged; use the already validated 16 GiB placement.
    for key in ("internvl_offload_layers", "internvl_compact_kv_cache"):
        if key in shared:
            cfg[key] = shared[key]
    return cfg


def preflight(source, root):
    current.pc.forbid_overlap(root, [source, current.pc.ROOT, current.pc.SNAPSHOT,
                                   REPO / ".local", REPO / "src", REPO / "assets",
                                   REPO / "outputs/qwen35_evidence_repair_v3"])
    info = probe(source)
    if info["duration"] < SECONDS:
        raise ValueError("this matched experiment requires at least 16 seconds of source")
    template = current.pc.load_protocol(current.pc.ROOT)
    local = settings(REPO)
    for name in ("ffmpeg", "ffprobe"):
        if not shutil.which(name):
            raise FileNotFoundError(name)
    for path in [*local.values(), REPO / ".local/qwen35_env/bin/python"]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    cfg = baseline_config(template, root / "common", root)
    for name, folder in cfg["models"].items():
        config = Path(folder) / ("vae/config.json" if name == "vae2d" else "config.json")
        if not config.is_file():
            raise FileNotFoundError(config)
    for name in ("ntscc_hyperprior_quality_4_psnr.pth", "unimatch.pth"):
        if not (REPO / ".local/checkpoints" / name).is_file():
            raise FileNotFoundError(name)
    code = dict(template["code"])
    for relative in ("src/semantic_transmission/lgvsc_qwen_baseline.py",
                     "src/semantic_transmission/qwen35_pairs_video.py",
                     "src/semantic_transmission/qwen35_state_delta.py",
                     "src/semantic_transmission/qwen35_event_pairs.py",
                     "src/semantic_transmission/qwen35_semantic_selection.py",
                     "src/semantic_transmission/reconstruction_console.py",
                     "scripts/run_lgvsc_qwen_baseline.sh", "scripts/etri_decoder_probe.py",
                     "02_semantic_encoder/skem/MLM-keyframe-internvl.py",
                     "configs/etri_official.json", "configs/official_opensora.py"):
        code[relative] = sha256(REPO / relative)
    # Pin vendor Python and all model file metadata for resume, without loading a model.
    for vendor in ("Open-Sora", "InternVL", "PLLaVA", "NTSCC_JSAC22"):
        folder = REPO / ".local/vendor" / vendor
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in sorted(folder.rglob("*.py")):
            code[str(path.relative_to(REPO))] = sha256(path)
    model_files = {}
    for folder in [*(Path(p) for p in cfg["models"].values()), REPO / ".local/checkpoints"]:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                st = path.stat()
                model_files[str(path)] = dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns,
                                              resolved=str(path.resolve()))
    identity = dict(policy=POLICY, source=str(source), source_sha256=sha256(source),
        source_info=info, duration_seconds=SECONDS, frames=FRAMES, fps=FPS, size=SIZE,
        normalization="first 16s; scale=576:320,fps=24; exactly 384 frames; x264 CRF18 medium yuv420p",
        template=str(current.TEMPLATE), template_sha256=sha256(current.TEMPLATE),
        baseline=cfg, qwen_policy="qwen35_semantic_video_v2",
        code=code, model_file_metadata=model_files, interpreters=local,
        scope="Released selector/caption/flow/receiver policies through local runtime adapters. "
              "Shared normalized input and frame-seeded NTSCC/AWGN transport. Not bit-exact paper/GIF reproduction.",
        noise_scope="Same SNR, channel seed, generation seed and steps. Shared keys get identical "
                    "channel noise; different segment shapes and decoder RNG policies are not identical diffusion tensors.")
    # Normalize tuples before comparison against a JSON receipt.
    identity = json.loads(json.dumps(identity))
    identity["signature"] = current.ae.digest(identity)
    receipt = root / "protocol.json"
    if receipt.exists():
        if read(receipt) != identity:
            raise ValueError("input/code/settings changed; preserve this output and choose a new --output")
    elif root.exists() and any(root.iterdir()):
        raise ValueError("unrecognized nonempty output; choose a new --output")
    return identity


def progress_message(label, elapsed, record=None):
    if record and isinstance(record.get("total"), (int, float)) and record["total"] > 0:
        done, total = record.get("done"), record["total"]
        if isinstance(done, (int, float)) and 0 <= done <= total:
            return f"{label}: {done:g}/{total:g} ({100*done/total:.1f}%) | 경과 {elapsed/60:.1f}분"
    return f"{label}: 초기화/처리 중 (진행량 미확정) | 경과 {elapsed/60:.1f}분"


def launch(command, log, env, resource, *, label=None, decoder_run=None, interval=10):
    """Read progress/log bytes automatically; no GPU polling or estimated percentages."""
    label = label or log.stem
    ipc = resource.with_suffix(".progress.json")
    ipc.unlink(missing_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(env, QUALITY_PROGRESS_FILE=str(ipc), PYTHONUNBUFFERED="1")
    start = time.monotonic()
    print(progress_message(label, 0), flush=True)
    reader = GrowingLog(log)
    decoder_reader = meter = None
    if decoder_run is not None:
        keys = read(decoder_run / "keyframes.json")["indices"]
        meter = Progress(len(keys)-1, read(decoder_run / "run_config.json")["steps"])
        meter.active = True
        decoder_reader = GrowingLog(decoder_run / "receiver/reconstruction/00000.log")
    with log.open("x") as stream:
        process = subprocess.Popen(list(map(str, command)), cwd=REPO, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        next_display = 0
        try:
            while True:
                try:
                    code = process.wait(timeout=min(interval, 1))
                except subprocess.TimeoutExpired:
                    code = None
                for line in reader.read(final=code is not None):
                    if (line.startswith(("START ", "DONE ", "REUSE ", "수신·복원 단계:", "전체 작업",
                                         "1/3 ", "2/3 ", "3/3 "))
                            or "% (" in line):
                        print("  " + line, flush=True)
                if decoder_reader:
                    for line in decoder_reader.read(final=code is not None):
                        meter.decoder_line(line)
                if code is not None:
                    if code:
                        # Include nested decoder errors, whose log is separate from the worker.
                        for path in [log, *([decoder_run / "receiver/reconstruction/00000.log"] if decoder_run else [])]:
                            if path.is_file():
                                with path.open("rb") as tail:
                                    tail.seek(max(0, path.stat().st_size-5000))
                                    print(tail.read().decode(errors="replace"), flush=True)
                        raise RuntimeError(f"{label} failed (exit {code}); log: {log}")
                    break
                elapsed = time.monotonic()-start
                if elapsed >= next_display:
                    try:
                        record = read(ipc)
                    except (FileNotFoundError, json.JSONDecodeError):
                        record = None
                    if meter:
                        if record:
                            meter.completed_segments(record)
                        record = (dict(done=meter.fraction*meter.segments*meter.steps,
                                       total=meter.segments*meter.steps) if meter.loop is not None else record)
                    print(progress_message(label, elapsed, record), flush=True)
                    next_display = elapsed + interval
            print(f"{label}: 100% 완료 | 경과 {(time.monotonic()-start)/60:.1f}분", flush=True)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=45 if "current" in command else 20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        finally:
            write_json(resource, dict(command=list(map(str, command)), seconds=time.monotonic()-start,
                                      returncode=process.returncode))


def prepare(root, p):
    import cv2
    common = root / "common"
    frames = common / "source_frames"
    frames.mkdir(parents=True)
    video = common / "normalized.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", p["source"],
        "-vf", "scale=576:320,fps=24", "-frames:v", str(FRAMES), "-an", "-c:v", "libx264",
        "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", str(video)], check=True)
    info = probe(video)
    if (info["frames"], info["fps"], info["width"], info["height"]) != (FRAMES, FPS, *SIZE):
        raise ValueError("common input must be 384 frames at 576x320/24fps")
    cap = cv2.VideoCapture(str(video))
    hashes = {}
    try:
        for i in range(FRAMES):
            ok, frame = cap.read()
            path = frames / f"{i}.png"
            if not ok or not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"could not extract common source frame {i}")
            hashes[str(i)] = sha256(path)
            if (i+1) % 24 == 0:
                emit(i+1, FRAMES)
    finally:
        cap.release()
    recipe = dict(width=SIZE[0], height=SIZE[1], fps=FPS, source_start_seconds=0,
                  source_end_seconds=SECONDS, duration_limit=SECONDS, normalization=p["normalization"])
    source = dict(id="matched16_"+p["source_sha256"][:8], source_id=Path(p["source"]).stem,
        source_path=p["source"], source_sha256=p["source_sha256"],
        source_duration_seconds=p["source_info"]["duration"], normalized_video=str(video),
        normalized_sha256=sha256(video), source_frames=str(frames), source_frame_hashes=hashes,
        frames=FRAMES, fps=FPS, width=SIZE[0], height=SIZE[1], recipe=recipe,
        sampled_indices=candidate_indices(), reconstruction_started=False)
    current.validate_source_window(source, SECONDS)
    write_json(common / "source_prepared.json", source)


def initialize_baseline(root, p):
    from .workers import csv_write
    source = read(root / "common/source_prepared.json")
    run = root / "lgvsc/run"
    cfg = dict(p["baseline"], input_sha256=source["normalized_sha256"])
    cfg["input_frames"] = FRAMES
    cfg["reconstructed_source_frames"] = candidate_indices()[-1]+1
    current.pc.initialize(root / "lgvsc", source, dict(indices=[0, FRAMES-1], selector="NOT_SELECTED"), cfg)
    # Initialization helper normally consumes a selected set. Remove its placeholder;
    # only the actual InternVL process is allowed to produce baseline keyframes.json.
    (run / "keyframes.json").unlink()
    folder = run / "data/frames/sample"
    csv_write(folder / "frames.csv", [dict(frame_path=str(folder / f"{i}.png"))
              for i in candidate_indices()], ["frame_path"])
    csv_write(run / "data/16x24/videos.csv", [dict(path=source["normalized_video"], frame_save_dir=str(folder))],
              ["path", "frame_save_dir"])


def reconstruct_baseline(run, cfg):
    from .codec_transport import decoder_config_text
    from .decoder_runner import run as decode
    keys = read(run / "keyframes.json")["indices"]
    if segment_lengths(keys)[0] < 17:
        raise ValueError("released reference policy cannot supply its five-latent history from this short first segment; "
                         "stopping without silently enabling current receiver repairs")
    text = decoder_config_text(cfg, REPO, read(run / "receiver/decoder_inputs.json"))
    text += ("\nconditioning_alignment='official_release'\nalign=5\ncpu_video_storage=True\n"
             "reuse_diffusion_model=False\ncache_text_embeddings=False\n")
    path = run / "receiver/decoder_config.py"
    path.write_text(text)
    trace = run / "receiver/reference_trace.json"
    write_json(trace, [])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(REPO / ".local/vendor/Open-Sora")]),
               ETRI_REFERENCE_TRACE=str(trace))
    # Probe records the unmodified append_generated call. No tail17, condition
    # collision, schedule or scoped-noise repair hooks are installed here.
    decode(run / "receiver/metadata.csv", run / "receiver/reconstruction", "key_frames_received",
           run / "receiver", decoder=REPO / "scripts/etri_decoder_probe.py", config=path, environment=env)


def audit_baseline(root, cfg):
    from .etri_60s_check import pts_audit
    run = root / "lgvsc/run"
    keys = read(run / "keyframes.json")["indices"]
    candidates = candidate_indices()
    if keys != sorted(set(keys)) or keys[0] != 0 or keys[-1] != candidates[-1] or not set(keys) <= set(candidates):
        raise ValueError("baseline keys are not the actual ordered released-grid selection")
    expected = output_source_indices(keys, "official_release")
    files = list((run / "receiver/reconstruction").glob("*.mp4"))
    if len(files) != 1:
        raise ValueError("expected one baseline reconstruction")
    pts_audit(files[0], len(expected), FPS)
    info = probe(files[0])
    if (info["width"], info["height"]) != SIZE:
        raise ValueError("baseline output resolution changed")
    trace = read(run / "receiver/reference_trace.json")
    lengths = segment_lengths(keys)
    if len(trace) != len(keys)-2 or any(r["generated_reference_shape"][2] != lengths[i]
                                        for i, r in enumerate(trace)):
        raise ValueError("baseline did not encode the complete previous generated segment")
    inputs = read(run / "receiver/decoder_inputs.json")
    if (inputs["indices"] != keys or inputs["decoder"]["policy"] != "official_release"
            or inputs["decoder"]["concatenation_policy"] != "official_release"):
        raise ValueError("baseline receiver policy/key mismatch")
    if read(run / "channel_accounting.json")["metadata_exact_match"] is not True:
        raise ValueError("baseline metadata did not survive the channel")
    write_json(run / "output_audit.json", dict(status="PASSED", video=str(files[0]),
        video_sha256=sha256(files[0]), frames=len(expected), fps=FPS,
        output_source_indices=expected, selected_keys=keys, source_input_frames=FRAMES,
        reconstructed_source_frames=keys[-1]+1, omitted_source_tail_frames=FRAMES-keys[-1]-1,
        caption_provider="PLLaVA-7B", selector="InternVL2-8B_SKEM_PSSS_0.35",
        selection_candidate_indices=candidates, fixed_key_count=False,
        current_receiver_repairs_enabled=False, quality_acceptance="NOT_EVALUATED"))


def baseline_jobs():
    method = "key_framesinternvl_diff_0.35"
    return [
        ("select", ["keyframes.json", "selector_runtime.json", f"data/frames/sample/{method}"]),
        ("caption", ["captions.json", "caption_sampling.json", "caption_provenance.json", "data/clips"]),
        ("flow", ["metadata_tx.json", "flow_sampling.json"]),
        ("send", ["transmitter", "sender_accounting.json"]),
        ("channel", ["received", "channel_accounting.json"]),
        ("receive", ["receiver/frames", "receiver/metadata.csv", "receiver/decoder_inputs.json", "receiver_accounting.json"]),
        ("reconstruct", ["receiver/decoder_config.py", "receiver/reconstruction", "receiver/reference_trace.json"]),
        ("audit", ["output_audit.json"])]


def baseline(root, p):
    dest, run = root / "lgvsc", root / "lgvsc/run"
    stages = Stages(dest, p["signature"])
    with current.pc.lock(REPO / ".local/etri_60s_check.lock"):
        products = [x for x in current.pc.initialization_products(FRAMES) if x != "run/keyframes.json"]
        products += ["run/data/16x24/videos.csv"]
        stages.step("initialize", ["run"], products, lambda _: initialize_baseline(root, p))
        jobs = baseline_jobs()
        for n, (name, outputs) in enumerate(jobs):
            print(f"LGVSC 단계 {n}/{len(jobs)} 완료 ({100*n/len(jobs):.1f}%, 단계 수 기준) | {name}", flush=True)
            python = p["interpreters"]["channel_python" if name == "channel" else "python"]
            env = dict(environment(p["baseline"]["seed"]), PYTHONPATH=str(REPO / "src"),
                       ETRI_RUN_SIGNATURE=p["signature"])
            if name == "channel":
                env["CUDA_VISIBLE_DEVICES"] = "-1"
            resource = dest / f"resources/{name}.json"
            paths = [*[f"run/{x}" for x in outputs], f"resources/{name}.json"]
            stages.step(name, paths, paths, lambda log, n=name, py=python, e=env, res=resource:
                launch([py, "-m", MODULE, "--output", root, "--worker", n], log, e, res,
                       label=f"2/3 LGVSC · {n}", decoder_run=run if n == "reconstruct" else None))


def run_current(root):
    prepared, dest = root / "common/source_prepared.json", root / "qwen"
    p = current.plan(prepared, dest, duration_limit=SECONDS)
    original = current.launch
    def current_launch(command, log, env, resource):
        run = dest / "reconstruction/event_pairs/run"
        return launch(command, log, env, resource, label=f"3/3 Qwen · {log.stem}",
                      decoder_run=run if log.stem == "reconstruct" else None)
    current.launch = current_launch
    try:
        current.execute(dest, p)
    finally:
        current.launch = original


def worker(root, stage):
    p = read(root / "protocol.json")
    if stage == "prepare":
        return prepare(root, p)
    if stage == "current":
        return run_current(root)
    from . import workers, codec_transport
    run = root / "lgvsc/run"
    cfg = read(run / "run_config.json")
    if stage == "select":
        # workers.select uses frames only to validate the final candidate here.
        # Keep the common 384-frame input; do NOT append an extra candidate/key.
        workers.select(dict(cfg, frames=candidate_indices()[-1]+1), REPO, run)
    elif stage in {"caption", "flow"}:
        getattr(workers, stage)(cfg, REPO, run)
        if stage == "caption":
            write_json(run / "caption_provenance.json", dict(provider="PLLaVA-7B",
                pllava_inference_executed=True, assistant_authored=False,
                model=cfg["models"]["pllava"], sampling="released subclips and four-frame sampler"))
    elif stage == "send":
        # The released candidate grid ends before EOF. Declare exactly the covered
        # frame range in transport, retaining the full common input as evidence.
        codec_transport.send(dict(cfg, frames=candidate_indices()[-1]+1), REPO, run)
    elif stage == "channel":
        current.pc.paired_channel(run)
    elif stage == "receive":
        codec_transport.receive(cfg, REPO, run)
    elif stage == "reconstruct":
        reconstruct_baseline(run, cfg)
    elif stage == "audit":
        audit_baseline(root, cfg)
    else:
        raise ValueError(stage)


def finalize(root, p):
    baseline_audit = read(root / "lgvsc/run/output_audit.json")
    qwen = read(root / "qwen/RESULT.json")
    qwen_run = root / "qwen/reconstruction/event_pairs/run"
    source = read(root / "common/source_prepared.json")
    if (baseline_audit["status"] != "PASSED" or qwen["frames"] != FRAMES
            or qwen["status"] != "COMPLETE_RECONSTRUCTION_INDEPENDENT_REVIEW_PENDING"):
        raise ValueError("both reconstructions must complete before reporting completion")
    for run in (root / "lgvsc/run", qwen_run):
        cfg = read(run / "run_config.json")
        if sha256(run / "data/normalized.mp4") != source["normalized_sha256"]:
            raise ValueError("arms did not use the identical common video")
        if any(cfg[k] != p["baseline"][k] for k in ("fps", "width", "height", "snr_db", "seed", "channel_seed", "steps")):
            raise ValueError("matched settings differ between arms")
    videos = dict(lgvsc=baseline_audit["video"], qwen=qwen["video"])
    result = dict(status="COMPLETE_RECONSTRUCTIONS_REVIEW_PENDING", signature=p["signature"],
        source=source["normalized_video"], source_sha256=source["normalized_sha256"],
        input_frames=FRAMES, fps=FPS, input_duration_seconds=SECONDS, videos=videos,
        video_sha256={k: sha256(v) for k, v in videos.items()},
        lgvsc=dict(keys=baseline_audit["selected_keys"], output_frames=baseline_audit["frames"],
            reconstructed_source_frames=baseline_audit["reconstructed_source_frames"],
            omitted_source_tail_frames=baseline_audit["omitted_source_tail_frames"],
            source_index_map="lgvsc/run/output_audit.json", concatenation_policy="official_release"),
        qwen=dict(keys=read(qwen_run / "keyframes.json")["indices"], output_frames=FRAMES,
                  concatenation_policy="endpoint_exact"),
        scope=p["scope"], noise_scope=p["noise_scope"], training_executed=False,
        independent_semantic_review="PENDING", temporal_quality_comparison="PENDING",
        quality_improvement_verified=False)
    write_json(root / "RESULT.json", result)
    def video_tag(path):
        relative = os.path.relpath(path, root)
        return f'<video controls preload="metadata" src="{html.escape(relative, quote=True)}"></video>'
    page = ('<!doctype html><html lang="ko"><meta charset="utf-8"><title>LGVSC / Qwen 16초 실행 결과</title>'
        '<style>body{font:16px sans-serif;max-width:1200px;margin:30px auto}video{width:100%;max-width:576px}'
        'section{margin:28px 0}code{overflow-wrap:anywhere}</style><h1>같은 입력의 LGVSC / Qwen 복원</h1>'
        '<p>세 작업 실행 완료. 버벅임·의미 보존의 비교 판정은 아직 수행하지 않았습니다.</p>'
        f'<p>공통 입력: 16초·384프레임·24fps. LGVSC는 공개 후보 규칙대로 380번에서 끝나며 '
        f'연결 시 중복 경계를 유지합니다. 원본 출력 {baseline_audit["frames"]}프레임과 Qwen 384프레임은 '
        '재생 시각이 완전히 일치하지 않을 수 있습니다. 동기 비교에는 output_source_indices를 사용해야 합니다.</p>'
        '<section><h2>공통 원본</h2>'+video_tag(source["normalized_video"])+ '</section>'
        '<section><h2>LGVSC — InternVL / PLLaVA / 공개 수신부 정책</h2>'+video_tag(videos["lgvsc"])+ '</section>'
        '<section><h2>Qwen 의미 변화 순차 선택 + 개선 캡션</h2>'+video_tag(videos["qwen"])+ '</section>'
        '<p><a href="RESULT.json">실행 결과</a> · <a href="protocol.json">설정·출처</a></p></html>')
    (root / "report.html").write_text(page)
    return result


def execute(root, p, prepare_only=False):
    root.mkdir(parents=True, exist_ok=True)
    with current.pc.lock(root / ".execution.lock"):
        current.freeze(root / "protocol.json", p)
        stages = Stages(root, p["signature"])
        env = dict(environment(p["baseline"]["seed"]), PYTHONPATH=str(REPO / "src"))
        if (root / "stages/01-common-input.json").exists():
            print("재개 검증 중 | 완료 단계의 파일 해시 확인", flush=True)
        else:
            print("전체 0/3 완료 (0.0%, 작업 수 기준) | 1/3 공통 입력 준비", flush=True)
        stages.step("01-common-input", ["common", "resources/prepare.json"], ["common", "resources/prepare.json"],
            lambda log: launch([p["interpreters"]["python"], "-m", MODULE, "--output", root, "--worker", "prepare"],
                log, env, root / "resources/prepare.json", label="1/3 공통 입력 준비"))
        if prepare_only:
            print("공통 입력 준비 완료 (1/3). 모델 추론은 실행하지 않았습니다.", flush=True)
            return
        print("전체 1/3 완료 (33.3%, 작업 수 기준) | 2/3 LGVSC", flush=True)
        # Empty ownership retains child-stage receipts/checkpoints on interruption.
        stages.step("02-lgvsc", [], ["lgvsc"], lambda _: baseline(root, p))
        print("전체 2/3 완료 (66.7%, 작업 수 기준) | 3/3 현재 Qwen", flush=True)
        stages.step("03-current-qwen", [], ["qwen", "resources/current.json"],
            lambda log: launch([REPO / ".local/qwen35_env/bin/python", "-m", MODULE,
                "--output", root, "--worker", "current"], log, env,
                root / "resources/current.json", label="3/3 현재 Qwen"))
        result = finalize(root, p)
        print("전체 3/3 완료 (100%, 작업 수 기준)", flush=True)
        print(json.dumps(dict(status=result["status"], report=str(root / "report.html"),
                              videos=result["videos"]), ensure_ascii=False, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="read-only preflight; no model inference")
    parser.add_argument("--prepare-only", action="store_true", help="only prepare common input; resume with the same command")
    parser.add_argument("--worker", choices=["prepare", "current", *[n for n, _ in baseline_jobs()]], help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
    root = args.output.resolve()
    try:
        if args.worker:
            if args.check or args.prepare_only:
                raise ValueError("worker cannot be combined with check/prepare-only")
            return worker(root, args.worker)
        print("입력·모델·기존 설정 확인 중 (모델 추론 없음)", flush=True)
        p = preflight(args.input.resolve(strict=True), root)
        print(f"준비 확인 완료 | 384프레임·24fps·16초 | 출력: {root}", flush=True)
        if args.check:
            print(json.dumps(dict(status="READY", model_inference_started=False, stages=3,
                                  output=str(root)), ensure_ascii=False))
            return
        execute(root, p, args.prepare_only)
    except KeyboardInterrupt:
        print("중단됨. 로그·완료 단계는 보존됐습니다. 같은 명령으로 재개할 수 있습니다.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"실패: {error}\n완료로 표시하지 않았습니다. 로그를 확인한 뒤 같은 명령으로 재개하세요.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
