"""Public-weight Framer pilot on hash-verified, already received SPARC packets.

No training, new key selection, source tracking, or decoder access to source RGB.
Inference receives receiver.json only; source RGB is opened by evaluation.
"""
import argparse
import csv
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .artifacts import sha256, write_json
from .webvid5 import fingerprint

REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / "outputs/kinetics_001_sparse_motion_20261019"
DEFAULT = REPO / "outputs/kinetics_001_framer_20261020"
VENDOR_REVISION = "d0ba81f1f23a639396d4c6a801c53a5b21412b06"
PROFILE = dict(
    steps=30, control_scale=1.0, min_guidance=1.0, max_guidance=3.0,
    motion_bucket_id=127, conditioning_fps=7, diffusion_seed=42,
    decode_chunk_size=8, dtype="float16", sift_track_update=False,
    resolution="native 576x320", temporal_policy="actual interval length, no resampling",
    training=False, residual_stream=False,
)


def read(path):
    return json.loads(Path(path).read_text())


def verify(files, base=None):
    for name, digest in files.items():
        path = Path(name) if base is None else Path(base) / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Artifact changed or missing: {path}")


def state(root, stage, status="RUNNING", **extra):
    path = root / "status.json"
    old = read(path) if path.exists() else {}
    now = time.time()
    value = dict(old, status=status, stage=stage, updated=now,
                 started=old.get("started", now), pid=os.getpid(), **extra)
    write_json(path, value)
    print(json.dumps({k: value[k] for k in ("status", "stage", *extra)}, ensure_ascii=False),
          flush=True)


def prepare(root, reference):
    """Audit historical transmission and freeze exactly the bytes available to RX."""
    from . import sparc_motion_transport as transport
    root.mkdir(parents=True, exist_ok=True)
    old = read(reference / "protocol.json")
    source = read(reference / "source.json")
    if source != old["source"]:
        raise ValueError("Source provenance differs")
    verify(source["hashes"], source["directory"])
    for stage_name in ("encode", "channel", "decode"):
        stage = read(reference / "stages" / f"{stage_name}.json")
        if stage["status"] != "PASSED":
            raise ValueError("Reference stage incomplete")
        verify(stage["artifacts"], reference)
    info = read(reference / "decoded/decoded.json")["keys"]
    received = reference / "received"
    keys = (received / "keys.bin").read_bytes()
    motion = (received / "motion.bin").read_bytes()
    header, _, indices = transport.parse_keys(keys)
    tracks = transport.parse_motion(motion, keys)
    if indices != info["indices"] or header["video"] != source["video"]:
        raise ValueError("Received timeline differs")
    if info["packet_sha256"] != sha256(received / "keys.bin"):
        raise ValueError("Decoded keyframes belong to different bytes")
    packets = read(received / "channel.json")["packets"]
    sender = read(reference / "sender/sender.json")
    for name in ("keys", "motion"):
        if packets[name]["bit_errors"] or sha256(received / f"{name}.bin") != sender["packet_sha256"][name]:
            raise ValueError("Reference channel did not preserve the packet")
    vendor = REPO / ".local/framer_vendor"
    if subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip() != VENDOR_REVISION:
        raise ValueError("Framer revision differs")
    from .framer_compat import check as check_vendor
    vendor_patch = check_vendor(vendor)
    weights = REPO / ".local/framer_weights"
    assets = read(weights / "manifest.json")
    for name, meta in assets.items():
        verify(meta["files"], weights / name)
    files = [reference / x for x in (
        "protocol.json", "source.json", "comparison.json", "received/keys.bin",
        "received/motion.bin", "received/channel.json", "decoded/decoded.json")]
    files += [reference / f"decoded/keys/{i:05d}.png" for i in indices]
    for name in ("keys", "motion"):
        row = read(reference / "comparison.json")["results"][name]
        movie = reference / f"reconstruction/{name}/reconstruction.mp4"
        if sha256(movie) != row["movie_sha256"]:
            raise ValueError("Historical comparison movie changed")
        files.append(movie)
    code_files = [Path(__file__).resolve(), Path(__file__).with_name("framer_compat.py").resolve()]
    code = {str(f): sha256(f) for f in code_files}
    p = dict(schema="sparc_framer_pilot_v1", reference=str(reference), source=source,
             profile=PROFILE, vendor_revision=VENDOR_REVISION, vendor_patch=vendor_patch, assets=assets, code=code,
             reference_files={str(f): sha256(f) for f in files},
             source_pixels_at_receiver=False, new_qwen_calls=0, new_tracking_calls=0,
             channel_uses=sum(packets[n]["complex_channel_uses"] for n in ("keys", "motion")),
             physical_link_overhead_included=False,
             public_training_geometry="512x320, public demo 14 frames",
             actual_inference_geometry=source["video"],
             limitation="Native resolution and variable length adaptation; not an exact paper reproduction")
    p["signature"] = fingerprint(p)
    if (root / "protocol.json").exists() and read(root / "protocol.json") != p:
        raise ValueError("Frozen experiment changed; retain outputs and choose another output directory")
    write_json(root / "protocol.json", p)
    receiver = dict(video=source["video"], indices=indices,
                    keys_directory=str(reference / "decoded/keys"),
                    keys_packet=str(received / "keys.bin"), motion_packet=str(received / "motion.bin"),
                    input_hashes={str(f): sha256(f) for f in files
                                  if "/received/" in str(f) or "/decoded/" in str(f)},
                    vendor=str(vendor), weights=str(weights), profile=PROFILE,
                    signature=p["signature"], trajectory_windows=len(tracks))
    write_json(root / "receiver.json", receiver)
    state(root, "prepared", segments_done=0, segments_total=len(indices)-1,
          frames_done=0, frames_total=source["video"]["frames"], error=None)
    return p


def tracks_for_receiver(trajectory, width, height):
    """Coordinates are quantized at 288x160 in the received packet."""
    import numpy as np
    valid = trajectory["valid"].all(axis=0)
    if not valid.any():
        raise ValueError("Framer pilot requires a persistent received track")
    xy = trajectory["xy"][:, valid].astype(np.float32).copy()
    xy *= np.asarray([width / 288, height / 160], dtype=np.float32)
    if not np.isfinite(xy).all() or (xy < 0).any():
        raise ValueError("Invalid received trajectory")
    return xy, int((~valid).sum())


def load_pipeline(receiver):
    import torch
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
    from .framer_compat import check as check_vendor
    check_vendor(receiver["vendor"])
    from .framer_compat import patch_temporal_feed_forward
    compatibility = patch_temporal_feed_forward()
    sys.path.insert(0, receiver["vendor"])
    from models_diffusers.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
    from models_diffusers.controlnet_svd import ControlNetSVDModel
    from pipelines.pipeline_stable_video_diffusion_interp_control import StableVideoDiffusionInterpControlPipeline
    weights = Path(receiver["weights"])
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        weights / "framer/unet", torch_dtype=torch.float16, low_cpu_mem_usage=True,
        custom_resume=True, local_files_only=True)
    controlnet = ControlNetSVDModel.from_pretrained(
        weights / "framer/controlnet", torch_dtype=torch.float16, local_files_only=True)
    unet.enable_xformers_memory_efficient_attention()
    unet.enable_forward_chunking(chunk_size=1, dim=0)
    controlnet.enable_forward_chunking(chunk_size=1, dim=0)
    pipe = StableVideoDiffusionInterpControlPipeline(
        vae=AutoencoderKLTemporalDecoder.from_pretrained(
            weights / "svd/vae", torch_dtype=torch.float16, variant="fp16", local_files_only=True),
        image_encoder=CLIPVisionModelWithProjection.from_pretrained(
            weights / "svd/image_encoder", torch_dtype=torch.float16, variant="fp16", local_files_only=True),
        unet=unet, controlnet=controlnet,
        scheduler=EulerDiscreteScheduler.from_pretrained(weights / "svd/scheduler", local_files_only=True),
        feature_extractor=CLIPImageProcessor.from_pretrained(weights / "svd/feature_extractor", local_files_only=True),
    )
    for module in (pipe.unet, pipe.controlnet, pipe.vae, pipe.image_encoder):
        # The custom optional ControlNet is not moved by DiffusionPipeline.to().
        module.to(device="cuda", dtype=torch.float16).eval().requires_grad_(False)
        if any(parameter.device.type != "cuda" for parameter in module.parameters()):
            raise ValueError("Framer module was not fully placed on CUDA")
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.sparc_compatibility = compatibility
    return pipe


def restore(root):
    import numpy as np
    import torch
    from PIL import Image
    from . import sparc_motion_transport as transport
    from .sparc_motion import export
    r = read(root / "receiver.json")
    verify(r["input_hashes"])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(PROFILE["diffusion_seed"])
    decoded_tracks = transport.parse_motion(Path(r["motion_packet"]).read_bytes(),
                                           Path(r["keys_packet"]).read_bytes())
    video = r["video"]
    ids = r["indices"]
    anchors = {i: np.asarray(Image.open(Path(r["keys_directory"]) / f"{i:05d}.png").convert("RGB"))
               for i in ids}
    pipe = None
    chunks = root / "segments"
    chunks.mkdir(exist_ok=True)
    frames = np.empty((video["frames"], video["height"], video["width"], 3), np.uint8)
    records = []
    started = time.time()
    for number, (a, b) in enumerate(zip(ids, ids[1:]), 1):
        segment = chunks / f"{a:05d}_{b:05d}.npy"
        record_path = segment.with_suffix(".json")
        identity = dict(protocol=r["signature"], start=a, end=b)
        if record_path.exists():
            record = read(record_path)
            if record["identity"] != identity or sha256(segment) != record["sha256"]:
                raise ValueError("Completed Framer segment changed")
            result = np.load(segment, allow_pickle=False)
            state(root, "restore (verified segment reuse)", segments_done=number,
                  frames_done=b+1, step=30, steps=30)
        else:
            if pipe is None:
                state(root, "load Framer public weights")
                pipe = load_pipeline(r)
                write_json(root / "runtime.json", dict(
                    python=sys.version, gpu=torch.cuda.get_device_name(0),
                    packages={n: importlib.metadata.version(n) for n in (
                        "torch", "torchvision", "diffusers", "transformers", "accelerate", "xformers")},
                    forward_chunk_size=1, controlnet_forward_chunk_size=1,
                    xformers=True, cpu_offload=False, compatibility=pipe.sparc_compatibility))
            points, discarded = tracks_for_receiver(decoded_tracks[a, b], video["width"], video["height"])
            if len(points) != b-a+1:
                raise ValueError("Received motion timeline differs")
            inputs = dict(identity=identity, points=points.shape[1], discarded_invalid_tracks=discarded,
                          all_received_visibility_valid=bool(decoded_tracks[a, b]["valid"].all()),
                          seed=PROFILE["diffusion_seed"]+a)
            torch.cuda.reset_peak_memory_stats()
            tick = time.time()

            def callback(pipeline, step, timestep, tensors):
                state(root, f"restore {a}..{b}", segments_done=number-1, segments_total=len(ids)-1,
                      frames_done=a, frames_total=video["frames"], step=step+1, steps=PROFILE["steps"])
                return tensors

            with torch.inference_mode():
                result = pipe(
                    image=Image.fromarray(anchors[a]), image_end=Image.fromarray(anchors[b]),
                    with_control=True, point_tracks=torch.from_numpy(points).to("cuda", torch.float16),
                    point_embedding=None, with_id_feature=False,
                    controlnet_cond_scale=PROFILE["control_scale"],
                    num_frames=b-a+1, width=video["width"], height=video["height"],
                    decode_chunk_size=PROFILE["decode_chunk_size"],
                    generator=torch.Generator(device="cpu").manual_seed(inputs["seed"]),
                    motion_bucket_id=PROFILE["motion_bucket_id"], fps=PROFILE["conditioning_fps"],
                    num_inference_steps=PROFILE["steps"], min_guidance_scale=PROFILE["min_guidance"],
                    max_guidance_scale=PROFILE["max_guidance"], sift_track_update=False,
                    callback_on_step_end=callback,
                ).frames[0]
            result = np.stack([np.asarray(f.convert("RGB")) for f in result])
            # Keep the actual received anchors, as in all previous SPARC receivers.
            result[0], result[-1] = anchors[a], anchors[b]
            record = dict(identity=identity, input=inputs, seconds=time.time()-tick,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
            with segment.with_suffix(".tmp").open("wb") as f:
                np.save(f, result, allow_pickle=False)
            os.replace(segment.with_suffix(".tmp"), segment)
            record["sha256"] = sha256(segment)
            write_json(record_path, record)
        if result.shape != (b-a+1, video["height"], video["width"], 3):
            raise ValueError("Framer changed the original video timeline")
        if not np.array_equal(result[0], anchors[a]) or not np.array_equal(result[-1], anchors[b]):
            raise ValueError("Received endpoint changed")
        frames[a:b+1] = result
        records.append(record)
        state(root, "restore", segments_done=number, frames_done=b+1, step=30, steps=30)
    dest = root / "reconstruction"
    movie = dest / "reconstruction.mp4"
    if movie.exists():
        raise ValueError("Unregistered final movie exists; preserve it before retry")
    state(root, "export final MP4")
    export(frames, video, dest)
    report = dict(signature=r["signature"], segments=records, frames=video["frames"],
                  seconds_this_invocation=time.time()-started, movie_sha256=sha256(movie),
                  source_pixels_at_receiver=False, anchors_preserved_before_mp4=True,
                  new_transmitted_data=False)
    write_json(root / "restore.json", report)


def evaluate(root):
    import numpy as np
    from PIL import Image
    from .official_quality import OfficialMetrics
    from .research_quality import read_video
    from .stutter_metrics import evaluate as motion_metrics
    p = read(root / "protocol.json")
    report = read(root / "restore.json")
    movie = root / "reconstruction/reconstruction.mp4"
    if report["signature"] != p["signature"] or sha256(movie) != report["movie_sha256"]:
        raise ValueError("Inference/evaluation provenance differs")
    source = p["source"]
    verify(source["hashes"], source["directory"])
    original = np.stack([np.asarray(Image.open(Path(source["directory"]) / f"{i}.png").convert("RGB"))
                         for i in range(source["video"]["frames"])])
    frames = read_video(movie)
    model = OfficialMetrics()
    quality, rows = model.evaluate(original, frames, lambda n, t:
                                  state(root, "evaluate", metric_done=n, metric_total=t) if n % 24 == 0 else None)
    with (root / "quality_per_frame.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    state(root, "evaluate motion diagnostics")
    motion = motion_metrics(original, frames)
    old = read(Path(p["reference"]) / "comparison.json")
    baselines = {}
    for name, row in {**old["results"], **old["historical"]}.items():
        baselines[name] = {k: v for k, v in row.items()
                           if k in ("quality", "channel_uses", "anchor_count", "movie_sha256")}
    result = dict(status="COMPLETE_FRAMER_PUBLIC_WEIGHT_PILOT", protocol_signature=p["signature"],
                  channel_uses=p["channel_uses"], quality=quality, motion_diagnostics=motion,
                  frames=len(frames), anchors=len(read(root / "receiver.json")["indices"]),
                  movie=str(movie), movie_sha256=sha256(movie), baselines=baselines,
                  new_training=False, source_pixels_at_receiver=False,
                  resources=model.resources, restore_seconds=sum(s["seconds"] for s in report["segments"]),
                  peak_inference_allocated_bytes=max(s["peak_allocated_bytes"] for s in report["segments"]),
                  limitations=["One video, one channel condition and one diffusion seed policy",
                               "Native 576x320 and variable interval length; public demo is 512x320 and 14 frames",
                               "Uses existing 8-point/5-knot packets; no learned residual correction",
                               "No independent action or semantic evaluation",
                               "Physical link overhead excluded, matching historical accounting"])
    write_json(root / "RESULT.json", result)
    state(root, "finished", status="COMPLETE", frames_done=len(frames), metric_done=len(frames)*4)


def run(root, reference):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(root, reference)
        if (root / "RESULT.json").exists():
            result = read(root / "RESULT.json")
            if result["protocol_signature"] != read(root / "protocol.json")["signature"]:
                raise ValueError("Result protocol differs")
            verify({result["movie"]: result["movie_sha256"]})
            state(root, "verified completed result", status="COMPLETE")
            return
        env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONNOUSERSITE="1",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        env.pop("LD_LIBRARY_PATH", None)
        try:
            for stage, python in (
                ("restore", REPO / ".local/framer_env/bin/python"),
                ("evaluate", REPO / ".local/bsu_codec_eval_env/bin/python"),
            ):
                if stage == "restore" and (root / "restore.json").exists():
                    verify({str(root / "reconstruction/reconstruction.mp4"): read(root / "restore.json")["movie_sha256"]})
                    continue
                subprocess.run([str(python), "-m", "semantic_transmission.sparc_framer",
                                stage, "--output", str(root)], cwd=REPO, env=env, check=True)
        except BaseException as exc:
            state(root, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                  status="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                  error=f"{type(exc).__name__}: {exc}")
            raise


def watch(root, interval, once=False):
    import shutil
    from .terminal_progress import fit
    live = sys.stdout.isatty() and not once
    previous = []
    if live:
        print("\033[?1049h\033[?25l", end="", flush=True)
    try:
        while True:
            s = read(root / "status.json") if (root / "status.json").exists() else {}
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                  "--format=csv,noheader"], capture_output=True, text=True, timeout=3).stdout.strip()
            n, total = s.get("frames_done", 0), s.get("frames_total", 240)
            end = s.get("updated", time.time()) if s.get("status") in ("COMPLETE", "FAILED", "INTERRUPTED") else time.time()
            rows = [f"Status   {s.get('status', 'NOT_STARTED')}", f"Stage    {s.get('stage', 'pending')}",
                    f"Frames   {n}/{total} ({100*n/total:.1f}%) | segments {s.get('segments_done',0)}/{s.get('segments_total',30)}",
                    f"Denoise  {s.get('step',0)}/{s.get('steps',30)} | metric work {s.get('metric_done',0)}/{s.get('metric_total','unknown')}",
                    f"Elapsed  {(end-s.get('started',end))/60:.1f} min | ETA unknown (variable-length segments)",
                    f"GPU      utilization / used VRAM / total VRAM: {gpu or 'unavailable'}",
                    f"Log      {root/'run.log'}", f"Error    {s.get('error') or 'none'}",
                    "Resume   bash scripts/run_sparc_framer.sh run (hash-verified segment reuse)"]
            if not live:
                print("\n".join(rows))
                return
            width = shutil.get_terminal_size((110, 24)).columns
            rows = [fit(r, width) for r in rows]
            for i, row in enumerate(rows):
                if i >= len(previous) or previous[i] != row:
                    print(f"\033[{i+1};1H{row}\033[K", end="")
            sys.stdout.flush()
            previous = rows
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        if live:
            print("\033[?25h\033[?1049l", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "restore", "evaluate", "watch", "status"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--interval", type=float, default=1)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.command in ("watch", "status"):
        if not 0 < args.interval <= 60:
            parser.error("interval must be in (0,60]")
        return watch(root, args.interval, args.command == "status")
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    if args.command == "run":
        return run(root, args.reference.resolve())
    try:
        return restore(root) if args.command == "restore" else evaluate(root)
    except BaseException as exc:
        state(root, args.command, status="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
              error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
