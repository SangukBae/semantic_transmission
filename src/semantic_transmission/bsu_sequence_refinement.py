"""Full-timeline BSU refinement with separately paid, channel-tested trajectories.

The sender reads source pixels. The receiver reads only received keys, received
trajectory bytes and generated frames. Historical two-key experiments stay frozen.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import shutil
import time

from .artifacts import sha256, write_json
from .packets import accounting
from .wire import pack, unpack

POLICY = "bsu_multisegment_trajectory_v1"
TRIALS = ((16, 9), (12, 9), (12, 7), (8, 7), (8, 5), (4, 5), (4, 3))
MODES = ("combined", "cadence", "none")


def read(path):
    return json.loads(Path(path).read_text())


def progress(label, done, total, started):
    elapsed = time.monotonic() - started
    eta = f"{elapsed / done * (total-done):.1f}s" if done else "unknown"
    print(f"{label} {done}/{total} ({100*done/total:.1f}%) elapsed={elapsed:.1f}s ETA={eta}", flush=True)


def validate_keys(keys, frames):
    if (frames < 2 or len(keys) < 2 or keys != sorted(set(keys))
            or keys[0] != 0 or keys[-1] != frames-1
            or any(type(k) is not int for k in keys)):
        raise ValueError("keys must cover the complete timeline with unique ordered endpoints")


def paths_for_run(base):
    run = Path(base) / "reconstruction/event_pairs/run"
    audit = read(run / "output_audit.json")
    video = Path(audit["video"])
    return run, video, video.with_name(video.stem + "_frames")


def sender_candidates(frames, start, end):
    """Track once, then use only sender geometry to form predeclared budgets."""
    import numpy as np
    from . import sparse_trajectory as sparse
    tracks = [sparse.track(frames, reverse=r) for r in (False, True)]
    options = []
    for points, knots in TRIALS:
        pairs = []
        for reverse, (xy, valid) in enumerate(tracks):
            selected = sparse.choose_tracks(xy, valid, points // 2, bool(reverse))
            pairs.append((xy[:, selected], valid[:, selected], np.full(len(selected), reverse)))
        xy = np.concatenate([p[0] for p in pairs], axis=1)
        valid = np.concatenate([p[1] for p in pairs], axis=1)
        origins = np.concatenate([p[2] for p in pairs])
        if xy.shape[1] < 3:
            return []
        times = sparse.adaptive_knots(xy, valid, knots)
        options.append(dict(start=start, end=end,
            trajectory=sparse.encode(xy[times], valid[times], origins, times, len(frames))))
    return options


def choose_packet(candidates, video, base_digest, max_uses, bits_per_symbol=4):
    """One paid packet for the video; never count trajectories as free metadata."""
    for level in range(len(TRIALS)):
        windows = [c[level] for c in candidates if c]
        if not windows:
            break
        packet = pack(dict(policy=POLICY, video=video, base_metadata_sha256=base_digest, windows=windows))
        if accounting(len(packet), bits_per_symbol=bits_per_symbol)["complex_channel_uses"] <= max_uses:
            return packet, dict(level=level, sent_windows=len(windows), omitted_windows=len(candidates)-len(windows))
    # Do not pick scenes by reconstructed quality. If the bounded representation
    # cannot cover the available windows, explicitly fall back to cadence only.
    return None, dict(sent_windows=0, omitted_windows=len(candidates), reason="no useful full-video trajectory packet fits the declared budget")


def decode_packet(packet, base_packet):
    from .sparse_trajectory import decode
    header, payload = unpack(packet)
    base, _ = unpack(base_packet)
    keys = [k["index"] for k in base["keyframes"]]
    validate_keys(keys, base["video"]["frames"])
    if (payload or header.get("policy") != POLICY or header.get("video") != base["video"]
            or header.get("base_metadata_sha256") != hashlib.sha256(base_packet).hexdigest()):
        raise ValueError("trajectory packet does not match received base video")
    intervals = set(zip(keys, keys[1:]))
    decoded = {}
    for window in header["windows"]:
        interval = (window["start"], window["end"])
        if interval not in intervals or interval in decoded:
            raise ValueError("duplicate or invalid trajectory interval")
        decoded[interval] = decode(window["trajectory"], interval[1]-interval[0]+1)
    return decoded


def prepare(job, output):
    import cv2
    import numpy as np
    from PIL import Image
    run, _, _ = paths_for_run(job["base"])
    base_packet = (run / "transmitter/metadata.bin").read_bytes()
    header, _ = unpack(base_packet)
    keys = [k["index"] for k in header["keyframes"]]
    validate_keys(keys, header["video"]["frames"])
    # This cap is relative to this video's measured BSU cost, not to an
    # unmeasured LGVSC result or the paper's rounded dataset CBR.
    base_uses = job["base_channel_uses"]
    bits = job["digital_bits_per_symbol"]
    max_uses = math.floor(base_uses * job["extra_rate_ratio"])
    cv2.setNumThreads(2); cv2.setRNGSeed(2025)
    candidates = []
    started = time.monotonic()
    if job["mode"] == "combined" and max_uses >= accounting(1, bits_per_symbol=bits)["complex_channel_uses"]:
        source = Path(job["source_frames"])
        for i, (a, b) in enumerate(zip(keys, keys[1:]), 1):
            if b-a+1 > 65535:
                raise ValueError("trajectory interval exceeds the encoded time range")
            frames = []
            for index in range(a, b+1):
                with Image.open(source / f"{index}.png") as image:
                    frames.append(cv2.resize(np.asarray(image.convert("RGB")), (288, 160), interpolation=cv2.INTER_AREA))
            candidates.append(sender_candidates(np.stack(frames), a, b))
            progress("TRAJECTORY sender windows", i, len(keys)-1, started)
        packet, audit = choose_packet(candidates, header["video"], sha256(run / "transmitter/metadata.bin"), max_uses, bits)
    else:
        packet, audit = None, dict(reason="mode does not request trajectories or budget below one LDPC block", sent_windows=0)
    output.mkdir(parents=True, exist_ok=True)
    if packet is not None:
        decode_packet(packet, base_packet)
        (output / "trajectory.bin").write_bytes(packet)
    write_json(output / "sender.json", dict(packet_present=packet is not None, audit=audit,
        extra_channel_uses=accounting(len(packet), bits_per_symbol=bits)["complex_channel_uses"] if packet else 0,
        maximum_extra_channel_uses=max_uses, base_channel_uses=base_uses,
        budget_reference="measured same-video BSU base transmission", mode=job["mode"], bits_per_symbol=bits))


def channel(job, sender, received):
    sender_info = read(sender / "sender.json")
    received.mkdir(parents=True, exist_ok=True)
    report = dict(status="PASSED", packet_present=sender_info["packet_present"], complex_channel_uses=0)
    if sender_info["packet_present"]:
        from .metadata_channel import transmit
        original = (sender / "trajectory.bin").read_bytes()
        packet, report = transmit(original, job["snr_db"], job["trajectory_channel_seed"],
                                  bits_per_symbol=job["digital_bits_per_symbol"])
        # CRC failure never feeds distorted coordinates to the receiver.
        run, _, _ = paths_for_run(job["base"])
        decode_packet(packet, (run / "received/metadata.bin").read_bytes())
        if packet != original:
            raise ValueError("trajectory metadata did not survive the channel exactly")
        (received / "trajectory.bin").write_bytes(packet)
        report.update(status="PASSED", packet_present=True, exact_match=True)
    write_json(received / "channel.json", report)
    print(f"TRAJECTORY channel 1/1 (100%): {report['complex_channel_uses']} complex symbols", flush=True)


def refine_windows(frame_paths, keys, key_directory, trajectories, destination, mode, callback=None):
    import cv2
    import numpy as np
    from PIL import Image
    from .trajectory_receiver import refine_references, refine_temporal
    if mode not in MODES:
        raise ValueError("unsupported full-video refinement mode")
    validate_keys(keys, len(frame_paths))
    cv2.setNumThreads(2); cv2.setRNGSeed(2025)
    destination.mkdir(parents=True, exist_ok=True)
    audits = []
    for i, (a, b) in enumerate(zip(keys, keys[1:])):
        pixels = []
        for p in frame_paths[a:b+1]:
            with Image.open(p) as image:
                pixels.append(np.asarray(image.convert("RGB")))
        base = np.stack(pixels)
        current = base
        audit = dict(start=a, end=b)
        # Each pass has its own counter; reference and cadence are not additive
        # frame completion percentages for the whole video.
        pass_started = time.monotonic()
        def tick(n, total):
            if n == 1 or n % 32 == 0 or n == total:
                progress(f"WINDOW {i+1}/{len(keys)-1} {phase}", n, total, pass_started)
        if mode == "combined":
            received_keys = []
            for k in (a, b):
                with Image.open(key_directory / f"{k}.png") as image:
                    received_keys.append(np.asarray(image.convert("RGB")))
            phase = "reference"; pass_started = time.monotonic()
            current, audit["reference"] = refine_references(current, received_keys, trajectories.get((a,b)), tick)
        if mode != "none":
            phase = "cadence"; pass_started = time.monotonic()
            current, audit["cadence"] = refine_temporal(current, tick)
        if not np.array_equal(current[[0,-1]], base[[0,-1]]):
            raise ValueError("refinement changed a shared key endpoint")
        for offset, pixels in enumerate(current):
            if i and offset == 0:
                continue  # Exactly one copy of every shared key boundary.
            Image.fromarray(pixels).save(destination / f"{a+offset:05d}.png")
        audits.append(audit)
        if callback:
            callback(b+1, len(frame_paths))
    return audits


def refine(job, received, output):
    run, movie, frames = paths_for_run(job["base"])
    base_packet = (run / "received/metadata.bin").read_bytes()
    header, _ = unpack(base_packet)
    keys = [k["index"] for k in header["keyframes"]]
    paths = sorted(frames.glob("*.png"))
    if len(paths) != header["video"]["frames"] or [p.name for p in paths] != [f"{i:05d}.png" for i in range(len(paths))]:
        raise ValueError("base generated frames do not cover the complete timeline")
    packet = received / "trajectory.bin"
    trajectories = decode_packet(packet.read_bytes(), base_packet) if packet.exists() else {}
    started = time.monotonic()
    last = [0]
    def tick(n, total):
        if n == total or n-last[0] >= 32:
            progress("REFINEMENT frames (reference/cadence passes)", n, total, started); last[0] = n
    audits = refine_windows(paths, keys, run / "receiver/frames/sample/key_frames_received",
                            trajectories, output / "frames", job["mode"], tick)
    print("EXPORT full-length MP4; encoding ETA unknown", flush=True)
    if job["mode"] == "none":
        shutil.copyfile(movie, output / "reconstruction.mp4")
    else:
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-framerate", str(header["video"]["fps"]),
            "-i", str(output / "frames/%05d.png"), "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-pix_fmt", "yuv420p", str(output / "reconstruction.mp4")], check=True)
    from .etri_60s_check import pts_audit
    pts_audit(output / "reconstruction.mp4", len(paths), header["video"]["fps"])
    write_json(output / "refinement.json", dict(policy=POLICY, mode=job["mode"], windows=audits,
        receiver_reads_source=False, all_key_endpoints_preserved=True, frames=len(paths),
        fps=header["video"]["fps"], seconds=time.monotonic()-started))


def evaluate(job, output):
    import csv
    from .official_quality import OfficialMetrics
    from .research_quality import read_video, read_frames
    from .stutter_metrics import evaluate as temporal_metrics
    run, _, _ = paths_for_run(job["base"])
    source = read_video(run / "data/normalized.mp4")
    result = dict(evaluation_profile="lgvsc_official_metrics_v1", source_sha256=sha256(run / "data/normalized.mp4"),
                  video_sha256=sha256(output / "reconstruction.mp4"))
    metrics = OfficialMetrics()
    for boundary in ("lossless_frames", "delivered_mp4"):
        frames = read_frames(output / "frames") if boundary == "lossless_frames" else read_video(output / "reconstruction.mp4")
        started = time.monotonic()
        def tick(n, total):
            if n == 1 or n % 100 == 0 or n == total:
                progress(f"METRICS {boundary}", n, total, started)
        summary, rows = metrics.evaluate(source, frames, tick)
        result[boundary] = summary
        with (output / f"quality_{boundary}.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        if boundary == "delivered_mp4" and len(frames) >= 3:
            started = time.monotonic()
            def temporal_tick(n, total):
                if n == 1 or n % 100 == 0 or n == total:
                    progress("TEMPORAL delivered_mp4 frame pairs", n, total, started)
            result["temporal"] = temporal_metrics(source, frames, temporal_tick)
    write_json(output / "quality.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "channel", "refine", "evaluate"))
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = read(args.job)
    from .bsu_webvid import verify_plan
    plan = read(args.job.parent.parent.parent / "protocol.json")
    verify_plan(plan)
    if job["benchmark_signature"] != plan["signature"]:
        raise ValueError("worker job belongs to another benchmark")
    root = args.job.parent
    if args.stage == "prepare": prepare(job, root / "sender")
    elif args.stage == "channel": channel(job, root / "sender", root / "received")
    elif args.stage == "refine": refine(job, root / "received", root / "refined")
    else: evaluate(job, root / "refined")


if __name__ == "__main__":
    main()
