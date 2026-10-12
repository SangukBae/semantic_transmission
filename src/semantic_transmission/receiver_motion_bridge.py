"""Opt-in, received-reference motion bridge after Open-Sora decoding.

Inference reads only decoder metadata, received keys, and decoded frames. It does
not repair VAE latent placement, change captions, or train a model. On short,
camera-dominated intervals, affine registration places frames on the transmitted
timeline. The hybrid mode blends motion-compensated received references where
they agree; uncertain areas retain generated content. This is a hybrid receiver,
not a claim that Open-Sora itself has learned better motion.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from .artifacts import sha256, write_json

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN = REPO / "outputs/qwen35_pairs_zea_20261007/reconstruction/event_pairs/run"
DEFAULT_OUTPUT = REPO / "outputs/qwen35_pairs_zea_20261007_temporal_repair"
IDENTITY = np.eye(3, dtype=np.float32)
POLICY = "received_affine_motion_bridge_v1"


def register(reference, target):
    """Return a transform from reference coordinates to target coordinates."""
    h, w = reference.shape[:2]
    size = (min(w, 288), max(8, round(h * min(w, 288) / w)))
    def gray(im):
        value = cv2.cvtColor(cv2.resize(im, size), cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(value.astype(np.float32) / 255, (5, 5), 1)
    a, b = gray(reference), gray(target)
    if min(float(a.std()), float(b.std())) < .01:
        return 0., IDENTITY.copy()
    try:
        score, matrix = cv2.findTransformECC(
            a, b, IDENTITY[:2].copy(), cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), None, 3)
        scale = np.diag([size[0] / w, size[1] / h, 1])
        matrix = np.linalg.inv(scale) @ np.vstack([matrix, [0, 0, 1]]) @ scale
        if not np.isfinite(matrix).all() or not np.isfinite(score):
            return 0., IDENTITY.copy()
        return float(score), matrix.astype(np.float32)
    except cv2.error:
        return 0., IDENTITY.copy()


def plausible(matrix, shape, scale_bound=.18):
    h, w = shape[:2]
    scales = np.linalg.svd(matrix[:2, :2], compute_uv=False)
    return bool(np.linalg.det(matrix[:2, :2]) > 0
                and np.all(abs(scales - 1) <= scale_bound)
                and abs(matrix[0, 2]) <= .15 * w
                and abs(matrix[1, 2]) <= .15 * h)


def pair_plan(left, right, gap, fps):
    if gap / fps > 1.0:
        return dict(accepted=False, reason="gap_over_one_second", score=0., matrix=IDENTITY.tolist())
    score, matrix = register(left, right)
    reverse_score, reverse = register(right, left)
    h, w = left.shape[:2]
    corners = np.array([[0, 0, 1], [w-1, 0, 1], [0, h-1, 1], [w-1, h-1, 1]]).T
    cycle_error = float(np.linalg.norm((reverse @ matrix @ corners - corners)[:2], axis=0).max())
    accepted = (min(score, reverse_score) >= .94 and plausible(matrix, left.shape, .05)
                and cycle_error <= 2.)
    return dict(accepted=bool(accepted), reason="accepted" if accepted else "unreliable_camera_match",
                score=score, reverse_score=reverse_score, cycle_error_px=cycle_error,
                matrix=matrix.tolist())


def warp(im, matrix):
    h, w = im.shape[:2]
    return cv2.warpAffine(im, matrix[:2], (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def validity(shape, matrix):
    h, w = shape[:2]
    return cv2.warpAffine(np.ones((h, w), np.float32), matrix[:2], (w, h),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def edge_weight(frame, start, end, previous_ok, next_ok):
    # At a rejected neighboring interval meet its unchanged generated frame.
    ramp = min(4, (end - start) / 2)
    weight = 1.
    if not previous_ok:
        weight = min(weight, (frame - start) / ramp)
    if not next_ok:
        weight = min(weight, (end - frame) / ramp)
    return max(0., min(1., weight))


def repair_frame(generated, left, right, matrix, fraction, strength=1., anchor_weight=0.):
    desired = (1 - fraction) * IDENTITY + fraction * matrix
    score, observed = register(left, generated)
    corrected = generated
    if score >= .8 and plausible(observed, generated.shape):
        transform = desired @ np.linalg.inv(observed)
        full = warp(generated, transform)
        valid = validity(generated.shape, transform)[..., None] * strength
        corrected = np.rint(full * valid + generated * (1 - valid)).clip(0, 255).astype(np.uint8)
    l = warp(left, desired)
    right_transform = desired @ np.linalg.inv(matrix)
    r = warp(right, right_transform)
    difference = np.max(np.abs(cv2.GaussianBlur(l, (5, 5), 0).astype(np.float32)
                               - cv2.GaussianBlur(r, (5, 5), 0).astype(np.float32)), axis=2)
    alpha = np.clip((45 - difference) / 25, 0, 1)
    alpha = cv2.GaussianBlur(alpha, (15, 15), 3)
    # Never borrow reflected image borders as reference evidence.
    alpha *= (validity(left.shape, desired) > .999) & (validity(right.shape, right_transform) > .999)
    # Exact transmitted timestamps anchor to their own received frame. Fade this
    # additional confidence over 4 frames to avoid a one-frame endpoint flash.
    alpha += anchor_weight * (1 - alpha)
    alpha *= strength
    bridge = cv2.addWeighted(l, 1 - fraction, r, fraction, 0)
    hybrid = np.rint(bridge * alpha[..., None] + corrected * (1 - alpha[..., None])).clip(0, 255).astype(np.uint8)
    return corrected, hybrid, dict(generated_score=score, reference_weight=float(alpha.mean()))


class Progress:
    def __init__(self, output):
        self.output, self.start = output, time.monotonic()

    def show(self, phase, done, total, force=False):
        if not force and done % 20 and done != total:
            return
        percent = 100 * done / max(1, total)
        elapsed = time.monotonic() - self.start
        print(f"{phase}: {percent:6.2f}% ({done}/{total}) | elapsed {elapsed:.1f}s", flush=True)
        write_json(self.output / "status.json", dict(phase=phase, done=done, total=total,
                                                     percent=percent, elapsed_seconds=elapsed))


def load_inputs(run):
    receiver = run / "receiver"
    metadata = receiver / "decoder_inputs.json"
    value = json.loads(metadata.read_text())
    video, keys = value["video"], value["indices"]
    count, fps = video["frames"], video["fps"]
    if (count < 2 or not np.isfinite(fps) or fps <= 0 or len(keys) < 2
            or any(type(k) is not int for k in keys) or keys != sorted(set(keys))
            or keys[0] != 0 or keys[-1] != count-1):
        raise ValueError("requires increasing unique keys including both video endpoints")
    shape = (video["height"], video["width"], 3)
    folder = receiver / "reconstruction/sample_0000_frames"
    frames = sorted(folder.glob("*.png"), key=lambda p: int(p.stem))
    if [int(p.stem) for p in frames] != list(range(count)):
        raise ValueError("generated lossless frame timeline is incomplete")
    paths = [receiver / f"frames/sample/key_frames_received/{k}.png" for k in keys]
    images = [read_image(p, shape) for p in paths]
    return value, images, frames, [metadata, *paths, *frames]


def read_image(path, shape):
    image = cv2.imread(str(path))
    if image is None or image.shape != shape:
        raise ValueError(f"invalid frame or dimensions: {path}")
    return image


@contextmanager
def lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def encode(folder, target, fps, count, progress):
    temporary = target.with_suffix(".partial.mp4")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-framerate", str(fps), "-i", str(folder / "%05d.png"),
               "-frames:v", str(count), "-c:v", "libx264", "-preset", "slow", "-crf", "18",
               "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               "-progress", "pipe:1", str(temporary)]
    with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as proc:
        for line in proc.stdout:
            if line.startswith("frame="):
                progress.show(f"encode/{target.stem}", min(int(line.split("=")[1]), count), count, True)
        if proc.wait():
            raise RuntimeError(f"video encoding failed: {target}")
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height",
        "-of", "json", str(temporary)], text=True))["streams"][0]
    numerator, denominator = map(int, probe["r_frame_rate"].split("/"))
    if int(probe["nb_read_frames"]) != count or abs(numerator / denominator - fps) > 1e-6:
        raise ValueError("encoded output timeline mismatch")
    temporary.replace(target)


def execute(run, output, mode="both"):
    cv2.setNumThreads(2)
    run, output = Path(run).resolve(), Path(output).resolve()
    if output == run or output in run.parents or run in output.parents:
        raise ValueError("output must be separate from the frozen input run")
    if output.exists() and any(p.name != ".lock" for p in output.iterdir()) and not (output / "protocol.json").exists():
        raise ValueError("refusing an existing unowned output directory")
    # Invalid/incomplete inputs must not leave an unowned status directory that
    # would prevent retrying after the upstream decoder finishes.
    meta, received, frame_paths, inputs = load_inputs(run)
    with lock(output):
        progress = Progress(output)
        progress.show("validate", 0, 1, True)
        modes = ["geometry", "hybrid"] if mode == "both" else [mode]
        protocol = dict(policy=POLICY, mode=mode, run=str(run), metadata=meta,
                        code_sha256=sha256(__file__), opencv=cv2.__version__, numpy=np.__version__,
                        inputs={str(p.relative_to(run)): sha256(p) for p in inputs},
                        source_images_used_for_inference=False, additional_channel_uses=0,
                        generator_rerun=False, captions_changed=False, training_executed=False)
        path = output / "protocol.json"
        if path.exists() and json.loads(path.read_text()) != protocol:
            raise ValueError("frozen inputs/code/options changed; choose another output directory")
        write_json(path, protocol)
        result_path = output / "RESULT.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            for name, digest in result["artifacts"].items():
                if sha256(output / name) != digest:
                    raise ValueError(f"completed output changed: {name}")
            progress.show("COMPLETE_REUSED", 1, 1, True)
            return result
        keys, count, fps = meta["indices"], meta["video"]["frames"], meta["video"]["fps"]
        plans = []
        for j, (a, b) in enumerate(zip(keys, keys[1:])):
            plans.append(dict(start=a, end=b, **pair_plan(received[j], received[j+1], b-a, fps)))
            progress.show("register received pairs", j+1, len(keys)-1, True)
        for name in modes:
            (output / f"{name}_frames").mkdir(exist_ok=True)
        trace = []
        shape = (meta["video"]["height"], meta["video"]["width"], 3)
        for j, plan in enumerate(plans):
            a, b = plan["start"], plan["end"]
            for t in range(a, b + (j == len(plans)-1)):
                g = read_image(frame_paths[t], shape)
                detail = dict(generated_score=None, reference_weight=0.)
                geometry = hybrid = g
                if plan["accepted"]:
                    strength = edge_weight(t, a, b, j == 0 or plans[j-1]["accepted"],
                                           j == len(plans)-1 or plans[j+1]["accepted"])
                    geometry, hybrid, detail = repair_frame(
                        g, received[j], received[j+1], np.array(plan["matrix"], np.float32),
                        (t-a)/(b-a), strength, anchor_weight=max(0., 1-min(t-a, b-t)/4))
                for name in modes:
                    target = output / f"{name}_frames/{t:05d}.png"
                    temporary = target.with_suffix(".partial.png")
                    if not cv2.imwrite(str(temporary), geometry if name == "geometry" else hybrid):
                        raise RuntimeError(f"frame write failed: {target}")
                    temporary.replace(target)
                trace.append(dict(frame=t, pair=j, accepted=plan["accepted"], **detail))
                progress.show("repair frames", t+1, count)
        write_json(output / "trace.json", dict(pairs=plans, frames=trace))
        for name in modes:
            encode(output / f"{name}_frames", output / f"{name}.mp4", fps, count, progress)
        artifacts = [output / "trace.json"]
        for name in modes:
            artifacts.extend([output / f"{name}.mp4", *sorted((output / f"{name}_frames").glob("*.png"))])
        result = dict(status="COMPLETE_REPAIR_INDEPENDENT_REVIEW_PENDING", policy=POLICY,
                      elapsed_seconds=time.monotonic()-progress.start, frames=count, fps=fps,
                      accepted_pairs=sum(p["accepted"] for p in plans), pairs=len(plans),
                      mean_received_reference_weight=float(np.mean([r["reference_weight"] for r in trace])),
                      artifacts={str(p.relative_to(output)): sha256(p) for p in artifacts})
        write_json(result_path, result)
        progress.show("COMPLETE", count, count, True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("geometry", "hybrid", "both"), default="both")
    args = parser.parse_args()
    result = execute(args.run, args.output, args.mode)
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
