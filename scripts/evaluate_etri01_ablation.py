"""Frame-aligned ETRI01 quality, visible-coat trajectory proxy, and comparison assets."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.official_quality import OfficialMetrics, pixel_scores
from semantic_transmission.research_quality import read_video, read_frames
from semantic_transmission.temporal import unique_output_positions

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "outputs/diagnostics/etri01_ablation_20260917"
BASE = REPO / "outputs/etri01_official_20260911_v2/01_person_walk"
CASES = ["historical", "replay", "aligned", "clean_keys", "action_caption", "dense_2s", "dense_1s", "combined"]


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def person_proxy(frame):
    """Fixed preregistered navy-coat mask, valid only for this fixed-camera clip.

    No source tracking is injected into the reconstructed-video extraction.
    Outputs are a color/shape proxy, not a validated person detector or pose score.
    """
    rgb = frame.astype(np.int16)
    r, g, b = rgb.transpose(2, 0, 1)
    mask = ((b-r > 4) & (b-g >= -4) & (r < 110) & (b < 165)).astype(np.uint8)
    mask[:22] = 0
    mask[265:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 120
                  and stats[i, cv2.CC_STAT_HEIGHT] >= 45]
    if not candidates:
        return None
    i = max(candidates, key=lambda j: stats[j, cv2.CC_STAT_AREA])
    x, y, w, h, area = map(int, stats[i])
    return {"center_x": float(centers[i, 0]), "center_y": float(centers[i, 1]),
            "coat_box": [x, y, x+w, y+h], "area": area,
            "roi": [max(0, x-25), max(0, y-35), min(frame.shape[1], x+w+25), min(frame.shape[0], y+h+95)]}


def motion_summary(source_tracks, tracks):
    valid = [i for i, (a, b) in enumerate(zip(source_tracks, tracks)) if a and b]
    errors = [abs(source_tracks[i]["center_x"]-tracks[i]["center_x"]) for i in valid]
    def turning(values):
        # Median smoothing removes gait-related centroid oscillation; the maximum
        # position is a turn-location proxy, not the first body-orientation change.
        x = np.array([r["center_x"] if r else np.nan for r in values])
        smooth = np.array([np.nanmedian(x[max(0,i-6):min(len(x),i+7)]) for i in range(len(x))])
        return int(np.nanargmax(smooth)) if np.isfinite(smooth).any() else None
    a, b = turning(source_tracks), turning(tracks)
    return {"proxy": "fixed_navy_coat_color_component_not_validated_person_tracking",
            "detected_frames": sum(v is not None for v in tracks), "paired_frames": len(valid),
            "horizontal_mae_pixels": float(np.mean(errors)) if errors else None,
            "horizontal_mae_width_fraction": float(np.mean(errors)/576) if errors else None,
            "source_rightmost_frame": a, "reconstruction_rightmost_frame": b,
            "rightmost_timing_error_seconds": abs(a-b)/24 if a is not None and b is not None else None}


def load_case(name, boundary):
    run = BASE if name == "historical" else ROOT / name
    video = run / "receiver/reconstruction/sample_0000.mp4"
    values = read_frames(video.with_name("sample_0000_frames")) if boundary == "lossless_frames" else read_video(video)
    inputs = read(run / "receiver/decoder_inputs.json")
    policy = inputs["decoder"].get("concatenation_policy", "official_release")
    positions = unique_output_positions(inputs["indices"], policy)
    if len(values) != (241 if name in {"historical", "replay"} else 240):
        raise ValueError(f"Unexpected output frame count: {name} {len(values)}")
    return run, video, values[positions], positions


def contact(name, source, values, source_tracks, tracks):
    chosen = [0, 48, 96, 144, 168, 179, 204, 239]
    canvas = Image.new("RGB", (576*4, 352*4), "white")
    draw = ImageDraw.Draw(canvas)
    for k, i in enumerate(chosen):
        for row, (frames, ts, label) in enumerate([(source, source_tracks, "source"), (values, tracks, name)]):
            x, y = (k % 4)*576, (k//4*2+row)*352
            canvas.paste(Image.fromarray(frames[i]), (x, y+32))
            draw.text((x+5, y+8), f"{label}  frame={i} t={i/24:.3f}s", fill="black")
            if ts[i]:
                a,b,c,d = ts[i]["coat_box"]
                draw.rectangle((x+a,y+32+b,x+c,y+32+d), outline="yellow", width=2)
    canvas.save(ROOT / f"contact_{name}.jpg", quality=92)


def evaluate(name, source, source_tracks, *, pixel_only):
    path = ROOT / f"evaluation_{name}.json"
    if path.exists() and (pixel_only or read(path).get("all_five_metrics")):
        print(f"SKIP evaluation {name}", flush=True)
        return
    result = {"name": name, "reference": str(BASE / "data/normalized.mp4"),
              "reference_sha256": sha256(BASE / "data/normalized.mp4"),
              "all_five_metrics": not pixel_only, "timeline": "240 unique source frames; later duplicate boundary removed",
              "scope": "one development video, seed42; manual caption and clean-key variants are diagnostic oracles"}
    for boundary in ("lossless_frames", "delivered_mp4"):
        run, video, values, positions = load_case(name, boundary)
        result["video_sha256"] = sha256(video)
        result["kept_output_positions"] = positions
        if pixel_only:
            rows = [dict(frame=i, **pixel_scores(a,b)) for i,(a,b) in enumerate(zip(source, values))]
            summary = {key: float(np.mean([r[key] for r in rows])) for key in ("psnr_db", "ssim")}
        elif name in {"historical", "replay"}:
            # Verified identical videos/source: reuse the prior official five-metric
            # reevaluation, not the older run's incompatible LPIPS-Alex/RGB SSIM.
            previous = REPO / "outputs/lgvsc_alignment_20260917_v1/etri01"
            old = read(previous / "quality.json")
            assert old["video_sha256"] == result["video_sha256"]
            assert old["reference_sha256"] == result["reference_sha256"]
            with (previous / f"quality_{boundary}.csv").open() as stream:
                full = list(csv.DictReader(stream))
            keys = ["psnr_db", "ssim", "lpips_vgg", "clip", "dists"]
            rows = [dict(frame=i, **{k: float(full[p][k]) for k in keys}) for i,p in enumerate(positions)]
            summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
            summary.update(frames=240, shape=list(values.shape))
            result["official_metric_provenance"] = str(previous)
        else:
            summary, rows = OfficialMetrics().evaluate(source, values)
        result[boundary] = summary
        write_csv(ROOT / f"metrics_{name}_{boundary}.csv", rows)
        if boundary == "lossless_frames":
            tracks = [person_proxy(frame) for frame in values]
            result["motion_proxy"] = motion_summary(source_tracks, tracks)
            roi_rows = []
            for i, (a,b, track) in enumerate(zip(source, values, source_tracks)):
                if track:
                    x,y,u,v = track["roi"]
                    roi_rows.append(dict(frame=i, **pixel_scores(a[y:v,x:u],b[y:v,x:u])))
            result["source_person_roi"] = {key: float(np.mean([r[key] for r in roi_rows])) for key in ("psnr_db", "ssim")}
            result["source_person_roi"]["frames"] = len(roi_rows)
            result["source_person_roi"]["definition"] = "same source-derived coat bounding box + fixed margins in every variant"
            write_csv(ROOT / f"roi_{name}.csv", roi_rows)
            write_json(ROOT / f"tracking_{name}.json", {"source": source_tracks, "reconstruction": tracks})
            contact(name, source, values, source_tracks, tracks)
            indices = read(run / "receiver/decoder_inputs.json")["indices"]
            result["keyframe_scores"] = [dict(index=i, **pixel_scores(source[i], values[i])) for i in indices]
        del values
    result["channel"] = read(run / "channel_accounting.json")
    if name == "clean_keys":
        result["channel"] = {"total_complex_channel_uses": None, "reason": "clean input oracle bypasses transmission"}
    if name != "historical":
        result["experiment"] = read(run / "experiment.json")
    write_json(path, result)
    print(f"DONE evaluation {name}: {result['lossless_frames']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--pixel-only", action="store_true")
    args = parser.parse_args()
    source = read_video(BASE / "data/normalized.mp4")
    tracks = [person_proxy(frame) for frame in source]
    write_json(ROOT / "source_tracking.json", tracks)
    for name in args.cases:
        evaluate(name, source, tracks, pixel_only=args.pixel_only)


if __name__ == "__main__":
    main()
