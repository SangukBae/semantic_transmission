"""Build synchronized visual evidence and paired quality tables; no model inference."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

import cv2
from PIL import Image, ImageDraw, ImageFont

from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.video_io import probe


SAMPLES = {
    "people_10s": [217, 228, 234, 238, 240, 244, 250, 260, 266],
    "car_55s": [1303, 1305, 1311, 1315, 1319, 1323, 1327, 1331, 1334],
    "door_59s": [1400, 1404, 1409, 1413, 1416, 1421, 1427, 1433, 1439],
}
METRICS = ("psnr_db", "ssim", "lpips_vgg", "clip", "dists")


def frames(path):
    cap = cv2.VideoCapture(str(path))
    output = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return output
            output.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        cap.release()


def build(root, name, variant="no_rounding", variant_label="ALIGN NONE - restarted context"):
    pairs = [p for p in json.loads((root / "pairs.json").read_text()) if p["window"] == name]
    assert len(pairs) == 2
    a, b = pairs
    for field in ("source_sha256", "receiver_metadata_sha256", "received_keyframe_sha256"):
        assert a[field] == b[field], f"paired inputs differ: {field}"
    keys = a["original_keyframes"]
    start, end = a["original_frames_inclusive"]
    source = root / "inputs" / name / "source.mp4"
    cases = ("baseline", variant)
    runs = [root / name / case for case in cases]
    clips = [source, *[r / "receiver/reconstruction/sample_0000.mp4" for r in runs]]
    traces = [json.loads((r / "receiver/conditioning_trace.json").read_text())["mask"] for r in runs]
    assert len(traces[0]) == len(traces[1]) == len(keys) - 1
    assert all(x["noise_sha256"] == y["noise_sha256"] for x, y in zip(*traces)), "paired noise differs"
    for path in clips:
        info = probe(path)
        assert info["frames"] == end - start + 1 and info["fps"] == 24.0
    destination = root / name
    labels = ["SOURCE", "BASELINE - restarted context", variant_label]
    command = ["ffmpeg", "-v", "error", "-y", "-nostdin"]
    for path in clips:
        command += ["-threads", "2", "-i", str(path)]
    filters = []
    for i, label in enumerate(labels):
        filters.append(f"[{i}:v]pad=iw:ih+32:0:32:black,drawtext=text='{label}':x=8:y=6:fontsize=18:fontcolor=white[v{i}]")
    filters.append("[v0][v1][v2]hstack=inputs=3[out]")
    comparison = destination / "comparison.mp4"
    command += ["-filter_complex_threads", "1", "-filter_complex", ";".join(filters), "-map", "[out]",
                "-an", "-threads", "2", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(comparison)]
    subprocess.run(command, check=True)
    assert probe(comparison)["frames"] == end-start+1
    decoded = [frames(path) for path in clips]
    assert all(len(values) == end-start+1 for values in decoded)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    sheets = []
    samples = SAMPLES[name]
    for page, offset in enumerate(range(0, len(samples), 5)):
        selected = samples[offset:offset+5]
        canvas = Image.new("RGB", (1728, 32 + len(selected)*352), "#121922")
        draw = ImageDraw.Draw(canvas)
        for col, label in enumerate(labels):
            draw.text((col*576+8, 6), label, font=font, fill="white")
        for row, frame in enumerate(selected):
            y = 32+row*352
            draw.text((8, y+6), f"Original frame {frame} | {frame/24:.4f}s", font=font, fill="white")
            for col, values in enumerate(decoded):
                canvas.paste(values[frame-start], (col*576, y+32))
        filename = f"comparison_samples_{page:02d}.jpg"
        canvas.save(destination / filename, quality=94)
        sheets.append(filename)
    quality = []
    for run in runs:
        with (run / "quality_delivered_mp4.csv").open() as stream:
            quality.append(list(csv.DictReader(stream)))
    rows, summaries = [], {}
    for local in range(end-start+1):
        row = {"original_frame": local+start, "original_time_s": (local+start)/24}
        for i, case in enumerate(cases):
            assert int(quality[i][local]["frame"]) == local
            row.update({f"{case}_{m}": float(quality[i][local][m]) for m in METRICS})
        rows.append(row)
    for scope, selected in {"whole_window": rows,
                            "target_intermediate_frames": [r for r in rows if keys[2] < r["original_frame"] < keys[3]]}.items():
        means = {case: {m: sum(r[f"{case}_{m}"] for r in selected)/len(selected) for m in METRICS}
                 for case in cases}
        summaries[scope] = {"frames": len(selected), "metrics": means,
                            "delta_after_minus_before": {m: means[variant][m]-means["baseline"][m] for m in METRICS}}
    with (destination / "paired_quality.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(destination / "evidence.json", {
        "window": name, "original_frames_inclusive": [start, end],
        "scope": "Identically restarted short contexts, not full-run trajectory replay",
        "identical_received_inputs": True, "identical_initial_noise": True,
        "quality": summaries, "contact_sheet_frames": samples, "contact_sheets": sheets,
        "comparison": {"path": str(comparison), "sha256": sha256(comparison), "video": probe(comparison)},
        "semantic_review": "Contact sheets are evidence for a separate AI review, not automatic semantic labels"})
    print(name, json.dumps(summaries), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/etri_conditioning_diagnosis_20260928")
    parser.add_argument("--window", choices=list(SAMPLES))
    parser.add_argument("--variant", default="no_rounding")
    parser.add_argument("--variant-label", default="ALIGN NONE - restarted context")
    args = parser.parse_args()
    for name in [args.window] if args.window else SAMPLES:
        build(args.output.resolve(), name, args.variant, args.variant_label)
