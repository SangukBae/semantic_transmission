#!/usr/bin/env python3
"""Prepare auditable, continuous 60-second inputs; never run LGVSC inference.

Uses ffmpeg/ffprobe and Pillow/numpy (the existing lgvsc environment).
Selections are explicit JSON records, independent of reconstruction outputs.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/etri_long_video_20260924"
FILTER = "scale=576:320:force_original_aspect_ratio=decrease,pad=576:320:(ow-iw)/2:(oh-ih)/2,setsar=1"


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Command failed: {cmd!r}\n{result.stderr[-3000:]}")
    return result


def probe(path):
    return json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration:format=duration",
        "-of", "json", str(path)]).stdout)


def extract_tum(data):
    for name in ("tum_desk", "tum_household"):
        archive = data / "archives" / (name + ".tgz")
        target = data / "raw/tum"
        count = 0
        with tarfile.open(archive, "r:gz") as t:
            for m in t:
                p = Path(m.name)
                if not m.isfile() or p.is_absolute() or ".." in p.parts:
                    continue
                if p.name != "rgb.txt" and not (p.suffix == ".png" and p.parent.name == "rgb"):
                    continue
                dest = target / p
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    dest.write_bytes(t.extractfile(m).read())
                count += 1
        print(json.dumps({"archive": name, "rgb_files_and_index": count}), flush=True)


def scan_video(path, data, dataset="tvsum"):
    """Candidate cuts only; no claim of independent ground truth."""
    p = probe(path)
    if float(p["format"]["duration"]) < 70:
        return {"source_id": path.stem, "probe": p, "candidate_count": -1, "status": "TOO_SHORT_FOR_10_TO_70"}
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-threads", "2", "-filter_threads", "2", "-ss", "10",
        "-t", "60", "-i", str(path), "-an", "-vf",
        "scale=256:-2,select='gt(scene,0.18)',showinfo", "-vsync", "vfr", "-f", "null", "-"]
    r = run(cmd)
    times = [float(x) for x in re.findall(r"pts_time:([\d.]+)", r.stderr)]
    # Closely spaced fade/flash detections are grouped as a candidate event.
    grouped = []
    for t in times:
        if not grouped or t - grouped[-1] >= .5:
            grouped.append(round(t, 6))
    row = {"source_id": path.stem, "path": str(path.resolve()), "probe": p,
        "clip_start_sec": 10, "clip_end_sec": 70,
        "candidate_boundaries_clip_sec": grouped, "candidate_count": len(grouped),
        "boundary_status": "AUTOMATIC_CANDIDATES_UNREVIEWED"}
    save(data / "reports" / (dataset + "_scan") / (path.stem + ".json"), row)
    return row


def scan(data, dataset):
    files = sorted((data / "raw" / dataset).glob("*.mp4"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda p: scan_video(p, data, dataset), files))
    rows.sort(key=lambda r: (r["candidate_count"], r["source_id"]))
    save(data / "reports" / (dataset + "_scan.json"), rows)
    print(json.dumps([{k: r[k] for k in ("source_id", "candidate_count")} for r in rows]))


def tum_frames(source):
    entries = []
    for line in (source / "rgb.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        ts, name = line.split()
        entries.append((float(ts), source / name))
    assert all(b[0] > a[0] for a, b in zip(entries, entries[1:]))
    return entries


def build_tum(data, item, out):
    import numpy as np
    source = Path(item["source_path"])
    frames = tum_frames(source)
    ts = np.array([x[0] for x in frames])
    start = item["clip_start_sec"]
    target = ts[0] + start + np.arange(1440) / 24
    assert ts[-1] >= ts[0] + start + 60, "Source does not cover the complete interval"
    right = np.searchsorted(ts, target).clip(1, len(ts) - 1)
    idx = np.where(target - ts[right - 1] <= ts[right] - target, right - 1, right)
    # Retain unique source images despite occasional irregular sensor intervals.
    # This preserves a real 60-second span without repeating a captured frame.
    for k in range(1, len(idx)):
        idx[k] = max(idx[k], idx[k - 1] + 1)
    assert idx[-1] < len(ts)
    errors = np.abs(ts[idx] - target)
    assert errors.max() < .1, "Timestamp gap too large; choose another continuous interval"
    mapping = [{"output_frame": k, "output_time_sec": k / 24,
        "source_timestamp": float(ts[j]), "source_relative_sec": float(ts[j] - ts[0]),
        "source_png": str(frames[j][1].resolve())} for k, j in enumerate(idx)]
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "image2pipe", "-framerate", "24",
        "-vcodec", "png", "-threads", "2", "-filter_threads", "2", "-i", "pipe:0",
        "-vf", FILTER, "-frames:v", "1440", "-an", "-c:v", "libx264",
        "-threads", "2", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-r", "24", "-movflags", "+faststart", str(out)]
    log = data / "reports" / (item["id"] + "_encode.log")
    with log.open("w") as errors_log:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors_log)
        try:
            for row in mapping:
                proc.stdin.write(Path(row["source_png"]).read_bytes())
        finally:
            proc.stdin.close()
            code = proc.wait()
        if code:
            raise RuntimeError(log.read_text())
    save(data / "metadata" / (item["id"] + "_time_mapping.json"), mapping)
    return {"source_duration_sec": float(ts[-1] - ts[0]), "source_rgb_frames": len(ts),
        "source_sha256_kind": "rgb_timestamp_index", "source_sha256": sha(source / "rgb.txt"),
        "source_archive_sha256": item["source_archive_sha256"],
        "max_timestamp_sampling_error_sec": float(errors.max()),
        "resampling_repeated_source_frames": int(len(idx) - len(np.unique(idx))),
        "resampling_selection_policy": "nearest timestamp subject to strictly increasing source frame index",
        "time_mapping": str((data / "metadata" / (item["id"] + "_time_mapping.json")).resolve()),
        "ffmpeg_command": cmd}


def build(data, selections):
    rows = json.loads(selections.read_text())["videos"]
    seen_sources = set()
    result = []
    for item in rows:
        key = (item["dataset"], item["source_video_id"])
        assert key not in seen_sources, "Each split must use an independent source"
        seen_sources.add(key)
        assert item["clip_end_sec"] - item["clip_start_sec"] == 60
        out = data / "processed" / (item["id"] + ".mp4")
        out.parent.mkdir(parents=True, exist_ok=True)
        if item["dataset"] == "TUM_RGBD":
            record = build_tum(data, item, out)
        else:
            source = Path(item["source_path"])
            p = probe(source)
            duration = float(p["format"]["duration"])
            assert duration >= item["clip_end_sec"]
            cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", str(item["clip_start_sec"]),
                "-i", str(source), "-t", "60", "-an", "-vf", "fps=24," + FILTER,
                "-c:v", "libx264", "-threads", "2", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
            run(cmd)
            record = {"source_sha256": sha(source), "source_sha256_kind": "video_file",
                "source_duration_sec": duration, "source_probe": p, "ffmpeg_command": cmd,
                "time_mapping": {"source_time_sec": "clip_start_sec + output_frame / 24",
                    "sampling": "ffmpeg fps=24; original elapsed time preserved"}}
        p = probe(out)
        s = p["streams"][0]
        assert abs(float(p["format"]["duration"]) - 60) < 1 / 24
        assert int(s["nb_frames"]) == 1440 and s["avg_frame_rate"] == "24/1"
        decode = run(["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-i", str(out), "-f", "null", "-"])
        assert not decode.stderr.strip(), decode.stderr
        record.update(item)
        record.update(output_path=str(out.resolve()), output_sha256=sha(out), output_probe=p,
            preparation_status="PREPARED_DECODE_VERIFIED", reconstruction_status="NOT_STARTED",
            hallucination_annotation_status="NOT_STARTED_REQUIRES_RECONSTRUCTIONS")
        save(data / "reports" / (item["id"] + "_preparation.json"), record)
        result.append(record)
        print(json.dumps({"id": item["id"], "status": record["preparation_status"]}), flush=True)
    save(data / "manifest.json", {"schema_version": 1, "videos": result})
    import csv
    keys = ["id", "dataset", "source_video_id", "split", "scene_transition_level", "clip_start_sec",
        "clip_end_sec", "source_duration_sec", "output_path", "output_sha256", "boundary_review_status",
        "preparation_status", "reconstruction_status", "license_status"]
    with (data / "manifest.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(result)


def contact_sheet(path, output, times):
    from PIL import Image, ImageDraw
    import io
    width, height = 288, 180
    sheet = Image.new("RGB", (width * 4, height * ((len(times) + 3) // 4)), "#202020")
    draw = ImageDraw.Draw(sheet)
    for i, t in enumerate(times):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(max(0, t)), "-i", str(path),
            "-frames:v", "1", "-vf", "scale=288:156:force_original_aspect_ratio=decrease,pad=288:156:(ow-iw)/2:(oh-ih)/2",
            "-f", "image2pipe", "-vcodec", "png", "-"], check=True, capture_output=True).stdout
        frame = Image.open(io.BytesIO(raw)).convert("RGB")
        x, y = i % 4 * width, i // 4 * height
        sheet.paste(frame, (x, y)); draw.text((x + 6, y + 158), f"{t:.3f} s", fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["extract-tum", "scan-tvsum", "scan-clipshots", "build", "sheet"])
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--selections", type=Path)
    p.add_argument("--video", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--times", default=",".join(str(x) for x in range(0, 60, 5)))
    a = p.parse_args()
    if a.command == "extract-tum": extract_tum(a.data)
    elif a.command in ("scan-tvsum", "scan-clipshots"): scan(a.data, a.command[5:])
    elif a.command == "build": build(a.data, a.selections)
    else: contact_sheet(a.video, a.output, [float(x) for x in a.times.split(",")])


if __name__ == "__main__":
    main()
