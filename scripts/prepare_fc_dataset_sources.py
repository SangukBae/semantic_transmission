#!/usr/bin/env python3
"""Prepare full local WebVid/TVSum source evidence without changing frozen runs.

This stage does not invent keyframes or captions. Six-fps contact sheets are
review inputs, not proof that review or semantic extraction has completed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import subprocess

from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.webvid5 import fingerprint, read_json

REPO = Path(__file__).resolve().parents[1]
DEFAULT = REPO / "outputs/fc_lgvsc_webvid_tvsum_full_20261001"
ROOTS = {"WebVid": REPO / "data/webvid55/raw",
         "TVSum": REPO / "data/etri_long_video_20260924/raw/tvsum"}
RECIPE = dict(width=576, height=320, fps=24, source_start_seconds=0,
              source_end="EOF", duration_limit=None, overview_stride=4,
              normalization="scale=576:320,fps=24; libx264 crf18 medium yuv420p; no audio")


def probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
        "-show_format", "-of", "json", str(path)], text=True))


def inventory(root):
    if (root / "inventory.json").exists():
        saved = read_json(root / "inventory.json")
        if saved["checksum"] != fingerprint({k:v for k,v in saved.items() if k != "checksum"}):
            raise ValueError("inventory checksum changed")
        expected = {(name, str(p.resolve())) for name, folder in ROOTS.items()
                    for p in folder.rglob("*") if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}}
        if expected != {(r["dataset"],r["source_path"]) for r in saved["videos"]}:
            raise ValueError("local source inventory changed; use a new output root")
        return saved
    benchmark = read_json(REPO / "data/etri_benchmark_v1_20260924/manifest.json")
    records = []
    for dataset, folder in ROOTS.items():
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
                continue
            info = probe(path)
            stream = info["streams"][0]
            digest = sha256(path)
            duration = float(stream.get("duration", info["format"]["duration"]))
            records.append(dict(id=f"{dataset.lower()}_{len([r for r in records if r['dataset']==dataset])+1:03d}_{digest[:8]}",
                dataset=dataset, source_id=path.stem, source_path=str(path.resolve()),
                source_sha256=digest, source_duration_seconds=duration,
                source_probe=info, benchmark_memberships=[dict(id=b["id"], split=b["split"])
                    for b in benchmark if b.get("source_sha256")==digest]))
    payload = dict(version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        scope="All locally stored original WebVid and TVSum videos, full duration, no 16/60-second truncation",
        videos=records, normalization=RECIPE,
        extraction_status="SOURCE_PREPARATION_ONLY", independent_ground_truth=False,
        hallucination_mitigation_verified=False, reconstruction_started=False)
    result = dict(payload, checksum=fingerprint(payload))
    write_json(root / "inventory.json", result)
    return result


def prepare_one(root, row):
    import cv2
    from PIL import Image, ImageDraw
    directory = root / row["dataset"].lower() / row["id"]
    directory.mkdir(parents=True, exist_ok=True)
    complete = directory / "source_prepared.json"
    if sha256(row["source_path"]) != row["source_sha256"]:
        raise ValueError(f"source bytes changed: {row['id']}")
    if complete.exists():
        saved = read_json(complete)
        if saved["source_sha256"] != row["source_sha256"] or saved["recipe"] != RECIPE:
            raise ValueError("source preparation identity changed")
        if sha256(saved["normalized_video"]) != saved["normalized_sha256"]:
            raise ValueError("normalized source changed")
        for index, digest in saved["source_frame_hashes"].items():
            if sha256(Path(saved["source_frames"]) / f"{index}.png") != digest:
                raise ValueError(f"prepared frame changed: {row['id']} {index}")
        for relative, digest in saved["evidence"].items():
            if sha256(directory / relative) != digest:
                raise ValueError("source overview changed")
        print(f"REUSED_SOURCE {row['id']} frames={saved['frames']}", flush=True)
        return saved
    video = directory / "normalized.mp4"
    if not video.exists():
        temporary = directory / "normalized.partial.mp4"
        command = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-threads", "2",
            "-i", row["source_path"], "-map", "0:v:0", "-vf", "scale=576:320,fps=24",
            "-an", "-c:v", "libx264", "-threads", "2", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
        subprocess.run(command, check=True)
        temporary.replace(video)
    frames = directory / "source_frames"
    sheets = directory / "observations"
    frames.mkdir(exist_ok=True)
    sheets.mkdir(exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    count, hashes, observations = 0, {}, []
    page = None
    sampled = []
    def append_observation(index, frame):
        nonlocal page
        n = len(sampled)
        if n % 72 == 0:
            page = Image.new("RGB", (1536, 32+9*132), "#202020")
            ImageDraw.Draw(page).text((6,8), f"{row['id']} | SOURCE | 6 fps | NOT REVIEWED", fill="white")
        position = n % 72
        x, y = position % 8 * 192, 32+position // 8 * 132
        image = Image.fromarray(cv2.cvtColor(cv2.resize(frame,(192,107)),cv2.COLOR_BGR2RGB))
        page.paste(image,(x,y+23))
        ImageDraw.Draw(page).text((x+3,y+3), f"f{index} {index/24:.3f}s", fill="white")
        sampled.append(index)
        if position == 71:
            target=sheets / f"{n//72:04d}.jpg"
            page.save(target,quality=94)
            observations.append(target)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            target = frames / f"{count}.png"
            if not target.exists():
                if not cv2.imwrite(str(target),frame):
                    raise RuntimeError("PNG write failed")
            else:
                import numpy as np
                if not np.array_equal(cv2.imread(str(target)),frame):
                    raise ValueError(f"partial source PNG mismatch: {target}")
            hashes[str(count)] = sha256(target)
            if count % 4 == 0:
                append_observation(count,frame)
            last_frame=frame
            count+=1
    finally:
        capture.release()
    if count < 2 or abs(count/24-row["source_duration_seconds"]) > max(.15, 2/24):
        raise ValueError(f"full-duration check failed: {row['id']} {count/24} vs {row['source_duration_seconds']}")
    if count-1 not in sampled:
        append_observation(count-1,last_frame)
    if len(sampled)%72:
        target=sheets / f"{(len(sampled)-1)//72:04d}.jpg"
        page.save(target,quality=94)
        observations.append(target)
    if set(p.stem for p in frames.glob("*.png")) != set(hashes):
        raise ValueError("unexpected source-frame inventory")
    prepared = dict(id=row["id"], dataset=row["dataset"], source_id=row["source_id"],
        source_path=row["source_path"],source_sha256=row["source_sha256"],
        source_duration_seconds=row["source_duration_seconds"],
        normalized_video=str(video),normalized_sha256=sha256(video),
        source_frames=str(frames),source_frame_hashes=hashes,frames=count,fps=24,width=576,height=320,
        recipe=RECIPE, sampled_indices=sampled,
        evidence={str(p.relative_to(directory)):sha256(p) for p in observations},
        extraction_status="SOURCE_PREPARED_CANDIDATES_AND_CAPTIONS_PENDING",
        review_status="NOT_REVIEWED", reconstruction_started=False,
        benchmark_memberships=row["benchmark_memberships"])
    write_json(complete,prepared)
    print(f"PREPARED_SOURCE {row['id']} frames={count} seconds={count/24:.3f} sheets={len(observations)}",flush=True)
    return prepared


def report(root, manifest):
    rows=[]
    for row in manifest["videos"]:
        relative=Path(row["dataset"].lower()) / row["id"]
        ready=(root / relative / "source_prepared.json").exists()
        rows.append(dict(id=row["id"],dataset=row["dataset"],source_name=Path(row["source_path"]).name,
                         seconds=row["source_duration_seconds"],prepared=ready,path=str(relative)))
    write_json(root / "preparation_status.json", dict(videos=len(rows), sources_prepared=sum(r["prepared"] for r in rows),
        stage="SOURCE_PREPARATION_ONLY", rows=rows, updated_utc=datetime.now(timezone.utc).isoformat()))
    table=''.join(f'<tr><td>{r["dataset"]}</td><td>{html.escape(r["source_name"])}</td><td>{r["seconds"]:.2f}</td>'
        f'<td>{"SOURCE PREPARED" if r["prepared"] else "PENDING"}</td><td><a href="{r["path"]}/source_prepared.json">source record</a></td></tr>' for r in rows)
    (root / "sources.html").write_text('<!doctype html><html><meta charset="utf-8"><title>FC-LGVSC full dataset sources</title>'
        '<style>body{font:15px system-ui;margin:30px}td{padding:6px;border:1px solid #ccc}table{border-collapse:collapse}</style>'
        '<h1>WebVid / TVSum full source preparation</h1><p>Source sheets are not selected keyframes or reviewed captions.</p>'
        f'<p>{len(rows)} original videos; {sum(r["seconds"] for r in rows)/60:.2f} minutes. Full duration, 24 fps, 576x320.</p>'
        '<table><tr><th>Dataset</th><th>Original video</th><th>Seconds</th><th>Preparation</th><th>Evidence</th></tr>'+table+'</table></html>')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["inventory","prepare","status"])
    parser.add_argument("--output",type=Path,default=DEFAULT)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--ids",nargs="*")
    args=parser.parse_args()
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=True)
    manifest=inventory(root)
    if args.action=="prepare":
        import cv2
        from semantic_transmission.etri_60s_check import lock
        cv2.setNumThreads(1)
        selected=[r for r in manifest["videos"] if not args.ids or r["id"] in args.ids]
        if args.ids and len(selected)!=len(set(args.ids)):
            raise ValueError("unknown or duplicate requested source IDs")
        with lock(root / ".prepare.lock"), ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(prepare_one,root,row) for row in selected]
            from concurrent.futures import as_completed
            errors=[]
            for future in as_completed(futures):
                try: future.result()
                except Exception as exc:
                    print(f"SOURCE_FAILED {type(exc).__name__}: {exc}",flush=True)
                    errors.append(str(exc))
                report(root,manifest)
            if errors:
                write_json(root / "preparation_errors.json",errors)
                raise RuntimeError(f"{len(errors)} source preparations failed")
    report(root,manifest)
    print(json.dumps({k:v for k,v in read_json(root / 'preparation_status.json').items() if k!='rows'}),flush=True)


if __name__=="__main__":
    main()
