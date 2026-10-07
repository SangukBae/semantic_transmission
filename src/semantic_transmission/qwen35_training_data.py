"""Source-grouped, hash-frozen data for the small Qwen multi-task pilot.

These are reviewed AI annotations, never human ground truth. Evaluation
originals are excluded before any window is sampled. No Qwen output is used
to construct the reference annotations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "outputs/fc_lgvsc_webvid_tvsum_full_20261001"
DEFAULT_DATA = ROOT / "outputs/qwen35_multitask_data_v1"
PILOT = {
    "train": [5, 6, 7, 10, 12, 13, 14, 16, 17, 21, 25, 26, 28, 36, 39, 51],
    "validation": [1, 24, 32, 47],
    "test": [19, 30, 31, 48],
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def exclusion_keys():
    ids, hashes = set(), set()
    evidence = []
    for rel, key in [
        ("data/etri_benchmark_v1_20260924/manifest.csv", "source_id"),
        ("data/etri_long_video_20260924/manifest.csv", "source_video_id"),
    ]:
        path = ROOT / rel
        for r in csv.DictReader(path.open()):
            ids.add(r[key])
            if r.get("source_sha256"):
                hashes.add(r["source_sha256"])
        evidence.append({"path": rel, "sha256": sha256(path)})
    path = ROOT / "configs/webvid5_manifest.json"
    for r in read_json(path)["videos"]:
        ids.add(Path(r["filename"]).stem)
        hashes.add(r["raw_sha256"])
    evidence.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    return ids, hashes, evidence


def prepare(out=DEFAULT_DATA):
    from PIL import Image, ImageDraw
    out = Path(out)
    if (out / "freeze.json").exists():
        raise ValueError("Dataset is frozen; use a new versioned output directory")
    ids, hashes, exclusion_evidence = exclusion_keys()
    inventory = read_json(INVENTORY / "inventory.json")["videos"]
    rows, sources = [], []
    assigned_groups = {}
    for v in inventory:
        excluded = v["source_id"] in ids or v["source_sha256"] in hashes
        split = "excluded_evaluation" if excluded else "reserve_unreviewed"
        n = int(v["id"].split("_")[1])
        if v["dataset"] == "WebVid" and not excluded:
            split = next((s for s, ns in PILOT.items() if n in ns), split)
        group = v["source_sha256"]
        if group in assigned_groups and assigned_groups[group] != split:
            raise ValueError(f"Duplicate source across splits: {v['id']}")
        assigned_groups[group] = split
        if sha256(v["source_path"]) != group:
            raise ValueError(f"Original changed: {v['id']}")
        sources.append({k: v[k] for k in ["id", "dataset", "source_id", "source_path", "source_sha256"]} | {"split": split})
        if split not in PILOT:
            continue
        folder = INVENTORY / v["dataset"].lower() / v["id"]
        prepared = read_json(folder / "source_prepared.json")
        if sha256(folder / "normalized.mp4") != prepared["normalized_sha256"]:
            raise ValueError("Normalized video changed")
        count = len(prepared["source_frame_hashes"])
        sheet = Image.new("RGB", (8 * 288, 4 * 184), "white")
        draw = ImageDraw.Draw(sheet)
        for w in range(4):
            a, b = w * count // 4, (w + 1) * count // 4
            indices = [a + round(j * (b - a - 1) / 7) for j in range(8)]
            frames = []
            for j, idx in enumerate(indices):
                path = folder / "source_frames" / f"{idx}.png"
                digest = sha256(path)
                if digest != prepared["source_frame_hashes"][str(idx)]:
                    raise ValueError(f"Frame changed: {path}")
                frames.append({"index": idx, "path": str(path), "sha256": digest})
                with Image.open(path) as im:
                    sheet.paste(im.convert("RGB").resize((288, 160)), (j * 288, w * 184 + 24))
                draw.text((j * 288 + 3, w * 184 + 4), f"w{w} i{j} f{idx} {idx / 24:.2f}s", fill="black")
            rows.append({"id": f"{v['id']}_w{w}", "source_id": v["source_id"],
                         "source_sha256": group, "source_video": v["source_path"],
                         "normalized_video": str(folder / "normalized.mp4"),
                         "normalized_sha256": prepared["normalized_sha256"],
                         "split": split, "fps": 24, "start": a, "end_exclusive": b,
                         "frames": frames, "video_id": v["id"], "window": w})
        path = out / "review" / f"{v['id']}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path, quality=95)
    write_json(out / "source_split.json", {"sources": sources, "exclusion_evidence": exclusion_evidence,
        "policy": "Original source ID AND SHA256 grouped; all known ETRI/long/WebVid5 evaluation originals excluded",
        "source_counts": dict(Counter(x["split"] for x in sources))})
    write_json(out / "windows.json", {"records": rows, "sampling": "Four disjoint quarters, eight chronological endpoint-inclusive images each; independent of annotations"})
    print(json.dumps({"sources": dict(Counter(x["split"] for x in sources)), "windows": len(rows)}))


def validate_record(row):
    n = len(row["frames"])
    if n != 8 or len({x["index"] for x in row["frames"]}) != n:
        raise ValueError("Expected eight distinct frames")
    indices = [x["index"] for x in row["frames"]]
    if indices != sorted(indices) or not row["start"] <= indices[0] < indices[-1] < row["end_exclusive"]:
        raise ValueError("Invalid frame chronology")
    if not row.get("caption") or not row.get("facts") or row.get("review_status") != "AI_SOURCE_REVIEWED":
        raise ValueError("Unreviewed or empty labels")
    for fact in row["facts"]:
        if not fact["text"] or not fact["evidence"] or not all(0 <= i < n for i in fact["evidence"]):
            raise ValueError("Fact lacks source evidence")
    for event in row["events"]:
        if not event["before"] or not event["after"]:
            raise ValueError("Event must have both sides")
        if not all(0 <= i < n for i in event["before"] + event["after"]) or max(event["before"]) >= min(event["after"]):
            raise ValueError("Invalid event chronology")
        if not row["start"] <= event["start_frame"] <= event["end_frame"] < row["end_exclusive"]:
            raise ValueError("Invalid event interval")


def freeze(out=DEFAULT_DATA, annotations=None):
    out = Path(out)
    if (out / "freeze.json").exists():
        return verify(out)
    annotations = Path(annotations or ROOT / "configs/qwen35_training_annotations_v1.json")
    reference = read_json(annotations)
    labels = reference["videos"]
    rows = read_json(out / "windows.json")["records"]
    for row in rows:
        a = labels[str(int(row["video_id"].split("_")[1]))][row["window"]]
        row["caption"] = a["caption"]
        row["facts"] = [{"id": f"{row['id']}_f{i}", "text": f[0], "evidence": f[1]} for i, f in enumerate(a["facts"])]
        row["events"] = [{"id": f"{row['id']}_e{i}", "description": e[0], "before": e[1], "after": e[2],
            "start_frame": row["frames"][max(e[1])]["index"], "end_frame": row["frames"][min(e[2])]["index"],
            "timing_type": "observed_before_after_bracket_not_exact_onset_offset"} for i, e in enumerate(a["events"])]
        row["review_status"] = "AI_SOURCE_REVIEWED"
        row["review_sheet_sha256"] = sha256(out / "review" / f"{row['video_id']}.jpg")
        validate_record(row)
    write_json(out / "annotations_source.json", reference)
    for split in PILOT:
        write_json(out / f"{split}.json", {"records": [r for r in rows if r["split"] == split]})
    files = ["source_split.json", "windows.json", "annotations_source.json", "train.json", "validation.json", "test.json"]
    write_json(out / "freeze.json", {"schema_version": 1, "files": {p: sha256(out / p) for p in files},
        "annotation_provenance": reference["provenance"], "independent_human_truth": False,
        "counts": {s: {"sources": len({r["source_sha256"] for r in rows if r["split"] == s}),
                       "windows": sum(r["split"] == s for r in rows),
                       "events": sum(len(r["events"]) for r in rows if r["split"] == s),
                       "facts": sum(len(r["facts"]) for r in rows if r["split"] == s)} for s in PILOT}})
    return verify(out)


def verify(out=DEFAULT_DATA):
    out = Path(out)
    frozen = read_json(out / "freeze.json")
    for p, digest in frozen["files"].items():
        if sha256(out / p) != digest:
            raise ValueError(f"Frozen manifest changed: {p}")
    excluded_ids, excluded_hashes, current_evidence = exclusion_keys()
    if current_evidence != read_json(out / "source_split.json")["exclusion_evidence"]:
        raise ValueError("Evaluation registry changed; prepare a new dataset version")
    groups, source_ids, checked = {}, {}, set()
    for split in PILOT:
        for row in read_json(out / f"{split}.json")["records"]:
            validate_record(row)
            group = row["source_sha256"]
            if row["split"] != split or group in excluded_hashes or row["source_id"] in excluded_ids:
                raise ValueError("Evaluation contamination or split mismatch")
            if groups.setdefault(group, split) != split or source_ids.setdefault(row["source_id"], split) != split:
                raise ValueError("Source overlap across splits")
            for p, digest in [(row["source_video"], group), (row["normalized_video"], row["normalized_sha256"])] + [(f["path"], f["sha256"]) for f in row["frames"]]:
                if p not in checked and sha256(p) != digest:
                    raise ValueError(f"Source bytes changed: {p}")
                checked.add(p)
            sheet = out / "review" / f"{row['video_id']}.jpg"
            if sha256(sheet) != row["review_sheet_sha256"]:
                raise ValueError("Review evidence changed")
    return {"status": "PASS", "freeze_sha256": sha256(out / "freeze.json"), "counts": frozen["counts"], "checked_files": len(checked)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "freeze", "verify"])
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--annotations", type=Path)
    a = p.parse_args()
    if a.command == "prepare":
        prepare(a.data)
    else:
        print(json.dumps(freeze(a.data, a.annotations) if a.command == "freeze" else verify(a.data), indent=2))


if __name__ == "__main__":
    main()
