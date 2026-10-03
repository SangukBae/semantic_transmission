"""Read-only dataset suitability audit; never changes the frozen ETRI split.

Remote ZIP reads use bounded HTTP ranges rather than downloading all videos.
Summary annotations are not reconstruction-error ground truth.
"""
import argparse
from collections import Counter, defaultdict
import concurrent.futures
import csv
import datetime
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import urllib.request
import zipfile
import zlib

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "outputs/etri_public_summary_fit_20260929"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class RemoteZip(io.RawIOBase):
    """Seekable remote file with validated, cached 1 MiB HTTP range blocks."""
    def __init__(self, root, name):
        self.root, self.name, self.position = root, name, 0
        self.meta = json.loads((root / "sources/availability.json").read_text())[name]
        headers = {k.lower(): v for k, v in self.meta["headers"].items()}
        self.length = int(headers["content-range"].split("/")[-1])
        self.tail = (root / f"sources/{name}_tail.bin").read_bytes()
        self.cache = root / f"sources/ranges/{name}"
        self.cache.mkdir(parents=True, exist_ok=True)

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        self.position = (0 if whence == 0 else self.position if whence == 1 else self.length) + offset
        if self.position < 0: raise ValueError("negative seek")
        return self.position

    def read(self, size=-1):
        if size < 0: size = self.length - self.position
        size = min(size, self.length - self.position)
        if size <= 0: return b""
        pieces = []
        while size:
            if self.position >= self.length - len(self.tail):
                chunk = self.tail[self.position - self.length + len(self.tail):][:size]
            else:
                a = self.position // (1 << 20) * (1 << 20)
                b = min(self.length, a + (1 << 20)) - 1
                path = self.cache / f"{a}-{b}.bin"
                if not path.exists():
                    url = self.meta["url"] + f"?audit_range={a}-{b}"
                    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
                    with urllib.request.urlopen(req, timeout=40) as response:
                        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {a}-{b}/{self.length}":
                            raise ValueError("server did not honor requested byte range")
                        block = response.read(b - a + 2)
                    if len(block) != b - a + 1: raise ValueError("truncated byte range")
                    path.write_bytes(block)
                chunk = path.read_bytes()[self.position - a:][:size]
            pieces.append(chunk)
            self.position += len(chunk)
            size -= len(chunk)
        return b"".join(pieces)


def inventories(root):
    result = {}
    for name in ("summe_zip", "vsumm_user", "vsumm_video"):
        with zipfile.ZipFile(RemoteZip(root, name)) as z:
            result[name] = [{"name": p.filename, "bytes": p.file_size,
                             "compressed_bytes": p.compress_size, "offset": p.header_offset,
                             "crc32": p.CRC}
                            for p in z.infolist() if not p.is_dir()]
    write(root / "archive_inventory.json", result)
    return result


def probe(path, count=False):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if count: cmd.append("-count_frames")
    cmd += ["-show_entries", "stream=width,height,avg_frame_rate,nb_frames,nb_read_frames,duration:format=duration",
            "-of", "json", str(path)]
    result = subprocess.run(cmd, text=True, capture_output=True, check=True)
    value = json.loads(result.stdout)
    if result.stderr:
        log = ROOT / "logs" / f"{path.name}_{'decode' if count else 'header'}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(result.stderr)
        value["decoder_diagnostics"] = {"log": str(log.relative_to(ROOT)),
                                         "lines": len(result.stderr.splitlines())}
    return value


def save_member(root, archive, member, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with zipfile.ZipFile(RemoteZip(root, archive)) as z:
            destination.write_bytes(z.read(member))
    return {"archive": archive, "member": member, "path": str(destination.relative_to(root)),
            "bytes": destination.stat().st_size, "sha256": digest(destination)}


def audit(root):
    import numpy as np
    import scipy.io
    dataset = Path(json.loads((REPO / ".local/datasets.json").read_text())["root"])
    source = dataset / "etri_long_video_20260924"
    manifest_path = dataset / "etri_benchmark_v1_20260924/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    inventory = inventories(root)
    receipts = json.loads((root / "sample_downloads.json").read_text())
    for row in receipts:
        path = root / row["path"]
        entry = next(x for x in inventory[row["archive"]] if x["name"] == row["member"])
        assert digest(path) == row["sha256"]
        assert zlib.crc32(path.read_bytes()) == entry["crc32"]
    result = {"status": "DATASET_SUITABILITY_AUDIT_COMPLETE", "scope": "Metadata, labels and bounded CPU decoding only; no new GPU reconstruction or independent semantic-error annotation",
              "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "official_sources": {"TVSum": "https://github.com/yalesong/tvsum",
                  "VSUMM": "https://github.com/sandraavila/vsumm",
                  "SumMe_archive": "https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip",
                  "SumMe_paper": "https://www.varcity.ethz.ch/paper/eccv2014_gygli_vidsum.pdf"},
              "sample_download_crc_verified": True, "new_reconstruction_validation": False,
              "annotation_selection_is_oracle_baseline": True,
              "requirements_sha256": digest(REPO / "docs/ETRI_FOLLOWUP_EMAIL_SUMMARY.md"),
              "benchmark_manifest_sha256": digest(manifest_path), "datasets": {}}
    # Inspect all 50 annotation shapes, without using held-out scores to tune a selector.
    annotation = source / "metadata/tvsum_archive_ydata-tvsum50-data.zip"
    labels = defaultdict(list)
    with zipfile.ZipFile(annotation) as archive:
        for row in csv.reader(io.StringIO(archive.read("data/ydata-tvsum50-anno.tsv").decode()), delimiter="\t"):
            values = np.fromstring(row[2], dtype=np.int16, sep=",")
            assert values.min() >= 1 and values.max() <= 5
            labels[row[0]].append(len(values))
    assert len(labels) == 50 and all(len(v) == 20 for v in labels.values())
    paths = sorted((source / "raw/tvsum").glob("*.mp4"))
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        probes = dict(pool.map(lambda p: (p.stem, probe(p)), paths))
    write(root / "tvsum_source_probes.json", probes)
    mismatches = []
    for name, p in probes.items():
        count = int(p["streams"][0]["nb_frames"])
        if sorted(set(labels[name])) != [count]:
            mismatches.append({"video": name, "container_frames": count,
                               "annotation_frames": sorted(set(labels[name]))})
    primary = [r for r in manifest if r["dataset"] == "TVSum" and r["frames"] == 1440]
    def normalized(row):
        path = Path(row["processed_path"])
        assert digest(path) == row["processed_sha256"]
        value = probe(path, True)
        stream = value["streams"][0]
        assert int(stream["nb_read_frames"]) == 1440
        assert Fraction(stream["avg_frame_rate"]) == 24
        assert float(value["format"]["duration"]) == 60
        assert "decoder_diagnostics" not in value
        return row["id"], value
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        normalized_probes = dict(pool.map(normalized, primary))
    write(root / "tvsum_normalized_decode.json", normalized_probes)
    durations = [float(p["format"]["duration"]) for p in probes.values()]
    result["datasets"]["TVSum"] = {
        "videos": len(probes), "header_duration_range_sec": [min(durations), max(durations)],
        "header_at_least_60s": sum(v >= 60 for v in durations), "human_annotations": 1000,
        "annotation_type": "2-second importance judgments expanded to a per-frame score axis; not ready-made reconstruction keyframes",
        "annotation_source_sha256": digest(annotation), "frame_count_mismatches": mismatches,
        "normalized_60s_full_decode_passed": len(normalized_probes),
        "existing_scene_candidate_groups": dict(Counter(r["level"] for r in primary)),
        "scene_groups_are_independent_ground_truth": False,
        "role": "Preferred existing source-video set with ClipShots; importance-based selection is an unvalidated oracle baseline",
        "ready_to_replace_skem": False}
    # SumMe stores segment-membership IDs, not just 0/1; gt_score uses >0.
    rows = []
    for path in sorted((root / "sources/summe").glob("*.mat")):
        d = scipy.io.loadmat(path, simplify_cells=True)
        scores = np.asarray(d["user_score"])
        assert scores.shape[0] == int(d["nFrames"])
        assert np.allclose(np.mean(scores > 0, axis=1), d["gt_score"])
        rows.append({"video": path.stem, "frames": int(d["nFrames"]), "fps": float(d["FPS"]),
                     "metadata_duration_sec": float(d["video_duration"]),
                     "label_axis_duration_sec": int(d["nFrames"]) / float(d["FPS"]),
                     "annotators": scores.shape[1], "max_segment_membership_id": int(scores.max()),
                     "sha256": digest(path)})
    assert len(rows) == 25
    write(root / "summe_annotation_audit.json", rows)
    sample_paths = (list((root / "sources/vsumm").glob("*.mpg")) +
                    [p for p in (root / "sources/summe").iterdir() if p.suffix in (".mp4", ".webm")])
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        sample_probes = dict(pool.map(lambda p: (str(p.relative_to(root)), probe(p, True)), sample_paths))
    write(root / "sample_video_probes.json", sample_probes)
    summe_samples = {}
    for path, value in sample_probes.items():
        if "/summe/" not in path: continue
        gt = next(row for row in rows if row["video"] == Path(path).stem)
        stream = value["streams"][0]
        summe_samples[Path(path).name] = {"video_duration_sec": float(value["format"]["duration"]),
            "label_frames": gt["frames"], "container_frames": int(stream["nb_frames"]) if "nb_frames" in stream else None,
            "decoded_frames": int(stream["nb_read_frames"]),
            "resolution": [stream["width"], stream["height"]], "fps": stream["avg_frame_rate"],
            "decoder_diagnostics": value.get("decoder_diagnostics")}
    result["datasets"]["SumMe"] = {"videos": 25,
        "metadata_at_least_60s": sum(r["metadata_duration_sec"] >= 60 for r in rows),
        "label_axis_at_least_60s": sum(r["label_axis_duration_sec"] >= 60 for r in rows),
        "under_60s_metadata": [r["video"] for r in rows if r["metadata_duration_sec"] < 60],
        "annotation_type": "Human-selected temporal segments and frame membership; not individual reconstruction anchor images",
        "sample_decodes": summe_samples, "role": "Conditional auxiliary source-video set after length, decode and time-axis QA",
        "ready_to_replace_skem": False}
    groups = defaultdict(list)
    for row in inventory["vsumm_user"]:
        match = re.fullmatch(r"UserSummary/(v\d+)/(user\d+)/Frame(\d+)\.jpeg", row["name"])
        if match: groups[(match[1], match[2])].append(int(match[3]))
    listing = []
    for line in (root / "sources/vsumm_readme.txt").read_text().splitlines():
        if not re.match(r"\[v\d+\]", line): continue
        cells = line.split("|")
        mm, ss = map(int, cells[3].strip().split(":"))
        listing.append({"video": re.match(r"\[(v\d+)\]", line)[1],
                        "declared_frames": int(cells[2].strip().replace(",", "")),
                        "rounded_duration_sec": mm * 60 + ss})
    write(root / "vsumm_summary_indices.json", [{"video": k[0], "user": k[1], "one_based_filenames": sorted(v)} for k, v in sorted(groups.items())])
    gaps = [max(np.diff(sorted(v))) / 30 for v in groups.values() if len(v) > 1]
    result["datasets"]["VSUMM"] = {"audited_subset": "Original Open Video 50; additional YouTube subset not downloaded",
        "videos_in_archive": len(inventory["vsumm_video"]), "user_summaries": len(groups),
        "jpeg_keyframes": sum(map(len, groups.values())),
        "keys_per_summary_range": [min(map(len, groups.values())), max(map(len, groups.values()))],
        "readme_under_60s": [r["video"] for r in listing if r["rounded_duration_sec"] < 60],
        "readme_rounded_60s": [r["video"] for r in listing if r["rounded_duration_sec"] == 60],
        "within_summary_max_gap_nominal30fps_sec": {"median": float(np.median(gaps)), "max": max(gaps)},
        "gap_scope": "Filename-spacing diagnostic at declared 30 fps only; exact temporal alignment is not established",
        "sample_decodes": {p: v for p, v in sample_probes.items() if "/vsumm/" in p},
        "jpeg_mapping": json.loads((root / "vsumm_jpeg_mapping.json").read_text()),
        "late_jpeg_mapping": json.loads((root / "vsumm_late_frame_mapping.json").read_text()),
        "role": "Static-keyframe reference only; sampled original videos/indices need repair or verified mapping before ETRI reconstruction use",
        "ready_to_replace_skem": False}
    result["requirements"] = {
        "E1_temporal_fidelity": "Original continuous video is usable input; keyframe coverage, motion and flicker need reconstruction evaluation",
        "E2_hallucination": "None of the three standard annotation packages provides independent Added/Missing/Distorted reconstruction-error labels",
        "E3_new_metrics": "Human summary labels are not semantic-error ground truth; collect independent error annotations for metric validation",
        "E4_transmission": "Count actual visual and semantic-packet channel uses; fewer summary frames alone does not establish equal-quality savings",
        "AWGN_same_input_before_after": "Pipeline test condition, not an intrinsic dataset annotation; feasible using paired original videos",
        "continuous_60s": "Use an unstitched 60s original interval; keyframe slideshows or concatenated selected summary shots do not satisfy this",
        "scene_change_low_medium_high": "TVSum already has 10/10/10 candidate inputs; all classes still require independent confirmation. SumMe/VSUMM lack these ETRI strata",
        "refresh_and_missed_cut_fallback": "Must be tested by the transmission/reconstruction system; not supplied by summary annotations",
        "training_history_and_dynamic_information": "Pretraining provenance and adequacy of motion/captions require separate module evidence"}
    result["decision"] = "Keep existing TVSum+ClipShots benchmark. Consider qualified SumMe videos as supplementary inputs. Do not adopt public summaries as a quality-preserving SKEM replacement."
    result["provenance"] = {str(p.relative_to(root)): digest(p) for p in (root / "sources").rglob("*") if p.is_file() and "ranges" not in p.parts}
    result["audit_code_sha256"] = digest(Path(__file__))
    write(root / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "TVSum_60s": len(normalized_probes),
        "SumMe_metadata_60s": result["datasets"]["SumMe"]["metadata_at_least_60s"],
        "VSUMM_summaries": len(groups), "samples_crc_verified": True}, ensure_ascii=False))


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inventory", "audit"])
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.output.resolve()
    if args.action == "audit":
        audit(args.output.resolve())
        return
    result = inventories(args.output.resolve())
    for name, rows in result.items():
        print(name, len(rows), "members", [r["name"] for r in rows[:8]])


if __name__ == "__main__": main()
