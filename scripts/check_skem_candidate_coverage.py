"""CPU-only sparse-candidate coverage audit on development scene changes.

This checks input coverage, not SKEM decisions or reconstruction quality. Stored
ClipShots annotations are a reference with a two-frame mapping tolerance, not
new independently verified ground truth.
"""
import argparse
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from semantic_transmission.artifacts import sha256, write_json


def main():
    import cv2
    import numpy as np
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=REPO / "outputs/skem_speed_20260929")
    args = p.parse_args()
    root = args.output.resolve()
    destination = root / "candidate_coverage.json"
    if destination.exists():
        raise ValueError("coverage report exists; preserve previous evidence")
    protocol = json.loads((root / "protocol.json").read_text())
    manifest_path = REPO / "data/etri_benchmark_v1_20260924/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    dataset_root = Path(json.loads((REPO / ".local/datasets.json").read_text())["root"])
    annotation_path = dataset_root / "etri_long_video_20260924/metadata/clipshots_annotations_only_gradual.json"
    annotations = json.loads(annotation_path.read_text())
    module_spec = importlib.util.spec_from_file_location("speed", REPO / "scripts/benchmark_skem_speed.py")
    speed = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(speed)
    speed.validate(root)
    result = {"status": "CPU_CANDIDATE_COVERAGE_ONLY", "tolerance_frames": 2,
        "reference": "Stored ClipShots annotations; no new independent framewise review",
        "manifest_sha256": sha256(manifest_path), "annotation_sha256": sha256(annotation_path),
        "code_sha256": sha256(Path(__file__)), "selection_protocol": protocol["signature"],
        "videos": {}}
    cv2.setNumThreads(1)
    for name in ("cs_high_04", "cs_high_08", "cs_high_10"):
        row = next(r for r in manifest if r["id"] == name)
        assert row["split"] == "development"
        video = Path(row["processed_path"])
        assert sha256(video) == row["processed_sha256"]
        start = time.perf_counter()
        cap = cv2.VideoCapture(str(video))
        thumbs = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            thumbs.append(cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32) / 255)
        cap.release()
        assert len(thumbs) == row["frames"] == 1440
        scores = [0.] + [float(np.abs(b-a).mean()) for a, b in zip(thumbs, thumbs[1:])]
        cuts = [i for i, value in enumerate(scores) if value >= protocol["cut_mad_threshold"]]
        grid = speed.candidates(len(thumbs), protocol["candidate_stride"])
        augmented = speed.candidates(len(thumbs), protocol["candidate_stride"], cuts)
        source_fps = float(Fraction(row["probe"]["streams"][0]["avg_frame_rate"]))
        references = []
        for first, last in annotations[row["source_id"] + ".mp4"]["transitions"]:
            # The last annotated source frame is mapped to the normalized time axis.
            frame = round((last / source_fps - row["clip_start_sec"]) * row["frame_rate"])
            if not 2 <= frame < row["frames"] - 2:
                continue
            distance = min((abs(c-frame) for c in cuts), default=None)
            references.append({"source_frames": [first, last], "mapped_frame": frame,
                "nearest_detection_distance": distance,
                "detected_within_tolerance": distance is not None and distance <= 2,
                "grid_has_candidate_within_one_frame": any(abs(c-frame) <= 1 for c in grid),
                "augmented_has_candidate_within_one_frame": any(abs(c-frame) <= 1 for c in augmented),
                "local_scores": scores[frame-2:frame+3]})
        result["videos"][name] = {"split": row["split"], "video_sha256": row["processed_sha256"],
            "frames": len(thumbs), "decoded_scan_seconds": time.perf_counter()-start,
            "grid_comparisons": len(grid)-1, "augmented_comparisons": len(augmented)-1,
            "detected_frames": cuts, "reference_count": len(references),
            "matched_reference_count": sum(r["detected_within_tolerance"] for r in references),
            "reference_events": references}
    write_json(destination, result)
    for name, r in result["videos"].items():
        print(name, "reference matches", r["matched_reference_count"], "/", r["reference_count"],
              "comparisons", r["augmented_comparisons"], "/", r["frames"]-1)


if __name__ == "__main__":
    main()
