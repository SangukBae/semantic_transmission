#!/usr/bin/env python3
"""Verify historical SGD-JSCC packet sizes and exact source pixels, read-only."""
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.research_quality import read_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sgd-repo", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/etri_video_eval/processed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sgd = args.sgd_repo.resolve()
    root = sgd / "outputs/integrated_semantic_validation_10db_20260828_093017"
    table = root / "integrated_per_video.csv"
    rows = list(csv.DictReader(table.open()))
    selected = []
    for row in rows:
        if (row["decoder_policy"], row["guide_profile"]) not in [
            ("full50", "baseline"), ("full50", "candidate_both_omit"), ("few10", "candidate_both_omit")]:
            continue
        video, policy, config = row["video"], row["decoder_policy"], row["config"]
        matches = list((root / f"reconstruction/{policy}/workers").glob(f"*/recon_videos/{video}/{config}"))
        if len(matches) != 1:
            raise ValueError(f"missing/ambiguous historical reconstruction: {video}/{config}")
        frames = matches[0]
        worker = frames.parents[2]
        files = sorted((worker / f"packets/{video}/{config}").glob("*.sgbundle"))
        size = sum(p.stat().st_size for p in files)
        if size != int(row["total_bundle_bytes"]):
            raise ValueError("historical packet byte total mismatch")
        source = args.input_dir / f"{video}.mp4"
        decoded = read_video(source)
        old = worker / f"logs/{video}_frames"
        original = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in sorted(old.glob("*.png"))])
        if not np.array_equal(original, decoded):
            raise ValueError(f"historical source pixels differ: {video}/{policy}/{config}")
        if len(files) != len(decoded):
            raise ValueError("historical transmission must contain one bundle per source frame")
        selected.append({"video": video, "decoder_policy": policy, "guide_profile": row["guide_profile"],
            "source": str(source.resolve()), "source_sha256": sha256(source), "frames_directory": str(frames),
            "video_path": str(frames / "recon.mp4"), "bundle_count": len(files), "verified_bundle_bytes": size,
            "packet_files": [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files],
            "historical_metrics": {k: row[k] for k in ("mean_psnr", "mean_ssim", "mean_lpips", "total_elapsed_s")}})
    if len(selected) != 30 or len({r["video"] for r in selected}) != 10:
        raise ValueError("expected 10 videos x 3 preserved operating points")
    write_json(args.output, {"source_video_pixels_match": True, "baselines": selected,
                            "table_sha256": sha256(table)})
    print("Verified 30 historical packet sets and source-frame sequences across all 10 videos")


if __name__ == "__main__":
    main()
