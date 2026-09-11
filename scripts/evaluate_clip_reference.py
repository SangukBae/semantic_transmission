#!/usr/bin/env python3
"""Supplementary frame-paired CLIP similarity, using one CPU evaluator for all models.

This measures broad visual similarity; it does not establish object truthfulness,
correct motion direction, or absence of hallucination. The original LGVSC evaluator
reports (1 + cosine) / 2 with the standard ViT-B/32 center-crop preprocessing.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.research_quality import read_frames


def pixel_hash(frames):
    return hashlib.sha256(frames.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--baseline-cache", type=Path)
    args = parser.parse_args()
    if bool(args.run_root) != bool(args.baseline_cache):
        parser.error("final evaluation requires both --run-root and --baseline-cache")
    import clip
    import numpy as np
    from PIL import Image
    import torch
    torch.set_num_threads(2)
    checkpoint = Path.home() / ".cache/clip/ViT-B-32.pt"
    expected = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
    if sha256(checkpoint) != expected:
        raise ValueError("CLIP checkpoint differs from the original ViT-B/32 release")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inventory = json.loads(args.baseline_inventory.read_text())
    if not inventory["source_video_pixels_match"]:
        raise ValueError("source PNGs must first be verified against the original videos")
    def original_frames(baseline):
        worker = Path(baseline["frames_directory"]).parents[2]
        return read_frames(worker / "logs" / (baseline["video"] + "_frames"))
    model, preprocess = clip.load(str(checkpoint), device="cpu", jit=False)
    model.eval()

    def encode(frames):
        features = []
        with torch.inference_mode():
            for start in range(0, len(frames), 8):
                images = torch.stack([preprocess(Image.fromarray(frame)) for frame in frames[start:start+8]])
                values = model.encode_image(images).float()
                features.append((values / values.norm(dim=-1, keepdim=True)).numpy())
        return np.concatenate(features)

    def score(name, label, frames, original, reference):
        if frames.shape != original.shape:
            raise ValueError("CLIP requires the same original frame count and dimensions")
        cosine = np.sum(reference * encode(frames), axis=1)
        row = {"video": name, "model": label, "frames": len(frames),
               "clip_cosine": float(cosine.mean()), "clip_paper_scaled": float(((cosine+1)/2).mean()),
               "source_pixels_sha256": pixel_hash(original), "reconstruction_pixels_sha256": pixel_hash(frames)}
        with (output / f"{name}_{label}_frames.csv").open("w", newline="") as stream:
            writer = csv.writer(stream);writer.writerow(["frame", "clip_cosine", "clip_paper_scaled"])
            writer.writerows((i, float(x), float((x+1)/2)) for i, x in enumerate(cosine))
        print(f"CLIP evaluated {name}: {label}", flush=True)
        return row

    rows, references, sources = [], {}, {}
    if args.baseline_cache:
        cached = json.loads((args.baseline_cache / "report.json").read_text())
        if cached["baseline_inventory_sha256"] != sha256(args.baseline_inventory) or cached["checkpoint_sha256"] != expected:
            raise ValueError("baseline CLIP cache provenance mismatch")
        rows = cached["rows"]
        references = dict(np.load(args.baseline_cache / "source_features.npz", allow_pickle=False))
        manifest = json.loads((args.run_root / "batch_manifest.json").read_text())
        if manifest["status"] != "PASSED" or manifest["completed_videos"] != 10:
            raise ValueError("final CLIP comparison requires all ten successful LGVSC outputs")
        for baseline in inventory["baselines"]:
            row = next(r for r in rows if r["video"] == baseline["video"] and
                       r["model"] == f"SGD_{baseline['decoder_policy']}_{baseline['guide_profile']}")
            if pixel_hash(read_frames(baseline["frames_directory"])) != row["reconstruction_pixels_sha256"]:
                raise ValueError("baseline reconstruction changed after CLIP evaluation")
        for source in manifest["inputs"]:
            name = source["id"]
            if sha256(source["path"]) != source["sha256"]:
                raise ValueError("LGVSC source changed")
            baseline = next(b for b in inventory["baselines"] if b["video"] == name)
            if baseline["source_sha256"] != source["sha256"]:
                raise ValueError("CLIP and LGVSC source file mismatch")
            original = original_frames(baseline)
            prior = next(r for r in rows if r["video"] == name)
            if pixel_hash(original) != prior["source_pixels_sha256"]:
                raise ValueError("CLIP cache source mismatch")
            video = next((args.run_root / name / "receiver/reconstruction").glob("*.mp4"))
            frames = read_frames(video.parent / (video.stem + "_frames"))
            rows.append(score(name, "LGVSC_SKEM_DSA_50", frames, original, references[name]))
    else:
        for baseline in inventory["baselines"]:
            name = baseline["video"]
            if sha256(baseline["source"]) != baseline["source_sha256"]:
                raise ValueError("historical source changed")
            if name not in references:
                sources[name] = original_frames(baseline)
                references[name] = encode(sources[name])
            rows.append(score(name, f"SGD_{baseline['decoder_policy']}_{baseline['guide_profile']}",
                              read_frames(baseline["frames_directory"]), sources[name], references[name]))
        np.savez_compressed(output / "source_features.npz", **references)
    aggregate = []
    for label in dict.fromkeys(r["model"] for r in rows):
        group = [r for r in rows if r["model"] == label]
        if len(group) != 10:
            raise ValueError("CLIP operating point is missing videos")
        aggregate.append({"model": label, "videos": 10,
                          **{key: statistics.mean(r[key] for r in group) for key in ("clip_cosine", "clip_paper_scaled")}})
    report = {"status": "PASSED", "boundary": "lossless_RGB_uint8_reconstruction_frames",
              "checkpoint_sha256": expected, "torch": torch.__version__, "device": "CPU_FP32",
              "clip_code_sha256": sha256(Path(clip.__file__).with_name("model.py")),
              "baseline_inventory_sha256": sha256(args.baseline_inventory), "rows": rows, "aggregate": aggregate,
              "scope": "supplementary_broad_visual_similarity_not_hallucination_or_temporal_correctness"}
    write_json(output / "report.json", report)
    for name, values in (("per_video", rows), ("aggregate", aggregate)):
        with (output / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)


if __name__ == "__main__":
    main()
