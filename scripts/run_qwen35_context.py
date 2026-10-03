"""Frozen, resumable paired caption extraction with causal visual context.

Uses source images only: other captions and review labels are never model inputs.
The target-only control and context arm have equal image counts per interval.
Scene scanning is a conservative heuristic, not a validated shot detector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import qwen35_advanced_caption as advanced
from qwen35_advanced_caption import base

ARMS = ("context", "target_only")
SCOPE = """
Each image is explicitly labelled CONTEXT or TARGET with its source timestamp.
Write the caption ONLY about the TARGET images and the specified TARGET interval.
CONTEXT images are earlier original frames, supplied only to help recognize an
object that is ALSO visibly present in TARGET images. Never mention an object,
person, action, or setting solely because it appears in CONTEXT. Do not carry
earlier events or object counts into the TARGET interval. Describe changes or
motion only when distinct TARGET images establish them; CONTEXT alone cannot
establish motion within TARGET. If no CONTEXT images are supplied, use TARGET
images alone. Do not refer to the labels, timestamps, or this instruction in the
caption. Inspect the edges of TARGET images before claiming an entity is absent.
""".strip()


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def select_target(row, count):
    """Preserve all distinct frozen source samples, then fill the largest gaps."""
    start, end = row["start"], row["end_exclusive"]
    chosen = set(row["source_indices"])
    if not chosen or not all(start <= i < end for i in chosen):
        raise ValueError("Invalid frozen target samples")
    count = min(count, end - start)
    if len(chosen) > count:
        raise ValueError("Budget cannot preserve the frozen target samples")
    while len(chosen) < count:
        chosen.add(max((i for i in range(start, end) if i not in chosen),
                       key=lambda i: (min(abs(i-j) for j in chosen), -i)))
    return sorted(chosen)


def select_context(start, lower, count):
    count = min(count, start - lower)
    if count <= 0:
        return []
    if count == 1:
        return [start - 1]
    return [lower + round(i * (start - 1 - lower) / (count - 1)) for i in range(count)]


def sample_pair(row, fps, scene_starts, seconds=2.0):
    start, end = row["start"], row["end_exclusive"]
    budget = min(16, end - start)
    scene_start = max(i for i in scene_starts if i <= start)
    lower = max(scene_start, start - round(seconds * fps))
    internal_cut = any(start < i < end for i in scene_starts)
    context_count = 0 if internal_cut else min(4, start-lower,
                                             budget-len(set(row["source_indices"])))
    context = select_context(start, lower, context_count)
    return dict(segment=row["segment"], matched_pair=row["matched_pair"],
                start=start, end_exclusive=end, original_indices=row["source_indices"],
                scene_start=scene_start, target_contains_detected_cut=internal_cut,
                image_budget=budget, context_indices=context,
                context_target_indices=select_target(row, budget-len(context)),
                control_target_indices=select_target(row, budget))


def scan_scenes(source, count):
    import cv2
    import numpy as np
    rows, previous, hist_previous = [], None, None
    for i in range(count):
        image = cv2.imread(str(source / f"{i}.png"))
        if image is None:
            raise ValueError(f"Missing source frame {i}")
        image = cv2.resize(image, (180, 100))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        if previous is not None:
            mad = float(np.mean(np.abs(image.astype(np.float32)-previous.astype(np.float32)))/255)
            distance = float(cv2.compareHist(hist_previous, hist, cv2.HISTCMP_BHATTACHARYYA))
            rows.append(dict(frame=i, mean_rgb_difference=mad,
                             gray_histogram_distance=distance, candidate=mad >= .20 and distance >= .35))
        previous, hist_previous = image, hist
    return dict(method="Adjacent RGB MAD >= 0.20 AND grayscale histogram distance >= 0.35",
                resize=[180, 100], histogram_bins=32,
                scene_starts=[0] + [r["frame"] for r in rows if r["candidate"]], scores=rows,
                limitation="Heuristic boundaries; similar-looking cuts or fades may be missed. No shot-detector accuracy claim.")


def prepare(root, inputs_file):
    if (root / "plan.json").exists():
        raise FileExistsError(root / "plan.json")
    dataset = read(inputs_file)
    for name, digest in dataset["input_files_sha256"].items():
        if base.sha256(name) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    source = Path(dataset["source_frames"])
    for frame, digest in dataset["source_png_hashes"].items():
        if base.sha256(source / f"{frame}.png") != digest:
            raise ValueError(f"Frozen image changed: {frame}")
    count = len(list(source.glob("*.png")))
    scene = scan_scenes(source, count)
    base.write_json(root / "scene_scan.json", scene)
    prompt_file = base.REPO / "configs/captions/faithful_visible_under80.txt"
    prompt = prompt_file.read_text().strip().replace(
        "supplied, chronologically ordered source frames", "chronologically ordered TARGET frames") + "\n\n" + SCOPE
    (root / "prompt.txt").write_text(prompt + "\n")
    records = [sample_pair(r, dataset["fps"], scene["scene_starts"]) for r in dataset["records"]]
    plan = dict(status="FROZEN_BEFORE_INFERENCE", created_utc=base.now(),
                model_id=base.MODEL_ID, revision=base.REVISION, quantization="nf4",
                inputs_file=str(inputs_file), inputs_sha256=base.sha256(inputs_file),
                source=dataset["source"], source_frames=str(source), fps=dataset["fps"],
                source_video_sha256=dataset["source_video_sha256"],
                source_frame_count=count, covered_seconds=dataset["all_fc_covered_seconds"],
                prompt=prompt, prompt_sha256=base.sha256(root / "prompt.txt"),
                prompt_policy_sha256=base.sha256(prompt_file),
                code_sha256={str(Path(p).resolve()): base.sha256(p)
                             for p in [__file__, advanced.__file__, base.__file__]},
                scene_scan_sha256=base.sha256(root / "scene_scan.json"),
                source_png_hashes={str(i): base.sha256(source / f"{i}.png") for i in range(count)},
                scene_starts=scene["scene_starts"], context_seconds=2.0,
                max_images=16, max_context_images=4, max_pixels=262144,
                max_new_tokens=256, retry_token_budget=384, reserve_mib=1024,
                retry_policy="Regenerate from the same images with a shorter-word reminder; save every attempt; never truncate text.",
                arms=list(ARMS), records=records,
                input_policy="Original images and instructions only; no FC, PLLaVA, Qwen baseline captions or error labels",
                pair_policy="Equal image count and resolution per interval; context replaces up to four additional target samples. All original distinct target samples retained. Very short intervals have fewer context frames.")
    plan["signature"] = fingerprint(plan)
    base.write_json(root / "plan.json", plan)
    print(json.dumps(dict(status=plan["status"], intervals=len(records), arms=list(ARMS),
                          context_intervals=sum(bool(r["context_indices"]) for r in records),
                          scene_starts=scene["scene_starts"])))


def policy(caption, truncated):
    words = len(caption.split())
    sentences = len([s for s in re.split(r"(?<=[.!?])\s+", caption.strip()) if s])
    guesses = re.findall(r"\b(?:possibly|probably|maybe|perhaps|might|could be)\b", caption.lower())
    valid = not truncated and 1 <= words <= 79 and sentences <= 6 and not guesses
    return dict(format_passed=valid, word_count=words, sentence_count=sentences,
                speculation_markers=guesses, visual_factual_accuracy_verified=False)


def check_identity(root):
    plan = read(root / "plan.json")
    if plan["signature"] != fingerprint({k:v for k,v in plan.items() if k != "signature"}):
        raise ValueError("Plan signature mismatch")
    for path, expected in plan["code_sha256"].items():
        if base.sha256(path) != expected:
            raise ValueError(f"Code changed after freeze: {path}")
    for path, expected in [(plan["inputs_file"], plan["inputs_sha256"]),
                           (root / "prompt.txt", plan["prompt_sha256"]),
                           (root / "scene_scan.json", plan["scene_scan_sha256"])]:
        if base.sha256(path) != expected:
            raise ValueError(f"Input changed after freeze: {path}")
    source = Path(plan["source_frames"])
    for frame, expected in plan["source_png_hashes"].items():
        if base.sha256(source / f"{frame}.png") != expected:
            raise ValueError(f"Source changed: {frame}")
    return plan


def infer(root):
    from PIL import Image
    plan = check_identity(root)
    tasks = []
    for row in plan["records"]:
        # Alternate arm order to avoid an arm-specific startup/timing advantage.
        for arm in (ARMS if row["segment"] % 2 == 0 else ARMS[::-1]):
            path = root / "captions" / arm / f"{row['segment']:03d}.json"
            if path.exists():
                old = read(path)
                if old["signature"] != plan["signature"] or not policy(old["caption"], old["truncated"])["format_passed"]:
                    raise ValueError("Invalid checkpoint")
            else:
                tasks.append((row, arm, path))
    total = 2 * len(plan["records"])
    if not tasks:
        print("All paired captions are complete.")
        return
    run_start = time.monotonic()
    model, processor, runtime = advanced.load(plan["quantization"], plan["reserve_mib"])
    runtime.update(created_utc=base.now(), signature=plan["signature"])
    runtime_file = root / "runtimes" / f"{time.time_ns()}.json"
    base.write_json(runtime_file, runtime)
    source = Path(plan["source_frames"])
    completed = total-len(tasks)
    for row, arm, path in tasks:
        context = row["context_indices"] if arm == "context" else []
        target = row["context_target_indices"] if arm == "context" else row["control_target_indices"]
        images, labels = [], []
        for role, indices in [("CONTEXT", context), ("TARGET", target)]:
            for index in indices:
                with Image.open(source / f"{index}.png") as image:
                    images.append(image.convert("RGB"))
                labels.append(f"{role} source frame {index}; time {index/plan['fps']:.4f} s")
        prompt = plan["prompt"] + f"\nTARGET interval: [{row['start']/plan['fps']:.4f}, {row['end_exclusive']/plan['fps']:.4f}) seconds."
        attempts = []
        for attempt_index in range(4):
            reminder = "" if attempt_index == 0 else (
                f"\nFORMAT REMINDER: Use only {max(35, 60-attempt_index*10)} to {max(45, 70-attempt_index*10)} words, at most six sentences. "
                "Do not guess or include possibilities. Return only the caption.")
            actual_prompt = prompt + reminder
            result = advanced.generate(model, processor, images, labels, actual_prompt,
                                       plan["max_pixels"], plan["max_new_tokens"] if attempt_index == 0 else plan["retry_token_budget"])
            result["policy"] = policy(result["caption"], result["truncated"])
            attempts.append(dict(attempt=attempt_index, prompt=actual_prompt, **result))
            if result["policy"]["format_passed"]:
                break
        record = dict(**row, arm=arm, supplied_context_indices=context,
                      supplied_target_indices=target, supplied_indices=context+target,
                      labels=labels, model_id=plan["model_id"], revision=plan["revision"],
                      quantization=plan["quantization"], signature=plan["signature"],
                      created_utc=base.now(), inference_provider="LOCAL_ONLY",
                      runtime_file=str(runtime_file), attempts=attempts, **result)
        if not result["policy"]["format_passed"]:
            base.write_json(root / "failed" / arm / path.name, record)
            raise RuntimeError(f"Caption failed format after four attempts: {arm}/{row['segment']}")
        base.write_json(path, record)
        completed += 1
        progress = dict(status="RUNNING", completed=completed, total=total,
                        last_arm=arm, last_segment=row["segment"], updated_utc=base.now(),
                        current_session_seconds=time.monotonic()-run_start)
        base.write_json(root / "progress.json", progress)
        print(json.dumps(dict(**progress, words=result["policy"]["word_count"])), flush=True)
        for image in images:
            image.close()
    base.write_json(root / "progress.json", dict(status="PAIRED_CAPTIONS_COMPLETE_REVIEW_PENDING",
                    completed=completed, total=total, updated_utc=base.now(),
                    current_session_seconds=time.monotonic()-run_start))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path,
                        default=base.REPO / "outputs/caption_compare_qwen_tv_low_08_20261002/inputs.json")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        prepare(root, args.inputs.resolve())
    else:
        infer(root)


if __name__ == "__main__":
    main()
