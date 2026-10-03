"""Measure 30 local Qwen keyframe decisions; do not run full-video selection.

Three 10-candidate sequential blocks start at existing InternVL keyframes.
This is a runtime pilot for a new short-answer selector, not a PSSS reproduction
or a quality comparison. No historical selection or caption output is modified.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import statistics
import time
from pathlib import Path

import qwen35_advanced_caption as advanced
import qwen35_caption as base


PROMPT = """You select keyframes for faithful video reconstruction, not a highlight summary.
The first image is the latest accepted keyframe. The second is a later candidate.
Select the candidate if it contains a meaningful visible change: an object enters
or leaves, visibility or occlusion changes, a person's pose/action or object
position changes substantially, or the camera reveals different scene content.
Skip near-duplicates with only tiny movement, lighting flicker, or compression
differences. Compare only these two images; do not invent events between them.
Return ONLY JSON with exactly these fields:
{"select": true or false, "reason": "one short English phrase"}.
Use at most 12 words for the reason. Do not write a scene description."""


def read(path):
    return json.loads(Path(path).read_text())


def decision(text):
    # Preserve raw output in every record, even when formatting is invalid.
    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = "\n".join(candidate.splitlines()[1:-1])
    value = json.loads(candidate)
    if (not isinstance(value, dict) or set(value) != {"select", "reason"}
            or type(value["select"]) is not bool
            or not isinstance(value["reason"], str) or not value["reason"].strip()
            or len(value["reason"].split()) > 12):
        raise ValueError("Invalid keyframe decision schema")
    return value


def measure(model, processor, source, fps, reference, candidate):
    import torch
    from PIL import Image

    torch.cuda.synchronize()
    started = time.perf_counter()
    images = []
    try:
        for index in (reference, candidate):
            with Image.open(source / f"{index}.png") as image:
                images.append(image.convert("RGB"))
        labels = [f"REFERENCE accepted frame {reference}, time {reference / fps:.4f} s",
                  f"CANDIDATE frame {candidate}, time {candidate / fps:.4f} s"]
        data = base.preprocess(processor, images, labels, PROMPT, 262144)
        if data.input_ids.shape[1] > 4096:
            raise ValueError("Input token budget exceeded")
        data = data.to("cuda:0")
        torch.cuda.synchronize()
        prepared = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**data, max_new_tokens=96, do_sample=False,
                                    use_cache=True)
        torch.cuda.synchronize()
        generated = time.perf_counter()
        tokens = output[0, data.input_ids.shape[1]:]
        raw = processor.decode(tokens, skip_special_tokens=True).strip()
        record = dict(reference=reference, candidate=candidate, raw_text=raw,
                      generated_tokens=len(tokens), input_tokens=data.input_ids.shape[1],
                      image_grid_thw=data.image_grid_thw.tolist(),
                      truncated=len(tokens) >= 96,
                      preparation_seconds=prepared - started,
                      generation_seconds=generated - prepared)
        try:
            if record["truncated"]:
                raise ValueError("Output reached token limit")
            record.update(decision(raw), valid=True)
        except (ValueError, TypeError) as exc:
            record.update(valid=False, error=str(exc))
        record["pair_seconds"] = time.perf_counter() - started
        return record
    finally:
        for image in images:
            image.close()


def run(root, baseline):
    import torch

    if (root / "plan.json").exists():
        raise FileExistsError("Use a new output directory; preserve prior timing attempts")
    started = time.perf_counter()
    prep = read(baseline / "prepare.json")
    original_keys = read(baseline / "keyframes.json")["indices"]
    if prep["frames"] != 337 or prep["fps"] != 24:
        raise ValueError("This pilot expects the verified 337-frame single_subject input")
    blocks = [("early", 0), ("middle", 167), ("late", 313)]
    source = baseline / "data/frames/sample"
    used = sorted({i for _, anchor in blocks for i in range(anchor, anchor + 11)})
    if not all(anchor in original_keys for _, anchor in blocks):
        raise ValueError("Block anchors must exist in the historical selection")
    input_video = Path(read(baseline / "run_config.json")["input"])
    if base.sha256(input_video) != prep["source_sha256"]:
        raise ValueError("Source video hash differs from the baseline")
    manifest = read(base.REPORTS / "model_manifest.json")
    for item in manifest["files"]:
        path = base.SNAPSHOT / item["file"]
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise ValueError(f"Missing or incomplete model file: {path}")
    plan = dict(created_utc=base.now(), model_id=base.MODEL_ID, revision=base.REVISION,
                quantization="nf4", attention="sdpa", enable_thinking=False,
                do_sample=False, seed=2025, max_pixels=262144, max_new_tokens=96,
                prompt=PROMPT, baseline=str(baseline), source_frames=str(source),
                source_video=str(input_video), source_video_sha256=prep["source_sha256"],
                frames=prep["frames"], fps=prep["fps"], full_video_comparisons=336,
                source_png_hashes={str(i): base.sha256(source / f"{i}.png") for i in used},
                code_sha256={p.name: base.sha256(p) for p in
                             (Path(__file__), Path(base.__file__), Path(advanced.__file__))},
                model_manifest_sha256=base.sha256(base.REPORTS / "model_manifest.json"),
                blocks=[dict(name=name, anchor=anchor, candidates=list(range(anchor+1, anchor+11)))
                        for name, anchor in blocks], timed_pairs=30, warmup_pairs=2,
                sampling_scope="Three local sequential blocks seeded by historical InternVL keys; not a full Qwen trajectory",
                selection_rule="One short JSON decision; update reference on select=true; not original two-round PSSS",
                timing_scope="PNG loading, preprocessing, device transfer, generation, decoding and schema validation; record I/O tracked separately",
                quality_verified=False, full_video_selection_completed=False)
    base.write_json(root / "plan.json", plan)
    setup_started = time.perf_counter()
    model, processor, runtime = advanced.load("nf4", reserve_mib=1024)
    runtime["model_and_processor_setup_seconds"] = time.perf_counter() - setup_started
    runtime["created_utc"] = base.now()
    base.write_json(root / "runtime.json", runtime)
    print(json.dumps(dict(phase="LOADED", model_load_seconds=runtime["model_load_seconds"])), flush=True)
    warmups = []
    for number in range(2):
        record = measure(model, processor, source, prep["fps"], 0, 1)
        warmups.append(record)
        base.write_json(root / "warmup.json", warmups)
        if not record["valid"]:
            raise RuntimeError("Warmup decision failed validation; see warmup.json")
    torch.cuda.reset_peak_memory_stats()
    batch_started = time.perf_counter()
    records = []
    with advanced.Monitor() as monitor:
        for block, anchor in blocks:
            reference = anchor
            for candidate in range(anchor + 1, anchor + 11):
                record = measure(model, processor, source, prep["fps"], reference, candidate)
                record.update(block=block, ordinal=len(records), created_utc=base.now())
                base.write_json(root / "pairs" / f"{len(records):02d}.json", record)
                if not record["valid"]:
                    raise RuntimeError("Timed decision failed validation; see pairs directory")
                records.append(record)
                if record["select"]:
                    reference = candidate
                base.write_json(root / "progress.json", dict(status="RUNNING", completed=len(records), total=30,
                                last_pair_seconds=record["pair_seconds"], updated_utc=base.now()))
                print(json.dumps({k: record[k] for k in
                                 ("ordinal", "block", "reference", "candidate", "select", "pair_seconds", "generated_tokens")}), flush=True)
        batch_seconds = time.perf_counter() - batch_started
    times = [r["pair_seconds"] for r in records]
    block_means = {name: statistics.mean(r["pair_seconds"] for r in records if r["block"] == name)
                   for name, _ in blocks}
    # Include observed output/progress write overhead in full-run projections.
    overhead_per_pair = max(0., batch_seconds - sum(times)) / len(times)
    setup = runtime["model_and_processor_setup_seconds"]
    warmup_seconds = sum(r["pair_seconds"] for r in warmups)
    summary = dict(status="PASS_30_PAIR_TIMING", created_utc=base.now(),
                   model_id=base.MODEL_ID, quantization="nf4", timed_pairs=len(records),
                   mean_pair_seconds=statistics.mean(times), median_pair_seconds=statistics.median(times),
                   min_pair_seconds=min(times), max_pair_seconds=max(times),
                   mean_generation_seconds=statistics.mean(r["generation_seconds"] for r in records),
                   mean_generated_tokens=statistics.mean(r["generated_tokens"] for r in records),
                   block_mean_pair_seconds=block_means, timed_batch_seconds=batch_seconds,
                   record_overhead_per_pair_seconds=overhead_per_pair,
                   model_load_seconds=runtime["model_load_seconds"],
                   model_and_processor_setup_seconds=setup, warmup_seconds=warmup_seconds,
                   total_pilot_wall_seconds=time.perf_counter() - started,
                   selected_candidates=sum(r["select"] for r in records),
                   all_outputs_valid=all(r["valid"] and not r["truncated"] for r in records),
                   full_video_comparisons=336,
                   projected_full_selection_seconds=setup + 336 * (statistics.mean(times) + overhead_per_pair),
                   projected_block_mean_range_seconds=[setup + 336 * (x + overhead_per_pair)
                                                       for x in (min(block_means.values()), max(block_means.values()))],
                   projection_scope="One model load + 336 short single-call decisions; existing source PNGs; excludes source decoding, captions, transmission and reconstruction",
                   uncertainty="Block-mean range is descriptive, not a confidence interval; full-run references/output lengths and GPU load may differ",
                   peak_pytorch_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                   peak_pytorch_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                   device_memory=monitor.summary(), quality_verified=False,
                   full_video_selection_completed=False, plan_sha256=base.sha256(root / "plan.json"))
    base.write_json(root / "RESULT.json", summary)
    base.write_json(root / "progress.json", dict(status=summary["status"], completed=30, total=30,
                                               updated_utc=base.now()))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path,
                        default=base.REPO / "outputs/webvid1_ablation_single_subject_611ed735e038/baseline")
    args = parser.parse_args()
    root = args.root.resolve()
    with (base.REPO / ".local/etri_60s_check.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(root, args.baseline.resolve())
        except Exception as exc:
            base.write_json(root / f"failure_{time.time_ns()}.json",
                            dict(status="FAILED", error=repr(exc), created_utc=base.now()))
            raise


if __name__ == "__main__":
    main()
