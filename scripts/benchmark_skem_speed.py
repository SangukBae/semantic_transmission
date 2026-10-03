"""Bounded SKEM speed ablations; never changes the frozen selector or old runs.

Each window restarts at a previously selected keyframe. BF16, int8, brief
descriptions, and sparse candidates differ in exactly one intervention.
Selection agreement is diagnostic, not a reconstruction quality guarantee.
"""
import argparse
import ast
import csv
import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.webvid5 import fingerprint, read_json

SOURCE = REPO / "02_semantic_encoder/skem/MLM-keyframe-internvl.py"
WINDOWS = {"people": (217, 241), "car": (1303, 1327), "door": (1400, 1424)}
MODES = ("baseline", "int8", "brief", "sparse")
BRIEF = ('Image-1: <image>\nImage-2: <image>\nCompare the two images. '
         'Describe each in at most 45 words. Retain the setting, object counts, '
         'positions, appearances, and actions; explicitly note changed or missing objects. '
         'Use: img1{description} img2{description}.')


def candidates(count, stride, cuts=()):
    if count < 2 or stride < 1:
        raise ValueError("invalid candidate sampling")
    result = set(range(0, count, stride)) | {0, count - 1}
    for cut in cuts:
        result.update(i for i in (cut - 1, cut, cut + 1) if 0 <= i < count)
    return sorted(result)


def prepare(root, baseline):
    import cv2
    import numpy as np
    base = baseline / "baseline"
    original = read_json(base / "run_config.json")
    manifest = {"version": 1, "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "baseline": str(baseline), "windows": {}, "modes": list(MODES),
                "brief_prompt": BRIEF, "brief_max_new_tokens": 256,
                "candidate_stride": 6, "cut_mad_threshold": 0.12,
                "source_code_sha256": {str(p.relative_to(REPO)): sha256(p) for p in
                    (SOURCE, Path(__file__).resolve(), REPO / "src/semantic_transmission/internvl_memory.py",
                     REPO / "src/semantic_transmission/exact_reuse.py")},
                "baseline_protocol_sha256": sha256(baseline / "protocol.json"),
                "scope": "Three restarted 25-frame development windows, one source and seed. Not 60s validation.",
                "predeclared_review": {"selection": "time, choices, PSSS, truncation, failures",
                    "quality": "full-window and interior PSNR/LPIPS, temporal error; inspect Added/Missing/Distorted",
                    "acceptance": "No automatic adoption from speed or keyframe agreement; independent semantic review remains required.",
                    "transmission": "actual visual plus digital channel uses; report increased cost"}}
    if (root / "protocol.json").exists():
        raise ValueError("protocol already exists")
    oldkeys = read_json(base / "keyframes.json")["indices"]
    for name, (start, end) in WINDOWS.items():
        assert start in oldkeys
        folder = root / "inputs" / name
        folder.mkdir(parents=True)
        video = folder / "source.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(base / "data/normalized.mp4"),
            "-vf", f"trim=start_frame={start}:end_frame={end+1},setpts=PTS-STARTPTS", "-an",
            "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p", "-threads", "2", str(video)], check=True)
        frames = folder / "frames"
        frames.mkdir()
        cap = cv2.VideoCapture(str(video))
        hashes, values = {}, []
        for i in range(end - start + 1):
            ok, frame = cap.read()
            if not ok: raise ValueError("window truncated")
            old = cv2.imread(str(base / f"data/frames/sample/{start+i}.png"))
            if not np.array_equal(old, frame): raise ValueError("window pixels changed")
            path = frames / f"{i}.png"
            assert cv2.imwrite(str(path), frame)
            hashes[str(i)] = sha256(path)
            values.append(cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32) / 255)
        assert not cap.read()[0]
        cap.release()
        begin = time.perf_counter()
        differences = [float(np.abs(b-a).mean()) for a,b in zip(values, values[1:])]
        cuts = [i+1 for i,d in enumerate(differences) if d >= manifest["cut_mad_threshold"]]
        count = len(values)
        cfg = dict(original, input=str(video), input_sha256=sha256(video), frames=count, max_frames=count,
                   profile="skem_speed_short_development_v1", preserve_input=True, official_preprocessing=False)
        for key in ("selector_checkpoint", "caption_checkpoint"):
            cfg.pop(key, None)
        write_json(folder / "config.json", cfg)
        manifest["windows"][name] = {"start": start, "end": end, "frames": count,
            "input_sha256": sha256(video), "frame_sha256": hashes, "cuts": cuts,
            "cut_scores": differences, "cut_math_seconds": time.perf_counter()-begin,
            "original_keys_relative": sorted({i-start for i in oldkeys if start <= i <= end} | {count-1}),
            "sparse_candidates": candidates(count, 6, cuts)}
    manifest["signature"] = fingerprint(manifest)
    write_json(root / "protocol.json", manifest)
    print(json.dumps({"prepared": str(root), "windows": {k: v["sparse_candidates"] for k,v in manifest["windows"].items()}}))


def validate(root):
    protocol = read_json(root / "protocol.json")
    assert fingerprint({k:v for k,v in protocol.items() if k != "signature"}) == protocol["signature"]
    for path, digest in protocol["source_code_sha256"].items():
        if sha256(REPO / path) != digest: raise ValueError(f"experiment source changed: {path}")
    for name, spec in protocol["windows"].items():
        folder = root / "inputs" / name
        assert sha256(folder / "source.mp4") == spec["input_sha256"]
        for i,digest in spec["frame_sha256"].items():
            assert sha256(folder / f"frames/{i}.png") == digest
    return protocol


def select(root, mode):
    protocol = validate(root)
    sys.path.insert(0, str(REPO / ".local/vendor/InternVL"))
    if mode == "int8": sys.path.insert(0, str(REPO / ".local/selector_speed_deps"))
    import torch
    from transformers import AutoModel, AutoTokenizer
    from semantic_transmission.exact_reuse import FrameTensorCache
    from semantic_transmission.internvl_memory import place_internvl
    tree = ast.parse(SOURCE.read_text())
    prompts = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("prompt_ask_image", "prompt_compare_image")}
    spec = importlib.util.spec_from_file_location("official_skem", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = read_json(root / "inputs/people/config.json")
    torch.manual_seed(cfg["seed"])
    mode_root = root / "selection" / mode
    mode_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    options = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                   use_flash_attn=cfg["flash_attn"], trust_remote_code=True)
    if mode == "int8": options.update(load_in_8bit=True, device_map={"": 0})
    model = AutoModel.from_pretrained(cfg["models"]["internvl"], **options).eval()
    if mode != "int8":
        model = place_internvl(model, gpu_head=True, cpu_layers=3, compact_cache=True)
    model.chat = types.MethodType(module.custom_chat, model)
    tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["internvl"], trust_remote_code=True, use_fast=True)
    torch.cuda.synchronize()
    info = {"mode": mode, "model_load_seconds": time.perf_counter()-started,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "device_map": getattr(model, "hf_device_map", None), "protocol_signature": protocol["signature"]}
    if mode == "int8":
        import bitsandbytes
        info["bitsandbytes"] = bitsandbytes.__version__
    write_json(mode_root / "runtime.json", info)
    generation = dict(max_new_tokens=256 if mode == "brief" else 1024, do_sample=False,
                      output_scores=True, return_dict_in_generate=True)
    yes = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no = tokenizer.encode("No", add_special_tokens=False)[0]
    for name, window in protocol["windows"].items():
        destination = mode_root / f"{name}.json"
        records = read_json(destination)["records"] if destination.exists() else []
        chosen = [0] + [r["candidate"] for r in records if r["selected"]]
        queue = window["sparse_candidates"] if mode == "sparse" else list(range(window["frames"]))
        assert [r["candidate"] for r in records] == queue[1:len(records)+1]
        cache = FrameTensorCache(lambda p: module.load_image(p, max_num=cfg["max_tiles"]).to(torch.bfloat16).cuda())
        frames = root / "inputs" / name / "frames"
        for index in queue[1+len(records):]:
            begin, wall = time.perf_counter(), time.time()
            previous = chosen[-1]
            images = [cache.get(str(frames / f"{i}.png")) for i in (previous, index)]
            pixels = torch.cat(images)
            patches = [len(im) for im in images]
            q1 = protocol["brief_prompt"] if mode == "brief" else prompts["prompt_ask_image"]
            torch.cuda.synchronize()
            a = time.perf_counter()
            with torch.inference_mode():
                response, history, _ = model.chat(tokenizer, pixels, q1,
                    dict(generation, output_scores=False), num_patches_list=patches, return_history=True)
                torch.cuda.synchronize()
                b = time.perf_counter()
                response2, _, scores = model.chat(tokenizer, pixels, prompts["prompt_compare_image"],
                    dict(generation), num_patches_list=patches, history=history, return_history=True)
                probabilities = torch.softmax(scores[0], dim=-1)[0]
                p_yes, p_no = float(probabilities[yes]), float(probabilities[no])
            torch.cuda.synchronize()
            c = time.perf_counter()
            selected = p_no - p_yes > cfg["threshold"]
            record = {"candidate": index, "reference": previous, "selected": selected,
                "p_yes": p_yes, "p_no": p_no, "psss": p_no-p_yes,
                "description": response, "answer": response2,
                "description_tokens_reencoded": len(tokenizer.encode(response, add_special_tokens=False)),
                "round2_generated_tokens": len(scores), "round1_seconds": b-a,
                "round2_seconds": c-b, "total_seconds": c-begin, "wall_seconds": time.time()-wall}
            records.append(record)
            if selected: chosen.append(index)
            final_keys = sorted(set(chosen + [window["frames"]-1]))
            write_json(destination, {"mode": mode, "window": name, "protocol_signature": protocol["signature"],
                "status": "COMPLETE" if index == queue[-1] else "RUNNING",
                "candidates": queue, "indices": final_keys, "records": records,
                "selection_seconds": sum(r["total_seconds"] for r in records),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            print(f'{mode}/{name}: {len(records)}/{len(queue)-1} keys={final_keys} {record["total_seconds"]:.2f}s', flush=True)
            del scores, probabilities, pixels, images, history
    print(f"SELECTION_COMPLETE {mode}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "select"])
    p.add_argument("--output", type=Path, default=REPO / "outputs/skem_speed_20260929")
    p.add_argument("--baseline", type=Path, default=REPO / "outputs/etri_60s_tv_low_08_42057b2ee8ed")
    p.add_argument("--mode", choices=MODES)
    args = p.parse_args()
    if args.action == "prepare": prepare(args.output.resolve(), args.baseline.resolve())
    else:
        if not args.mode: p.error("select requires --mode")
        select(args.output.resolve(), args.mode)


if __name__ == "__main__":
    main()
