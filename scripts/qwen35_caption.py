"""Offline Qwen3.5-9B image/frame captioning in an isolated environment.

The default is NF4 inference on one GPU. Check free memory before loading weights;
never stop another process or change the existing LGVSC caption pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
CACHE = REPO / ".local/qwen35_hf"
SNAPSHOT = CACHE / "models--Qwen--Qwen3.5-9B" / "snapshots" / REVISION
REPORTS = REPO / "outputs/qwen35_setup"
PROMPT = (
    "These source video frames are in chronological order. Write a concise English "
    "caption of at most 80 words describing only visible objects, their positions, "
    "actions, and changes across the supplied frames. Distinguish camera movement "
    "from object movement. Do not invent identities, intentions, sounds, unseen "
    "events, or details that are unclear. A still image does not establish motion. "
    "Return only the caption."
)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu_status():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,memory.used",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
        )
        rows = []
        for line in result.stdout.strip().splitlines():
            index, name, total, free, used = [v.strip() for v in line.split(",")]
            rows.append(dict(index=int(index), name=name, total_mib=int(total),
                             free_mib=int(free), used_mib=int(used)))
        return {"gpus": rows}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": str(exc), "detail": getattr(exc, "stderr", "")}


def packages():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "torchvision", "transformers", "accelerate", "bitsandbytes",
             "huggingface-hub", "safetensors", "Pillow", "numpy")}


def download():
    from huggingface_hub import HfApi, snapshot_download
    from safetensors import safe_open
    path = Path(snapshot_download(
        MODEL_ID, revision=REVISION, cache_dir=CACHE, max_workers=2,
        allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors", "LICENSE", "README.md"],
    ))
    index = json.loads((path / "model.safetensors.index.json").read_text())
    expected = index["weight_map"]
    info = HfApi().model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    remote_hashes = {file.rfilename: file.lfs.sha256 for file in info.siblings if file.lfs}
    found = {}
    manifest = []
    for name in sorted(set(expected.values())):
        file = path / name
        with safe_open(file, framework="pt", device="cpu") as f:
            for key in f.keys():
                found[key] = name
        digest = sha256(file)
        expected_digest = remote_hashes.get(name)
        if digest != expected_digest:
            raise RuntimeError(f"Weight SHA-256 differs from the Hub blob identifier: {name}")
        manifest.append(dict(file=name, size=file.stat().st_size, sha256=digest,
                             hub_lfs_sha256_verified=True))
    if found != expected:
        raise RuntimeError("Downloaded tensor keys do not match the official weight index")
    for file in sorted(path.iterdir()):
        if file.is_file() and not file.name.endswith(".safetensors"):
            manifest.append(dict(file=file.name, size=file.stat().st_size, sha256=sha256(file)))
    record = dict(status="OFFICIAL_WEIGHTS_DOWNLOADED_AND_HASHED", created_utc=now(),
                  model_id=MODEL_ID, revision=REVISION, model_path=str(path),
                  tensor_count=len(found), files=manifest)
    write_json(REPORTS / "model_manifest.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "files"}, indent=2))


def load_processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)


def messages_for(images, labels, prompt):
    content = []
    for image, label in zip(images, labels):
        content.extend([{"type": "text", "text": label}, {"type": "image", "image": image}])
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def preprocess(processor, images, labels, prompt, max_pixels):
    text = processor.apply_chat_template(messages_for(images, labels, prompt), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    return processor(text=[text], images=images, return_tensors="pt",
                     images_kwargs={"size": {"shortest_edge": 32 * 32 * 4,
                                             "longest_edge": max_pixels}})


def doctor():
    import torch
    import bitsandbytes
    from PIL import Image
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration
    config = AutoConfig.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
    processor = load_processor()
    sample = Image.new("RGB", (320, 192), (100, 120, 140))
    inputs = preprocess(processor, [sample] * 4, [f"Frame {i}" for i in range(4)], PROMPT,
                        256 * 32 * 32)
    manifest_path = REPORTS / "model_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run download to complete and validate the weight snapshot first")
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["files"]:
        file = SNAPSHOT / item["file"]
        if not file.is_file() or file.stat().st_size != item["size"]:
            raise RuntimeError(f"Missing or incomplete model file: {file}")
    record = dict(status="ENVIRONMENT_WEIGHTS_AND_PREPROCESSOR_READY", created_utc=now(),
                  model_id=MODEL_ID, revision=REVISION, python=sys.executable,
                  versions=packages(), model_class=Qwen3_5ForConditionalGeneration.__name__,
                  config_model_type=config.model_type,
                  input_shapes={k: list(v.shape) for k, v in inputs.items()},
                  cuda_available=torch.cuda.is_available(), torch_cuda=torch.version.cuda,
                  gpu=gpu_status(), full_model_inference="NOT_TESTED_BY_DOCTOR")
    write_json(REPORTS / "doctor.json", record)
    print(json.dumps(record, indent=2))


def source_frames(args):
    from PIL import Image
    if args.images:
        if len(args.images) > 8:
            raise ValueError("Use at most 8 images per call on the 16GB profile")
        images, records, labels = [], [], []
        for i, name in enumerate(args.images):
            path = Path(name).resolve()
            with Image.open(path) as im:
                images.append(im.convert("RGB"))
            records.append(dict(path=str(path), sha256=sha256(path), ordinal=i))
            labels.append(f"Source frame {i + 1} (chronological input order)")
        return images, labels, records
    import cv2
    path = Path(args.video).resolve()
    cap = cv2.VideoCapture(str(path))
    try:
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or fps <= 0 or count < 1:
            raise ValueError(f"Cannot decode video: {path}")
        start = int(args.start * fps)
        end = min(count, int(args.end * fps)) if args.end is not None else count
        if not 0 <= start < end <= count:
            raise ValueError("Invalid video interval")
        # Same four-frame segment sampling rule as the existing PLLaVA path.
        if args.frames == 4:
            size = (end - start - 1) / 4
            indices = [start + int(size / 2) + round(size * i) for i in range(4)]
        else:
            indices = [min(end - 1, start + int((i + .5) * (end - start) / args.frames))
                       for i in range(args.frames)]
        images = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode source frame {index}")
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        labels = [f"Source frame {i}, timestamp {i / fps:.3f} seconds" for i in indices]
        return images, labels, [dict(path=str(path), sha256=sha256(path), fps=fps,
                                     interval_frames=[start, end], sampled_indices=indices)]
    finally:
        cap.release()


def caption(args):
    import resource
    import torch
    from transformers import BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
    if args.output.exists():
        raise FileExistsError(f"Output exists; choose a new path: {args.output}")
    # Guard first: neither model loading nor CUDA allocations while SKEM uses the GPU.
    state = gpu_status()
    if args.device == "cuda":
        rows = state.get("gpus", [])
        if not rows or rows[0]["free_mib"] < args.min_free_mib:
            print(json.dumps(dict(status="GPU_BUSY_OR_UNAVAILABLE", required_free_mib=args.min_free_mib,
                                  gpu=state), indent=2), file=sys.stderr)
            return 75
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this process")
    else:
        memory = {line.split(":")[0]: int(line.split()[1]) for line in
                  Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")}
        if memory.get("MemAvailable", 0) < 11 * 1024 * 1024:
            raise RuntimeError("CPU inference requires at least 11 GiB available system RAM")
    torch.set_num_threads(args.threads)
    torch.manual_seed(2025)
    images, labels, sources = source_frames(args)
    processor = load_processor()
    inputs = preprocess(processor, images, labels, args.prompt, args.max_pixels)
    if inputs.input_ids.shape[1] > 4096:
        raise ValueError("Input exceeds the 4096-token limit of this local profile")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["visual", "lm_head"],
    )
    started = time.monotonic()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        SNAPSHOT, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
        device_map={"": args.device}, quantization_config=quantization,
        attn_implementation="sdpa",
    ).eval()
    load_seconds = time.monotonic() - started
    inputs = inputs.to(args.device)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                use_cache=True)
    if args.device == "cuda":
        torch.cuda.synchronize()
    tokens = output[0, inputs.input_ids.shape[1]:]
    text = processor.decode(tokens, skip_special_tokens=True).strip()
    if not text:
        raise RuntimeError("Model returned an empty caption")
    record = dict(status="CAPTION_GENERATED", created_utc=now(), model_id=MODEL_ID,
                  revision=REVISION, quantization="nf4", device=args.device, caption=text,
                  sources=sources, prompt=args.prompt, versions=packages(),
                  input_tokens=inputs.input_ids.shape[1], generated_tokens=len(tokens),
                  truncated=len(tokens) >= args.max_new_tokens, load_seconds=load_seconds,
                  generation_seconds=time.monotonic() - started, max_pixels=args.max_pixels,
                  image_count=len(images), inference_provider="LOCAL_ONLY", gpu_before=state,
                  cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  quality_verified=False)
    if args.device == "cuda":
        record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write_json(args.output, record)
    print(text)
    print(f"Saved: {args.output}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="Download and hash official pinned weights (network required)")
    sub.add_parser("doctor", help="Validate packages, weights config and four-image preprocessing")
    p = sub.add_parser("caption", help="Generate a caption offline; exit 75 if GPU memory is busy")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", nargs="+", help="Chronological source images")
    source.add_argument("--video", type=Path)
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--end", type=float)
    p.add_argument("--frames", type=int, choices=range(1, 9), default=4)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--max-pixels", type=int, default=256 * 32 * 32)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--min-free-mib", type=int, default=12000)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.command == "download":
        download()
    elif args.command == "doctor":
        doctor()
    else:
        if args.max_new_tokens < 1 or args.max_new_tokens > 512 or args.max_pixels < 4096 or args.max_pixels > 262144:
            parser.error("Use 1..512 output tokens and 4096..262144 max pixels for this profile")
        if args.start < 0 or (args.end is not None and args.end <= args.start):
            parser.error("Use a nonnegative start and an end greater than start")
        if not 1 <= args.threads <= 8 or args.min_free_mib < 12000:
            parser.error("Use 1..8 CPU threads and a GPU free-memory threshold of at least 12000 MiB")
        return caption(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
