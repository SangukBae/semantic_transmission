"""One-load, resumable inference on the frozen FC/PLLaVA comparison inputs.

The explicit confirmed model argument prevents labelling installed Qwen3.5 weights
as the originally requested Qwen3.6 model. This script does not score captions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import qwen35_caption as local


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--confirmed-model", required=True, choices=[local.MODEL_ID])
    parser.add_argument("--limit", type=int, help="Initial sequential probe; resume without this flag")
    args = parser.parse_args()
    root = args.root.resolve()
    inputs_file = root / "inputs.json"
    dataset = json.loads(inputs_file.read_text())
    for name, expected in dataset["input_files_sha256"].items():
        if local.sha256(name) != expected:
            raise RuntimeError(f"Frozen comparison input changed: {name}")
    source = Path(dataset["source_frames"])
    for frame, expected in dataset["source_png_hashes"].items():
        if local.sha256(source / f"{frame}.png") != expected:
            raise RuntimeError(f"Source frame changed: {frame}")
    identity = dict(model_id=args.confirmed_model, revision=local.REVISION,
                    quantization="nf4", dtype="bfloat16", attention="sdpa", device="cuda:0",
                    enable_thinking=False, do_sample=False, seed=2025, cpu_threads=2,
                    max_pixels=262144, max_new_tokens=256, truncated_retry_tokens=512,
                    prompt=dataset["prompt"], inputs_sha256=local.sha256(inputs_file),
                    inference_code_sha256=local.sha256(__file__),
                    helper_code_sha256=local.sha256(local.REPO / "scripts/qwen35_caption.py"))
    identity["signature"] = fingerprint(identity)
    config_file = root / "inference_config.json"
    if config_file.exists():
        if json.loads(config_file.read_text()) != identity:
            raise RuntimeError("Inference configuration changed; use a new experiment root")
    else:
        local.write_json(config_file, identity)
    destination = root / "captions"
    destination.mkdir(exist_ok=True)
    pending = []
    for row in dataset["records"]:
        file = destination / f"{row['segment']:03d}.json"
        if file.exists():
            old = json.loads(file.read_text())
            if old["signature"] != identity["signature"] or old["source_indices"] != row["source_indices"]:
                raise RuntimeError("Caption checkpoint identity mismatch")
            if old["truncated"] or not old["caption"].strip():
                raise RuntimeError("Incomplete caption checkpoint")
        else:
            pending.append(row)
    if args.limit is not None:
        pending = pending[:args.limit]
    if not pending:
        print("No pending captions.")
        return
    state = local.gpu_status()
    if not state.get("gpus") or state["gpus"][0]["free_mib"] < 12000:
        raise RuntimeError(f"GPU requires at least 12000 MiB free: {state}")
    import torch
    from PIL import Image
    from transformers import BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
    torch.set_num_threads(2)
    torch.manual_seed(2025)
    processor = local.load_processor()
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["visual", "lm_head"])
    started = time.monotonic()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(local.SNAPSHOT,
        local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, quantization_config=quantization,
        attn_implementation="sdpa").eval()
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - started
    versions = local.packages()
    local.write_json(root / "runtime.json", dict(created_utc=local.now(), versions=versions,
        gpu_before=state, model_load_seconds=load_seconds, gpu=local.gpu_status(),
        resident_allocated_bytes=torch.cuda.memory_allocated(), signature=identity["signature"]))
    for row in pending:
        images = []
        for index in row["source_indices"]:
            with Image.open(source / f"{index}.png") as image:
                images.append(image.convert("RGB"))
        labels = [f"Source frame {i}, timestamp {i / dataset['fps']:.3f} seconds"
                  for i in row["source_indices"]]
        data = local.preprocess(processor, images, labels, identity["prompt"], identity["max_pixels"])
        if data.input_ids.shape[1] > 4096:
            raise RuntimeError("Local profile input token cap exceeded")
        data = data.to("cuda:0")
        attempts = []
        for budget in [identity["max_new_tokens"], identity["truncated_retry_tokens"]]:
            torch.cuda.reset_peak_memory_stats()
            begin = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(**data, max_new_tokens=budget, do_sample=False, use_cache=True)
            torch.cuda.synchronize()
            tokens = generated[0, data.input_ids.shape[1]:]
            text = processor.decode(tokens, skip_special_tokens=True).strip()
            attempt = dict(caption=text, generated_tokens=len(tokens), truncated=len(tokens) >= budget,
                generation_seconds=time.monotonic() - begin, max_new_tokens=budget,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
            attempts.append(attempt)
            del generated, tokens
            if text and not attempt["truncated"]:
                break
        record = dict(segment=row["segment"], matched_pair=row["matched_pair"],
            start=row["start"], end_exclusive=row["end_exclusive"], source_indices=row["source_indices"],
            signature=identity["signature"], created_utc=local.now(), model_id=local.MODEL_ID,
            model_revision=local.REVISION, input_tokens=data.input_ids.shape[1],
            inference_provider="LOCAL_ONLY", attempts=attempts, **attempt)
        if not text or attempt["truncated"]:
            local.write_json(root / "incomplete" / f"{row['segment']:03d}.json", record)
            raise RuntimeError("Generation failed or reached the final token limit")
        local.write_json(destination / f"{row['segment']:03d}.json", record)
        completed = len(list(destination.glob("*.json")))
        progress = dict(status="RUNNING", completed=completed, total=len(dataset["records"]),
                        last_segment=row["segment"], updated_utc=local.now(),
                        last_generation_seconds=attempt["generation_seconds"])
        local.write_json(root / "progress.json", progress)
        print(json.dumps({**progress, "caption": text}, ensure_ascii=False), flush=True)
        del data, images
    completed = len(list(destination.glob("*.json")))
    local.write_json(root / "progress.json", dict(status="CAPTIONS_COMPLETE_REVIEW_PENDING"
        if completed == len(dataset["records"]) else "SEQUENTIAL_PROBE_COMPLETE",
        completed=completed, total=len(dataset["records"]), updated_utc=local.now()))


if __name__ == "__main__":
    main()
