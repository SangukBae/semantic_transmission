"""Repeat one identical frame pair after the user closed the game.

Separate evidence from the earlier noisy multi-window run. This measures a
single comparison, not full selector throughput or reconstruction quality.
"""
import argparse
import ast
import datetime
import importlib.util
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

from benchmark_skem_speed import REPO, SOURCE, validate
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.webvid5 import fingerprint, read_json


def gpu_context():
    return subprocess.check_output(["nvidia-smi",
        "--query-gpu=timestamp,memory.used,memory.free,utilization.gpu,power.draw",
        "--format=csv,noheader"], text=True).strip()


def worker(root, phase, repeats):
    protocol = validate(root)
    mode = "baseline" if phase.startswith("baseline") else phase
    sys.path.insert(0, str(REPO / ".local/vendor/InternVL"))
    if mode == "int8":
        sys.path.insert(0, str(REPO / ".local/selector_speed_deps"))
    import torch
    from transformers import AutoModel, AutoTokenizer
    from semantic_transmission.internvl_memory import place_internvl
    prompts = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.walk(ast.parse(SOURCE.read_text()))
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("prompt_ask_image", "prompt_compare_image")}
    spec = importlib.util.spec_from_file_location("official_skem", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = read_json(root / "inputs/people/config.json")
    result = {"phase": phase, "mode": mode, "status": "RUNNING", "records": [],
        "gpu_before_model": gpu_context(), "torch": torch.__version__}
    started = time.perf_counter()
    options = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                   use_flash_attn=cfg["flash_attn"], trust_remote_code=True)
    if mode == "int8":
        options.update(load_in_8bit=True, device_map={"": 0})
    torch.manual_seed(cfg["seed"])
    model = AutoModel.from_pretrained(cfg["models"]["internvl"], **options).eval()
    if mode != "int8":
        model = place_internvl(model, gpu_head=True, cpu_layers=3, compact_cache=True)
    model.chat = types.MethodType(module.custom_chat, model)
    tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["internvl"], trust_remote_code=True, use_fast=True)
    torch.cuda.synchronize()
    result["model_load_seconds"] = time.perf_counter() - started
    result["gpu_after_model"] = gpu_context()
    result["device_map"] = getattr(model, "hf_device_map", None)
    yes, no = [tokenizer.encode(word, add_special_tokens=False)[0] for word in ("Yes", "No")]
    generation = dict(max_new_tokens=256 if mode == "brief" else 1024, do_sample=False,
                      output_scores=True, return_dict_in_generate=True)
    destination = root / f"timing_recheck/{phase}.json"
    for iteration in range(repeats + 1):
        # Fresh image preparation and tensor per pair; only the within-pair
        # BF16 visual-feature reuse installed by place_internvl is retained.
        torch.manual_seed(cfg["seed"])
        torch.cuda.synchronize()
        context = gpu_context()
        begin = time.perf_counter()
        images = [module.load_image(str(root / f"inputs/people/frames/{i}.png"),
                    max_num=cfg["max_tiles"]).to(torch.bfloat16).cuda() for i in (0, 6)]
        pixels = torch.cat(images)
        patches = [len(im) for im in images]
        q1 = protocol["brief_prompt"] if mode == "brief" else prompts["prompt_ask_image"]
        torch.cuda.synchronize()
        a = time.perf_counter()
        with torch.inference_mode():
            description, history, _ = model.chat(tokenizer, pixels, q1,
                dict(generation, output_scores=False), num_patches_list=patches, return_history=True)
            torch.cuda.synchronize()
            b = time.perf_counter()
            answer, _, scores = model.chat(tokenizer, pixels, prompts["prompt_compare_image"],
                dict(generation), num_patches_list=patches, history=history, return_history=True)
            probabilities = torch.softmax(scores[0], dim=-1)[0]
            p_yes, p_no = float(probabilities[yes]), float(probabilities[no])
        torch.cuda.synchronize()
        c = time.perf_counter()
        record = {"iteration": iteration, "warmup": iteration == 0, "gpu_before_pair": context,
            "gpu_after_pair": gpu_context(), "total_seconds": c - begin,
            "round1_seconds": b - a, "round2_seconds": c - b,
            "description": description, "answer": answer, "p_yes": p_yes, "p_no": p_no,
            "psss": p_no - p_yes, "selected": p_no - p_yes > cfg["threshold"],
            "description_tokens_reencoded": len(tokenizer.encode(description, add_special_tokens=False))}
        result["records"].append(record)
        result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["status"] = "COMPLETE" if iteration == repeats else "RUNNING"
        write_json(destination, result)
        print(f"{phase} iteration {iteration}: {c-begin:.3f}s", flush=True)
        del scores, probabilities, pixels, images, history


def summarize(root):
    rows = []
    for phase in ("baseline_before", "int8", "brief", "baseline_after"):
        result = read_json(root / f"timing_recheck/{phase}.json")
        if result["status"] != "COMPLETE":
            raise ValueError(f"incomplete timing phase: {phase}")
        records = [r for r in result["records"] if not r["warmup"]]
        times = [r["total_seconds"] for r in records]
        rows.append({"phase": phase, "timed_repeats": len(times),
            "median_seconds": statistics.median(times), "range_seconds": [min(times), max(times)],
            "description_tokens": [r["description_tokens_reencoded"] for r in records],
            "same_output_within_phase": len({(r["description"], r["answer"], r["psss"]) for r in records}) == 1,
            "psss": [r["psss"] for r in records], "selected": [r["selected"] for r in records]})
    historical = read_json(root / "selection/historical_baseline/people.json")
    old = next(r for r in historical["records"] if r["candidate"] == 6 and r["reference"] == 0)
    comparisons = {phase: all(r["description"] == old["description"] and r["psss"] == old["psss"]
        for r in read_json(root / f"timing_recheck/{phase}.json")["records"])
        for phase in ("baseline_before", "baseline_after")}
    summary = {"status": "SINGLE_PAIR_RECHECK_COMPLETE", "conditions": rows,
        "baseline_matches_historical_output": comparisons,
        "scope": "One fixed pair (global frames 217,223), greedy, one warmup per phase. Not full-video throughput.",
        "game_state": "User reported game closed before this run; no other apps controlled or causal attribution."}
    write_json(root / "timing_recheck/SUMMARY.json", summary)
    print(summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=REPO / "outputs/skem_speed_20260929")
    p.add_argument("--phase", choices=("baseline_before", "int8", "brief", "baseline_after"))
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    if args.repeats < 1:
        p.error("--repeats must be positive")
    root = args.output.resolve()
    if args.phase:
        worker(root, args.phase, args.repeats)
        return
    from semantic_transmission.cli import settings
    from semantic_transmission.etri_60s_check import lock, run_command
    from semantic_transmission.webvid_ablation import environment
    folder = root / "timing_recheck"
    with lock(REPO / ".local/etri_60s_check.lock"):
        folder.mkdir()
        protocol = {"selection_protocol": validate(root)["signature"],
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "pair": {"window": "people", "local": [0, 6], "global": [217, 223]},
            "code_sha256": sha256(Path(__file__)), "warmup_per_phase": 1,
            "phases": {"baseline_before": 3, "int8": 3, "brief": 3, "baseline_after": 1}}
        write_json(folder / "protocol.json", dict(protocol, signature=fingerprint(protocol)))
        for phase, repeats in protocol["phases"].items():
            run_command(REPO, [settings(REPO)["internvl_python"], str(Path(__file__)), "--output", str(root),
                "--phase", phase, "--repeats", str(repeats)], folder / f"{phase}.log", environment(2025),
                folder / f"{phase}_resources.json", folder / f"{phase}_progress.json")
        summarize(root)


if __name__ == "__main__":
    main()
