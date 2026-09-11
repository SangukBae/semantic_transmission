"""Stress only: exercise SKEM with the maximum allowed first-round history.

Run with the lgvsc-internvl Python. This does not select research keyframes and
never changes a reconstruction's captions, thresholds or generation settings.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys
import types
import time

import torch
from transformers import AutoModel, AutoTokenizer

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo / ".local/vendor/InternVL"))
from semantic_transmission.artifacts import write_json
from semantic_transmission.internvl_memory import place_internvl

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_dir", type=Path, help="prepared probe directory with normalized frames")
parser.add_argument("--force-max-generation", action="store_true")
args = parser.parse_args()
started = time.monotonic()
cfg = json.loads((args.run_dir / "run_config.json").read_text())
source = repo / "02_semantic_encoder/skem/MLM-keyframe-internvl.py"
tree = ast.parse(source.read_text())
prompts = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.walk(tree)
           if isinstance(node, ast.Assign) and len(node.targets) == 1 and
           isinstance(node.targets[0], ast.Name) and
           node.targets[0].id in ("prompt_ask_image", "prompt_compare_image")}
spec = importlib.util.spec_from_file_location("skem_memory_probe", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model = AutoModel.from_pretrained(cfg["models"]["internvl"], torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True).eval()
model = place_internvl(model, gpu_head=True, cpu_layers=cfg.get("internvl_offload_layers", 0))
model.chat = types.MethodType(module.custom_chat, model)
tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["internvl"], trust_remote_code=True, use_fast=True)
frames = args.run_dir / "data/frames/sample"
images = [module.load_image(str(frames / f"{i}.png"), max_num=12).to("cuda", torch.bfloat16)
          for i in (0, 4)]
pixels = torch.cat(images)
tokens = tokenizer.encode("A person walks on a sidewalk. " * 300, add_special_tokens=False)[:1024]
history = [(prompts["prompt_ask_image"], tokenizer.decode(tokens))]
generation = dict(max_new_tokens=1024, do_sample=False, output_scores=True, return_dict_in_generate=True)
if args.force_max_generation:
    generation["min_new_tokens"] = 1024
response, _, scores = model.chat(tokenizer, pixels, prompts["prompt_compare_image"], generation,
    num_patches_list=[len(image) for image in images], history=history, return_history=True)
# Forced min length masks EOS with -inf, so test the actual PSSS probability boundary.
probabilities = torch.softmax(scores[0], dim=-1)
assert torch.isfinite(probabilities).all()
report = {"status": "PASSED", "scope": "synthetic maximum-context memory stress; not research output",
          "history_tokens": len(tokens), "generated_tokens": len(scores),
          "force_max_generation": args.force_max_generation, "torch": torch.__version__,
          "seconds": time.monotonic() - started,
          "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
          "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "finite_psss_distribution": True}
write_json(args.run_dir / "memory_stress.json", report)
print(json.dumps(report, indent=2))
