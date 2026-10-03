"""Precompute receiver T5 conditions on GPU, then serve immutable CPU tensors.

Preparation runs in its own process: T5 never shares VRAM with the decoder.
GPU BF16 is a numerical change from the legacy CPU FP32 encoder, not a claim
of identical reconstruction quality. Cache misses during decoding are errors.
"""
import ast
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import time

from .artifacts import sha256, write_json
from .webvid5 import fingerprint, read_json

POLICY = "t5_gpu_bf16_persistent_v1"
REPORT = "receiver/text_embeddings.json"
TRACE = "receiver/text_embedding_trace.json"
CODE = (
    "src/semantic_transmission/text_embedding_cache.py",
    "scripts/etri_t5_cached_decoder.py",
    "04_semantic_decoder/scripts/mydemo_new_align_sh.py",
    ".local/vendor/Open-Sora/opensora/models/text_encoder/t5.py",
    ".local/vendor/Open-Sora/opensora/utils/inference_utils.py",
)


def model_inventory(path):
    """Cheap read-only resume guard; the preparation contract also hashes bytes."""
    root = Path(path).resolve(strict=True)
    files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix in {".json", ".model", ".bin", ".safetensors"})
    if not (root / "config.json").is_file() or not any(p.suffix in {".bin", ".safetensors"} for p in files):
        raise ValueError("local T5 configuration and weights are required")
    return {p.name: dict(target=str(p.resolve()), size=p.stat().st_size,
                        mtime_ns=p.stat().st_mtime_ns, ctime_ns=p.stat().st_ctime_ns) for p in files}


def prepare_prompts(csv_path, cfg, repo):
    """Use the released score helper and the exact decoder cleaning order."""
    import pandas as pd
    from opensora.models.text_encoder.t5 import text_preprocessing
    from opensora.utils.inference_utils import split_prompt, merge_prompt, extract_prompts_loop
    if cfg.get("llm_refine", False) or cfg.get("batch_size", 1) != 1:
        raise ValueError("text cache requires batch_size=1 and no online prompt refinement")
    # Load just this pure function, without importing/running the video generator.
    source = repo / "04_semantic_decoder/scripts/mydemo_new_align_sh.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "append_multi_score_to_prompts")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    append_scores = namespace[node.name]
    df = pd.read_csv(csv_path)
    if list(df.columns) != ["path", "text", "flow"] or df.empty or df.isnull().any().any():
        raise ValueError("invalid receiver caption CSV")
    groups = {}
    for row in df.to_dict("records"):
        if not isinstance(row["text"], str) or "|" in row["text"] or "{" in row["text"]:
            raise ValueError("unsupported caption control syntax in text cache")
        groups.setdefault(Path(row["path"]).parent.name, []).append(row)
    prompts = []
    for rows in groups.values():
        segments, indices = split_prompt("".join(f'|{i}|{r["text"]}' for i, r in enumerate(rows)))
        scored = append_scores(segments, aes=cfg.get("aes"), flow=[r["flow"] for r in rows],
                               camera_motion=cfg.get("camera_motion"))
        merged = merge_prompt([text_preprocessing(p) for p in scored], indices)
        prompts.extend(extract_prompts_loop([merged], i)[0] for i in range(len(rows)))
    return prompts


def validate_tensors(tensors, length, dim):
    import torch
    if set(tensors) != {"y", "mask"}:
        raise ValueError("invalid T5 tensor names")
    y, mask = tensors["y"], tensors["mask"]
    if (tuple(y.shape) != (1, 1, length, dim) or y.dtype != torch.bfloat16
            or tuple(mask.shape) != (1, length) or mask.dtype != torch.int64
            or not torch.isfinite(y).all() or not ((mask == 0) | (mask == 1)).all()
            or not mask.any()):
        raise ValueError("invalid T5 tensor shape, precision, or values")


def load_entry(entry, signature, prompt, length, dim):
    import torch
    path = Path(entry["path"])
    if entry["contract"] != signature or entry["prompt"] != prompt or sha256(path) != entry["sha256"]:
        raise ValueError("text embedding cache identity/hash mismatch")
    tensors = torch.load(path, map_location="cpu", weights_only=True)
    validate_tensors(tensors, length, dim)
    return tensors


def save_entry(path, tensors, signature, prompt):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".tensor-", suffix=".pt")
    os.close(fd)
    try:
        torch.save(tensors, name)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
    entry = dict(path=str(path.resolve()), contract=signature, prompt=prompt, sha256=sha256(path))
    write_json(path.with_suffix(".json"), entry)
    return entry


def populate(prompts, contract, cache_root, factory, synchronize=lambda: None):
    """Lazy model construction, one caption per forward, resumable per prompt."""
    import torch
    signature = fingerprint(contract)
    length, dim = contract["max_length"], contract["output_dim"]
    start = time.monotonic()
    unique = list(dict.fromkeys(prompts))
    entries, missing = {}, []
    for prompt in unique:
        key = fingerprint(prompt)
        receipt = cache_root / signature / f"{key}.json"
        if receipt.exists():
            entry = read_json(receipt)
            load_entry(entry, signature, prompt, length, dim)
            entries[prompt] = entry
        else:
            missing.append(prompt)
    lookup_seconds = time.monotonic() - start
    model = None
    load_seconds = encode_seconds = 0.0
    try:
        if missing:
            started = time.monotonic()
            model = factory()
            synchronize()
            load_seconds = time.monotonic() - started
            started = time.monotonic()
            with torch.inference_mode():
                for prompt in missing:
                    tensors = {name: t.detach().cpu().contiguous() for name, t in model.encode([prompt]).items()}
                    validate_tensors(tensors, length, dim)
                    path = cache_root / signature / f"{fingerprint(prompt)}.pt"
                    entries[prompt] = save_entry(path, tensors, signature, prompt)
            synchronize()
            encode_seconds = time.monotonic() - started
    finally:
        del model
        gc.collect()
    return dict(policy=POLICY, contract=contract, signature=signature,
        prompts=prompts, entries=[entries[p] for p in unique], prompt_count=len(prompts),
        unique_prompts=len(unique), cache_hits=len(unique)-len(missing), cache_misses=len(missing),
        model_loaded=bool(missing), lookup_seconds=lookup_seconds, model_load_seconds=load_seconds,
        encode_and_save_seconds=encode_seconds, total_seconds=time.monotonic()-start,
        legacy_cpu_fp32_equivalence_verified=False, reconstruction_quality_verified=False)


def prepare(run, repo, config_text):
    # Environment must be set before CUDA initialization. This worker exits before
    # launching the decoder; even an exception releases its GPU allocations.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import sys
    sys.path.insert(0, str(repo / ".local/vendor/Open-Sora"))
    import torch
    from mmengine import Config
    from opensora.models.text_encoder.t5 import T5Encoder
    cfg = Config.fromstring(config_text, file_format=".py")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU T5 preparation requires a CUDA GPU supporting BF16")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(cfg.seed)
    prompts = prepare_prompts(run / "receiver/metadata.csv", cfg, repo)
    encoder = dict(cfg.text_encoder)
    if encoder.pop("type") != "t5" or set(encoder) != {"from_pretrained", "model_max_length"}:
        raise ValueError("unsupported T5 configuration; do not reuse incompatible conditions")
    model_path = Path(encoder["from_pretrained"])
    inventory = model_inventory(model_path)
    contract = dict(policy=POLICY, model_path=str(model_path.resolve()), model_inventory=inventory,
        model_sha256={name: sha256(model_path / name) for name in inventory},
        code={name: sha256(repo / name) for name in CODE},
        packages={n: importlib.metadata.version(n) for n in ("torch", "transformers", "tokenizers", "ftfy", "beautifulsoup4")},
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), dtype="bf16", batch_size=1,
        deterministic=True, tf32=False, max_length=encoder["model_max_length"],
        output_dim=read_json(model_path / "config.json")["d_model"])
    cache_root = repo / ".local/cache/t5_embeddings"
    report = populate(prompts, contract, cache_root,
        lambda: T5Encoder(**encoder, device="cuda", dtype=torch.bfloat16, local_files_only=True),
        torch.cuda.synchronize)
    torch.cuda.empty_cache()
    if model_inventory(model_path) != inventory:
        raise ValueError("T5 checkpoint changed during preparation")
    report.update(status="PASSED", metadata_sha256=sha256(run / "receiver/metadata.csv"),
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                  remaining_gpu_allocated_bytes=torch.cuda.memory_allocated())
    write_json(run / REPORT, report)
    print(json.dumps({k: report[k] for k in ("status", "prompt_count", "cache_hits", "cache_misses",
        "model_load_seconds", "encode_and_save_seconds", "peak_gpu_allocated_bytes",
        "remaining_gpu_allocated_bytes")}), flush=True)


class CachedTextEncoder:
    """T5 interface used by RFLOW; unconditional CFG stays in STDiT's embedder."""
    def __init__(self, report, trace=None):
        contract = report["contract"]
        if (report.get("status") != "PASSED" or report["policy"] != POLICY
                or fingerprint(contract) != report["signature"]):
            raise ValueError("invalid text embedding preparation report")
        self.model_max_length, self.output_dim = contract["max_length"], contract["output_dim"]
        self.y_embedder = None
        self.trace = trace
        self.calls = []
        self.entries = {e["prompt"]: load_entry(e, report["signature"], e["prompt"], self.model_max_length,
                        self.output_dim) for e in report["entries"]}
        if set(self.entries) != set(report["prompts"]) or len(report["prompts"]) != report["prompt_count"]:
            raise ValueError("text embedding report has missing prompts")

    def encode(self, texts):
        import torch
        if not texts or any(p not in self.entries for p in texts):
            raise ValueError("unprepared decoder prompt; rerun text preparation")
        self.calls.extend(texts)
        if self.trace is not None:
            write_json(self.trace, dict(policy=POLICY, calls=len(self.calls),
                prompt_sha256=[fingerprint(p) for p in self.calls], t5_model_loaded_in_decoder=False))
        # New tensors/dict on every call: RFLOW mutates model_args for CFG.
        return {name: torch.cat([self.entries[p][name] for p in texts], dim=0) for name in ("y", "mask")}

    def null(self, n):
        return self.y_embedder.y_embedding[None].repeat(n, 1, 1)[:, None]


def install_cached_encoder(report_path, trace, repo):
    from opensora import registry
    report = read_json(report_path)
    contract = report["contract"]
    if (model_inventory(contract["model_path"]) != contract["model_inventory"]
            or any(sha256(repo / name) != digest for name, digest in contract["code"].items())):
        raise ValueError("text cache model/code changed after preparation")
    original = registry.build_module
    used = False

    def build(config, reg, **kwargs):
        nonlocal used
        if isinstance(config, dict) and config.get("type") == "t5":
            if used or dict(config) != dict(type="t5", from_pretrained=contract["model_path"],
                                             model_max_length=contract["max_length"]):
                raise ValueError("decoder T5 configuration differs from prepared cache")
            if kwargs.get("device") != "cpu":
                raise ValueError("cached T5 adapter requires the existing CPU tensor-transfer wrapper")
            used = True
            return CachedTextEncoder(report, trace)
        return original(config, reg, **kwargs)
    registry.build_module = build
    return report


def validate_usage(run):
    report, trace = read_json(run / REPORT), read_json(run / TRACE)
    if (report["metadata_sha256"] != sha256(run / "receiver/metadata.csv")
            or trace["policy"] != POLICY or trace["t5_model_loaded_in_decoder"]
            or trace["calls"] != report["prompt_count"]
            or trace["prompt_sha256"] != [fingerprint(p) for p in report["prompts"]]):
        raise ValueError("decoder did not use every prepared text condition in order")
