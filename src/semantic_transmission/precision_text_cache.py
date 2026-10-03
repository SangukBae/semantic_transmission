"""FP32 T5 preparation without changing frozen BF16 cache code or receipts."""
import gc
import importlib.metadata
from pathlib import Path
import time

from . import text_embedding_cache as legacy
from .artifacts import sha256, write_json
from .webvid5 import fingerprint, read_json

POLICY = "t5_explicit_compute_precision_v1"
REPORT, TRACE = legacy.REPORT, legacy.TRACE
CODE = (*legacy.CODE, "src/semantic_transmission/precision_text_cache.py")


def populate(prompts, contract, root, factory):
    """Cache actual FP32/BF16 outputs; never upcast an old BF16 embedding."""
    import torch
    signature, start = fingerprint(contract), time.monotonic()
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[contract["compute_dtype"]]
    entries, missing = {}, []
    for prompt in dict.fromkeys(prompts):
        path = root / signature / f"{fingerprint(prompt)}.json"
        if path.exists():
            entry = read_json(path)
            load(entry, contract, prompt)
            entries[prompt] = entry
        else:
            missing.append(prompt)
    model = None
    try:
        if missing:
            model = factory()
            with torch.inference_mode():
                for prompt in missing:
                    tensors = {k: v.detach().cpu().contiguous() for k, v in model.encode([prompt]).items()}
                    validate(tensors, contract)
                    assert tensors["y"].dtype == dtype
                    entries[prompt] = legacy.save_entry(root / signature / f"{fingerprint(prompt)}.pt",
                                                         tensors, signature, prompt)
    finally:
        del model
        gc.collect()
    return dict(status="PASSED", policy=POLICY, contract=contract, signature=signature,
        prompts=prompts, entries=list(entries.values()), prompt_count=len(prompts),
        cache_hits=len(entries)-len(missing), cache_misses=len(missing),
        model_loaded=bool(missing), total_seconds=time.monotonic()-start)


def validate(tensors, contract):
    import torch
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[contract["compute_dtype"]]
    if set(tensors) != {"y", "mask"} or tensors["y"].dtype != dtype:
        raise ValueError("T5 cache computation precision mismatch")
    # The legacy validator checks dimensions, finite values and binary masks.
    if not torch.isfinite(tensors["y"]).all():
        raise ValueError("non-finite T5 condition")
    legacy.validate_tensors(dict(y=tensors["y"].to(torch.bfloat16), mask=tensors["mask"]),
                            contract["max_length"], contract["output_dim"])


def load(entry, contract, prompt):
    import torch
    if (entry["prompt"] != prompt or entry["contract"] != fingerprint(contract)
            or sha256(Path(entry["path"])) != entry["sha256"]):
        raise ValueError("T5 precision cache identity/hash mismatch")
    tensors = torch.load(entry["path"], map_location="cpu", weights_only=True)
    validate(tensors, contract)
    return tensors


def prepare(run, repo, config_text, precision):
    import os
    import sys
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path.insert(0, str(repo / ".local/vendor/Open-Sora"))
    import torch
    from mmengine import Config
    from opensora.models.text_encoder.t5 import T5Encoder
    cfg = Config.fromstring(config_text, file_format=".py")
    device = "cpu" if precision == "fp32" else "cuda"
    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    if device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("BF16 preparation requires a supporting CUDA GPU")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(cfg.seed)
    encoder = dict(cfg.text_encoder)
    if encoder.pop("type") != "t5" or set(encoder) != {"from_pretrained", "model_max_length"}:
        raise ValueError("unsupported T5 encoder configuration")
    model_path = Path(encoder["from_pretrained"])
    inventory = legacy.model_inventory(model_path)
    contract = dict(policy=POLICY, compute_dtype=precision, device=device,
        model_path=str(model_path.resolve()), model_inventory=inventory,
        model_sha256={name: sha256(model_path / name) for name in inventory},
        code={name: sha256(repo / name) for name in CODE},
        packages={n: importlib.metadata.version(n) for n in
                  ("torch", "transformers", "tokenizers", "ftfy", "beautifulsoup4")},
        max_length=encoder["model_max_length"], output_dim=read_json(model_path / "config.json")["d_model"],
        decoder_dtype="bf16", tf32=False, deterministic=True, batch_size=1)
    prompts = legacy.prepare_prompts(run / "receiver/metadata.csv", cfg, repo)
    report = populate(prompts, contract, repo / ".local/cache/t5_precision_embeddings",
        lambda: T5Encoder(**encoder, device=device, dtype=dtype, local_files_only=True))
    if legacy.model_inventory(model_path) != inventory:
        raise ValueError("T5 checkpoint changed during preparation")
    report["metadata_sha256"] = sha256(run / "receiver/metadata.csv")
    write_json(run / REPORT, report)


class Encoder:
    def __init__(self, report, trace):
        if (report.get("status") != "PASSED" or report.get("policy") != POLICY
                or fingerprint(report["contract"]) != report["signature"]):
            raise ValueError("invalid precision preparation report")
        contract = report["contract"]
        self.model_max_length, self.output_dim = contract["max_length"], contract["output_dim"]
        self.y_embedder, self.trace, self.calls = None, trace, []
        self.precision = contract["compute_dtype"]
        self.entries = {e["prompt"]: load(e, contract, e["prompt"]) for e in report["entries"]}
        if set(self.entries) != set(report["prompts"]) or len(report["prompts"]) != report["prompt_count"]:
            raise ValueError("missing prepared prompts")

    def encode(self, texts):
        import torch
        if not texts or any(t not in self.entries for t in texts):
            raise ValueError("unprepared caption")
        self.calls.extend(texts)
        write_json(self.trace, dict(policy=POLICY, compute_dtype=self.precision, calls=len(self.calls),
            prompt_sha256=[fingerprint(p) for p in self.calls], t5_model_loaded_in_decoder=False))
        # Existing decoder transfers these CPU tensors to its BF16 device.
        return {name: torch.cat([self.entries[p][name] for p in texts]) for name in ("y", "mask")}

    def null(self, n):
        return self.y_embedder.y_embedding[None].repeat(n, 1, 1)[:, None]


def install(report_path, trace, repo):
    from opensora import registry
    report = read_json(report_path)
    contract = report["contract"]
    if (legacy.model_inventory(contract["model_path"]) != contract["model_inventory"]
            or any(sha256(repo / name) != digest for name, digest in contract["code"].items())):
        raise ValueError("T5 model/code changed after preparation")
    original, used = registry.build_module, False

    def build(config, reg, **kwargs):
        nonlocal used
        if isinstance(config, dict) and config.get("type") == "t5":
            if used or dict(config) != dict(type="t5", from_pretrained=contract["model_path"],
                                             model_max_length=contract["max_length"]):
                raise ValueError("prepared T5 configuration mismatch")
            if kwargs.get("device") != "cpu":
                raise ValueError("cached T5 requires the CPU tensor transfer wrapper")
            used = True
            return Encoder(report, trace)
        return original(config, reg, **kwargs)
    registry.build_module = build


def validate_usage(run):
    report, trace = read_json(run / REPORT), read_json(run / TRACE)
    if (report["metadata_sha256"] != sha256(run / "receiver/metadata.csv")
            or trace["policy"] != POLICY or trace["compute_dtype"] != report["contract"]["compute_dtype"]
            or trace["t5_model_loaded_in_decoder"] or trace["calls"] != report["prompt_count"]
            or trace["prompt_sha256"] != [fingerprint(p) for p in report["prompts"]]):
        raise ValueError("prepared T5 conditions were not used in order")
