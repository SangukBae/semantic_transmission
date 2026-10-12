#!/usr/bin/env python3
"""Make a new BSU runtime template without modifying historical protocols.

The reference supplies the adopted receiver policy and pinned Qwen file hashes.
Local settings supply interpreters and receiver model locations. This prepares
configuration only; it does not certify inference or paper reproduction.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import time

from semantic_transmission.artifacts import sha256
from semantic_transmission.cli import settings
from semantic_transmission.qwen35_backend import MODEL_ID, REVISION, SNAPSHOT
from semantic_transmission.qwen35_caption_adapter import default_adapter, verify_adapter

REPO = Path(__file__).resolve().parents[1]


def build(reference, snr_db=None):
    original = json.loads(reference.read_text())
    caption = copy.deepcopy(original["caption"])
    if (caption["model_id"], caption["model_revision"], caption["quantization"]) != (MODEL_ID, REVISION, "nf4"):
        raise ValueError("reference is not the pinned BSU Qwen model")
    code = original["code"]
    for relative, digest in code.items():
        if sha256(REPO / relative) != digest:
            raise ValueError(f"reference implementation differs: {relative}")
    local = settings(REPO)
    interpreters = dict(qwen=str(REPO / ".local/qwen35_env/bin/python"),
                        **{k: local[k] for k in ("python", "channel_python")})
    for path in interpreters.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    caption.update(model_path=str(SNAPSHOT), caption_adapter=default_adapter())
    verify_adapter(caption["caption_adapter"])
    receiver = copy.deepcopy(original["receiver"])
    receiver["models"] = json.loads((REPO / ".local/model_paths.json").read_text())
    for path in receiver["models"].values():
        if not Path(path).is_dir():
            raise FileNotFoundError(path)
    if snr_db is not None:
        if not math.isfinite(snr_db):
            raise ValueError("SNR must be finite")
        receiver["snr_db"] = snr_db
    for name in ("input", "input_sha256"):
        receiver.pop(name, None)  # Each new plan binds its own prepared source.
    base_files = {}
    names = [Path(p).name for p in original["base_model_files"]]
    if len(set(names)) != len(names):
        raise ValueError("ambiguous base model filenames")
    started = time.monotonic()
    for n, (old, item) in enumerate(original["base_model_files"].items(), 1):
        path = SNAPSHOT / Path(old).name
        before = path.stat()
        if before.st_size != item["size"] or sha256(path) != item["sha256"]:
            raise ValueError(f"base model bytes differ: {path}")
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"base model changed during verification: {path}")
        base_files[str(path)] = dict(size=after.st_size, mtime_ns=after.st_mtime_ns, sha256=item["sha256"])
        print(f"MODEL_HASH {n}/{len(names)} ({100*n/len(names):.1f}%) elapsed={time.monotonic()-started:.1f}s", flush=True)
    return dict(policy="bsu_deployment_template_v1", caption=caption, receiver=receiver,
                interpreters=interpreters, base_model_files=base_files, code=code,
                reference=dict(path=str(reference), sha256=sha256(reference)),
                builder_sha256=sha256(__file__), inference_verified=False, paper_reproduction=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snr-db", type=float)
    args = parser.parse_args()
    output, reference = args.output.resolve(), args.reference.resolve()
    if output == reference:
        raise ValueError("output must be separate from the historical reference")
    value = build(reference, args.snr_db)
    if output.exists():
        if json.loads(output.read_text()) != value:
            raise ValueError("deployment configuration changed; use a new output")
        print(f"VERIFIED_EXISTING_TEMPLATE {output}", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    print(f"TEMPLATE_READY {output}; inference is a separate check", flush=True)


if __name__ == "__main__":
    main()
