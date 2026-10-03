#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
core=$(python3 -c 'import json; print(json.load(open(".local/settings.json"))["python"])')
exec "$core" - "$@" <<'PY'
import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path
from semantic_transmission.cli import settings
from semantic_transmission.etri_60s_check import lock, run_command
from semantic_transmission.webvid_ablation import environment
from semantic_transmission.artifacts import write_json
from semantic_transmission.webvid5 import read_json

repo = Path.cwd()
root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else repo / "outputs/skem_speed_20260929"
python = settings(repo)["internvl_python"]
core = settings(repo)["python"]
root.mkdir(parents=True, exist_ok=True)
source = repo / "scripts/benchmark_skem_speed.py"
module_spec = importlib.util.spec_from_file_location("speed_benchmark", source)
benchmark = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(benchmark)
failed = []
with lock(repo / ".local/etri_60s_check.lock"), lock(root / ".lock"):
    if not (root / "protocol.json").exists():
        benchmark.prepare(root, repo / "outputs/etri_60s_tv_low_08_42057b2ee8ed")
    protocol = benchmark.validate(root)
    if not (root / "candidate_coverage.json").exists():
        subprocess.run([core, str(repo / "scripts/check_skem_candidate_coverage.py"),
                        "--output", str(root)], check=True)
    for mode in ("int8", "baseline", "brief", "sparse"):
        if mode == "baseline" and (root / "baseline_reference.json").exists():
            sys.path.insert(0, str(repo / "scripts"))
            from validate_skem_speed import baseline_reference
            baseline_reference(root)
            print("REUSE audited historical baseline; its timing is historical context only", flush=True)
            continue
        log = root / f"logs/select_{mode}.log"
        metric = root / f"resources/select_{mode}.json"
        completed = metric.exists() and read_json(metric).get("returncode") == 0
        for window in protocol["windows"]:
            record = root / f"selection/{mode}/{window}.json"
            completed = completed and record.exists() and read_json(record).get("status") == "COMPLETE"
            if record.exists() and read_json(record).get("protocol_signature") != protocol["signature"]:
                raise ValueError("saved selection belongs to a different protocol")
        if completed:
            print(f"REUSE completed selection: {mode}", flush=True)
            continue
        if log.exists():
            archive = root / "failed_attempts" / f"select_{mode}_{time.time_ns()}"
            for path in (log, metric, metric.with_suffix(".time.txt"), root / f"selection/{mode}/runtime.json",
                         root / f"selection/{mode}/failure.json"):
                if path.exists():
                    dest = archive / path.relative_to(root)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), str(dest))
        try:
            run_command(repo, [python, str(repo / "scripts/benchmark_skem_speed.py"), "select",
                "--output", str(root), "--mode", mode], log, environment(2025),
                root / f"resources/select_{mode}.json", root / f"progress/select_{mode}.json")
        except Exception as error:
            write_json(root / f"selection/{mode}/failure.json", {"status": "FAILED", "error": str(error)})
            print(f"FAILED {mode}: {error}", flush=True)
            failed.append(mode)
if failed:
    raise SystemExit(f"Selection failed: {failed}; inspect saved logs before retrying")
subprocess.run([core, str(repo / "scripts/validate_skem_speed.py"), "--output", str(root)], check=True)
subprocess.run([core, str(repo / "scripts/report_skem_speed.py"), "--output", str(root)], check=True)
PY
