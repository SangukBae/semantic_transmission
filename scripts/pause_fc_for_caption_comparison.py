"""Interrupt only the known resumable FC SKEM worker; retain score checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "outputs/fc_lgvsc_webvid_tvsum_faithful_20261002"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    proc = Path(f"/proc/{args.pid}")
    command = (proc / "cmdline").read_bytes().split(b"\0")
    expected_python = b"/home/sangukbae/anaconda3/envs/lgvsc-internvl/bin/python"
    if command[:3] != [expected_python, b"scripts/correct_fc_dataset_extraction.py", b"run"]:
        raise RuntimeError("PID identity differs from the authorized SKEM worker")
    if (proc / "cwd").resolve() != REPO:
        raise RuntimeError("Worker is in an unexpected directory")
    target = ROOT / "paused_for_qwen_caption_comparison.json"
    if target.exists():
        raise FileExistsError(target)
    progress = json.loads((ROOT / "progress.json").read_text())
    record = dict(status="INTERRUPT_REQUESTED", created_utc=datetime.now(timezone.utc).isoformat(),
                  pid=args.pid, command=[x.decode() for x in command if x], progress=progress,
                  user_requested=True, signal="SIGINT", completed_scores_preserved=True,
                  resume_command="/home/sangukbae/anaconda3/envs/lgvsc-internvl/bin/python scripts/correct_fc_dataset_extraction.py run",
                  note="Atomic per-pair checkpoints are reused on restart; an in-flight pair may need recomputation.")
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    os.kill(args.pid, signal.SIGINT)
    for _ in range(50):
        if not proc.exists() or "State:\tZ" in (proc / "status").read_text():
            record["status"] = "STOPPED_RESUMABLE"
            break
        time.sleep(1)
    else:
        record["status"] = "SIGINT_SENT_PROCESS_STILL_PRESENT"
    record["checked_utc"] = datetime.now(timezone.utc).isoformat()
    record["completed_video_selections"] = len(list(ROOT.glob("*/*/extraction/selection_freeze.json")))
    record["saved_pair_scores"] = len(list(ROOT.glob("*/*/extraction/scores/*.json")))
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0 if record["status"] == "STOPPED_RESUMABLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
