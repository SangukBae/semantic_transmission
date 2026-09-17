#!/usr/bin/env python3
"""Replay the exact saved cases to add established temporal baselines."""
import argparse
import hashlib
from itertools import chain
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from semantic_transmission.artifacts import sha256
from semantic_transmission.automatic_validation import write, summarize, METRICS
from semantic_transmission.controlled_errors import controls, temporal_errors, object_errors, scene
from semantic_transmission.pair_inputs import sampled_video
from semantic_transmission.temporal_baselines import TemporalBaselines


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    root = args.run.resolve()
    output = root / "additional_baselines"
    output.mkdir(exist_ok=False)
    protocol = json.loads((root / "protocol.json").read_text())
    original_rows = [json.loads(line) for line in (root / "cases.jsonl").read_text().splitlines()]
    lookup = {(r["source_id"], r["case_id"]): r for r in original_rows}
    write(output / "protocol.json", {"parent_protocol_sha256": sha256(root / "protocol.json"), "parent_cases_sha256": sha256(root / "cases.jsonl"),
          "implementation_sha256": sha256(Path(__file__).parents[1] / "src/semantic_transmission/temporal_baselines.py"),
          "purpose": "additional baselines after first candidate results; no candidate/threshold retuning",
          "device": args.device, "reference": "https://github.com/thunil/TecoGAN/blob/master/metrics.py",
          "adaptation": "LPIPS-Alex v0.1; no spatial border crop; all sampled transitions; no x100 scaling"})
    evaluator = TemporalBaselines(args.device)
    rows = []
    with (output / "cases.jsonl").open("w") as stream:
        def evaluate_source(sid, source, variants):
            evaluator.cache.clear()
            source_hash = hashlib.sha256(source.tobytes()).hexdigest()
            for case, rec, _ in variants:
                row = dict(lookup[(sid, case)])
                if source_hash != row["source_pixel_sha256"] or hashlib.sha256(rec.tobytes()).hexdigest() != row["reconstruction_pixel_sha256"]:
                    raise ValueError(f"replayed pixels changed: {sid}/{case}")
                row.update(evaluator.evaluate(source, rec))
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(sid, len(rows), flush=True)
        for item in protocol["inventory"]:
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError("source bytes changed")
            source, _, _ = sampled_video(item["path"], protocol["frames_sampled"])
            evaluate_source(item["source_id"], source, chain(controls(source), temporal_errors(source)))
        for seeds in protocol["synthetic_seeds"].values():
            for seed in seeds:
                source, _ = scene(seed, protocol["frames_sampled"])
                evaluate_source(f"renderer/{seed}", source, chain(controls(source), temporal_errors(source), object_errors(seed, protocol["frames_sampled"])))
    if len(rows) != len(original_rows):
        raise ValueError("replay did not cover every original case")
    METRICS.update(tlp_alex=1, tof_farneback=1, psnr_global_db=-1)
    write(output / "summary.json", {"status": "COMPLETED", "cases": len(rows), "statistics": summarize(rows),
                                    "candidate_decisions": "unchanged; both NOT_PASSED in original run"})


if __name__ == "__main__":
    main()
