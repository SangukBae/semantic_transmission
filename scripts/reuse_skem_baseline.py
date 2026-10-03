"""Import an audited historical selector baseline without calling it a new run."""
import argparse
import ast
import datetime
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.webvid5 import read_json


def parse_log(path, threshold):
    text = path.read_text()
    stamp = r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)"
    starts = list(re.finditer(r"(?m)^" + stamp + r" - INFO \| frame_index: (\d+)$", text))
    clock = lambda value: datetime.datetime.strptime(value, "%Y-%m-%d %H:%M:%S,%f")
    selected, records = [0], {}
    for i, match in enumerate(starts):
        number = int(match[2])
        block = text[match.end():starts[i+1].start() if i+1 < len(starts) else len(text)]
        first = re.search(r"(?m)^" + stamp + r" - INFO \| response1: ", block)
        second = re.search(r"(?m)^" + stamp + r" - INFO \| response2: ([^\n]+)", block)
        end = re.search(r"(?m)^" + stamp + r" - INFO \| frame_numbers: (\[[^\n]+\])", block)
        no = float(re.search(r"no_prob: ([^\n]+)", block)[1])
        yes = float(re.search(r"yes_prob: ([^\n]+)", block)[1])
        reference = selected[-1]
        choose = no - yes > threshold
        if choose:
            selected.append(number)
        assert list(map(int, ast.literal_eval(end[2]))) == selected
        finish = clock(starts[i+1][1]) if i+1 < len(starts) else clock(end[1])
        records[number] = {"candidate": number, "reference": reference, "selected": choose,
            "p_yes": yes, "p_no": no, "psss": no-yes,
            "description": block[first.end():second.start()].strip(), "answer": second[2],
            "round1_seconds": (clock(first[1])-clock(match[1])).total_seconds(),
            "round2_seconds": (clock(second[1])-clock(first[1])).total_seconds(),
            "total_seconds": (finish-clock(match[1])).total_seconds(),
            "timing_source": "historical log wall-clock timestamps"}
    return records, sorted(set(selected + [max(records)]))


def main():
    import cv2
    import numpy as np
    from transformers import AutoTokenizer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=REPO / "outputs/skem_speed_20260929")
    args = p.parse_args()
    root = args.output.resolve()
    if (root / "baseline_reference.json").exists():
        raise ValueError("baseline reference already exists")
    protocol = read_json(root / "protocol.json")
    source = REPO / "outputs/etri_60s_tv_low_08_263507d45874"
    base = source / "baseline"
    old_protocol = read_json(source / "protocol.json")
    for relative in ("02_semantic_encoder/skem/MLM-keyframe-internvl.py",
                     "src/semantic_transmission/internvl_memory.py", "src/semantic_transmission/exact_reuse.py"):
        assert sha256(REPO / relative) == old_protocol["execution"]["code_sha256"][relative]
    cfg = read_json(base / "run_config.json")
    log = next(base.glob("internvl_diff_0.35_*.log"))
    records, chosen = parse_log(log, cfg["threshold"])
    assert chosen == read_json(base / "keyframes.json")["indices"]
    assert chosen == read_json(Path(protocol["baseline"]) / "baseline/keyframes.json")["indices"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["internvl"], trust_remote_code=True, use_fast=True)
    policy = {"status": "HISTORICAL_SELECTION_REUSED", "selection_protocol": protocol["signature"],
        "reason": "Reuse audited selector records; historical timing is not a new speed measurement.",
        "timing_scope": "Historical timing is context only; no confirmed speedup ratio against current runs.",
        "source_files": {str(path): sha256(path) for path in (log, source / "protocol.json", base / "keyframes.json", base / "run_config.json")},
        "records": {}, "fresh_overlap": {}}
    for window, spec in protocol["windows"].items():
        start, end = spec["start"], spec["end"]
        assert start in chosen
        rows = []
        for local, global_index in enumerate(range(start, end+1)):
            original = base / f"data/frames/sample/{global_index}.png"
            current = root / f"inputs/{window}/frames/{local}.png"
            assert np.array_equal(cv2.imread(str(original)), cv2.imread(str(current)))
            policy["source_files"][str(original)] = sha256(original)
            if not local:
                continue
            record = dict(records[global_index])
            record.update(candidate=local, reference=record["reference"]-start,
                description_tokens_reencoded=len(tokenizer.encode(record["description"], add_special_tokens=False)))
            assert record["reference"] >= 0
            rows.append(record)
        indices = sorted({0, end-start} | {r["candidate"] for r in rows if r["selected"]})
        assert indices == spec["original_keys_relative"]
        fresh = root / f"selection/baseline/{window}.json"
        if fresh.exists():
            measured = read_json(fresh)["records"]
            assert all((r["candidate"], r["reference"], r["selected"]) ==
                (s["candidate"], s["reference"], s["selected"]) for r,s in zip(measured,rows))
            policy["fresh_overlap"][window] = {"comparisons": len(measured),
                "choices_identical": True,
                "descriptions_identical": sum(r["description"] == s["description"] for r,s in zip(measured,rows))}
        path = root / f"selection/historical_baseline/{window}.json"
        write_json(path, {"status": "COMPLETE", "mode": "baseline", "window": window,
            "source_kind": "VERIFIED_HISTORICAL_LOG", "protocol_signature": protocol["signature"],
            "indices": indices, "records": rows, "candidates": list(range(spec["frames"])),
            "selection_seconds": sum(r["total_seconds"] for r in rows),
            "timing_source": "historical log; not a new measurement"})
        policy["records"][window] = {"path": str(path.relative_to(root)), "sha256": sha256(path)}
    write_json(root / "baseline_reference.json", policy)
    print(json.dumps({"status": policy["status"], "fresh_overlap": policy["fresh_overlap"]}, indent=2))


if __name__ == "__main__":
    main()
