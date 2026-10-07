"""Deliver completed anonymous caption bins while remaining GPU work continues.

Final evaluation validates these texts exactly against the completed output.
This script is separate from extraction and never feeds reference data to Qwen.
"""
import argparse
import json
from pathlib import Path
import random

from semantic_transmission import auto_extraction as ae, qwen35_feedback as fb
from semantic_transmission.artifacts import sha256, write_json


def prepare(root, lo, hi):
    plan = fb.check_plan(root)
    observations = ae.unseal(root / "ledger.json")["observations"]
    events = [dict(e, window=row["window"], origin="observe") for row in observations for e in row["result"]["events"]]
    candidates = {0, plan["frames"]-1, *range(0, plan["frames"], plan["optional_stride"])}
    mandatory = {0, plan["frames"]-1}
    for event in events:
        candidates.update((event["before"], event["after"]))
        if fb.protected(event):
            mandatory.update((event["before"], event["after"]))
    score_records = [ae.unseal(p) for p in (root / "calls/select").glob("*.json")]
    def score(a, b):
        prefix = plan["prompts"]["select"] + json.dumps(fb.notes_for(observations, a, b, events))
        matches = [r for r in score_records if r["identity"]["ids"] == [a,b] and r["identity"]["prompt"].startswith(prefix + "\nVALID")]
        if len(matches) != 1:
            raise ValueError("original sequential selection is not complete")
        return matches[0]["result"]
    keys, _ = fb.choose_keys(plan["frames"], candidates, mandatory, score, plan["max_gap"])
    mapping = dict(zip(random.Random(20261003).sample(["M17", "M43", "M82"], 3), ["historical", "forward", "feedback"]))
    code = next(k for k,v in mapping.items() if v == "forward")
    captions = []
    reviews = [ae.unseal(p) for p in (root / "calls/caption_review").glob("*.json")]
    for i, (a, b) in enumerate(zip(keys, keys[1:])):
        if not a < hi*24 or not lo*24 < b:
            continue
        matches = []
        for record in reviews:
            text = record["identity"]["prompt"]
            if not text.startswith(plan["prompts"]["caption"]):
                continue
            data, _ = json.JSONDecoder().raw_decode(text[len(plan["prompts"]["caption"]):])
            if data == dict(target=[a,b], shared=fb.notes_for(observations,a,b-1,events), source_review_issues=[]):
                matches.append(record)
        if len(matches) != 1:
            raise ValueError(f"caption [{a},{b}) is not complete")
        captions.append(dict(caption_id=f"{code}_{i:03d}",start=a,end_exclusive=b,caption=matches[0]["result"]["caption"]))
    reference = ae.read(root / "astra_reference/reference.json")
    protocol = ae.read(root / "evaluation/protocol.json")
    bins = [dict(reference=ref, candidates=[dict(method=code, captions=[c for c in captions
             if c["start"] < ref["end_exclusive"] and ref["start"] < c["end_exclusive"]])]) for ref in reference["bins"][lo:hi]]
    packet = dict(evaluator_model="gpt-6-astra",source_frames=plan["source_frames"],fps=24,
        frames=plan["frames"],reference_sha256=protocol["reference_sha256"],bins=bins,instructions=protocol)
    path = root / "evaluation/early_packets" / f"{lo:02d}_{hi-1:02d}_next.json"
    if path.exists() and ae.read(path) != packet:
        raise ValueError("previously delivered packet changed")
    write_json(path,packet)
    print(json.dumps(dict(path=str(path), sha256=sha256(path), bins=len(bins))))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=fb.DEFAULT_ROOT)
    p.add_argument("--lo",type=int,required=True)
    p.add_argument("--hi",type=int,required=True)
    args=p.parse_args()
    prepare(args.root.resolve(),args.lo,args.hi)
