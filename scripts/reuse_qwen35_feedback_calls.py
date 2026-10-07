"""Reuse identical generation requests after a parser-only correction.

Raw response bytes, runtime and parent hashes stay explicit. Never reuses output
after a model, source, prompt, sampling, or generation-budget change.
"""
import argparse
from pathlib import Path

from semantic_transmission import auto_extraction as ae, qwen35_feedback as fb
from semantic_transmission.artifacts import sha256, write_json


def reuse(old, new):
    before = ae.read(old / "plan.json")
    after = fb.check_plan(new)
    excluded = {"signature", "created_utc", "code_sha256", "prompts"}
    if {k:v for k,v in before.items() if k not in excluded} != {k:v for k,v in after.items() if k not in excluded}:
        raise ValueError("generation policy or source changed; cache reuse refused")
    changed_prompts = {k for k in before["prompts"] if before["prompts"][k] != after["prompts"][k]}
    dependencies = dict(observe={"observe"}, select={"select"}, caption={"caption"},
                        caption_review={"caption", "review"}, feedback={"feedback"})
    rows = []
    for path in sorted((old / "calls").glob("*/*.json")):
        record = ae.unseal(path)
        if record["identity"]["plan_signature"] != before["signature"]:
            raise ValueError("invalid parent inference provenance")
        if dependencies[record["identity"]["role"]] & changed_prompts:
            continue  # The actual request changed: regenerate this role.
        fb.Session(old, before).validate_record(record, lambda x:x)
        identity = dict(record["identity"], plan_signature=after["signature"])
        record.update(identity=identity, reused_from=dict(path=str(path.resolve()), sha256=sha256(path),
            plan_sha256=sha256(old / "plan.json"), reason="identical generation request; changed parser or another role's response schema"))
        out = new / "calls" / identity["role"] / f"{ae.digest(identity)}.json"
        if out.exists():
            raise FileExistsError(out)
        write_json(out, ae.seal(record))
        rows.append(dict(source=str(path.resolve()), source_sha256=sha256(path), target=str(out.resolve()), target_sha256=sha256(out)))
    write_json(new / "schema_cache_reuse.json", dict(source_plan_sha256=sha256(old / "plan.json"),
        target_plan_sha256=sha256(new / "plan.json"), policy="exact generation inputs preserved", records=rows))
    print(f"Reused {len(rows)} exact generation requests with parent provenance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    args = parser.parse_args()
    reuse(args.old.resolve(), args.new.resolve())
