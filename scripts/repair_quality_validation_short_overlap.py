#!/usr/bin/env python3
"""Rebind verified completed work after the one short-overlap decoder fix.

This is deliberately not a generic --force-resume: only the exact decoder edit
below is accepted, and every reused decoder job must be outside its new branch.
Original protocols and receipts are retained in an immutable repair snapshot.
"""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path

from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.cli import repository
from semantic_transmission.quality_validation import resolve_config, build_plan, inventory
from semantic_transmission.temporal import segment_lengths
from semantic_transmission.webvid5 import execution_identity, fingerprint, read_json

DECODER = "04_semantic_decoder/scripts/mydemo_new_align_sh.py"
BEFORE = '''                        if (conditioning_alignment == "endpoint_exact"
                                or cfg.get("decoder_policy") != "official_release"):
                            overlap = conditioning_indices(previous.shape[2], dframe_to_frame(condition_frame_length))'''
AFTER = '''                        if (conditioning_alignment == "endpoint_exact"
                                or cfg.get("decoder_policy") != "official_release"
                                or previous.shape[2] < dframe_to_frame(condition_frame_length)):
                            # A short first segment cannot supply five reference latents.
                            # Left-pad only that case; retain the released path for long clips.
                            overlap = conditioning_indices(previous.shape[2], dframe_to_frame(condition_frame_length))'''


def validate_patch(old, current, decoder_text):
    if decoder_text.count(AFTER) != 1:
        raise ValueError("expected the exact short-overlap decoder fix")
    original = decoder_text.replace(AFTER, BEFORE)
    if hashlib.sha256(original.encode()).hexdigest() != old["execution"]["code_sha256"][DECODER]:
        raise ValueError("decoder changes extend beyond the audited short-overlap fix")
    expected = copy.deepcopy(old)
    expected["execution"]["code_sha256"][DECODER] = hashlib.sha256(decoder_text.encode()).hexdigest()
    if expected != current:
        raise ValueError("other code, input, environment, model, or setting changes prevent reuse")
    return original


def rebind_prefix(plan, receipts, old_signature, new_signature, check_artifacts=inventory):
    old_chain, new_chain = fingerprint(old_signature), fingerprint(new_signature)
    updated, evidence = {}, []
    stopped = False
    for job in plan:
        receipt = receipts.get(job["name"])
        if receipt is None or receipt["status"] != "PASSED":
            stopped = True
            continue
        if stopped:
            raise ValueError("completed work is not a contiguous dependency prefix")
        if receipt["signature"] != fingerprint({"job": job, "previous": old_chain}):
            raise ValueError(f"original receipt dependency mismatch: {job['name']}")
        if check_artifacts(job["required"]) != receipt["artifacts"]:
            raise ValueError(f"completed artifact changed: {job['name']}")
        if job["spec"].get("stage") == "reconstruct":
            inputs = read_json(Path(job["spec"]["run"])/"receiver/decoder_inputs.json")
            # This repair is for the pinned 5-latent / 17-frame overlap profile.
            config = (Path(job["spec"]["run"])/"receiver/decoder_config.py").read_text()
            if "condition_frame_length = 5\n" not in config:
                raise ValueError("unaudited overlap configuration")
            lengths = segment_lengths(inputs["indices"], 17)
            previous_min = min(lengths[:-1], default=17)
            if previous_min < 17:
                raise ValueError(f"completed decoder job would enter the changed branch: {job['name']}")
            evidence.append({"job": job["name"], "previous_clip_min_frames": previous_min,
                             "changed_branch_executed": False})
        replacement = dict(receipt,
            signature=fingerprint({"job": job, "previous": new_chain}),
            produced_with_execution_signature=old_signature,
            rebound_for_execution_signature=new_signature,
            reuse_reason="verified unchanged outputs; short-overlap branch not used")
        updated[job["name"]] = replacement
        old_chain, new_chain = fingerprint(receipt), fingerprint(replacement)
    return updated, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--apply", action="store_true", help="apply the verified migration; default: inspect only")
    args = parser.parse_args()
    repo, root = repository(), args.run.resolve(strict=True)
    config_path = (args.config or repo/"configs/quality_validation.json").resolve()
    with (repo/".local/quality_validation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        repair = root/"repairs/short_overlap_v1"
        backup_path = repair/"original_state.json"
        # A retry after an interrupted migration always starts from the originals.
        if backup_path.exists():
            backup = read_json(backup_path)
        else:
            backup = {"protocol": read_json(root/"protocol.json"), "plan": read_json(root/"plan.json"),
                "receipts": {p.parent.name: read_json(p) for p in sorted((root/"tasks").glob("*/receipt.json"))}}
        old_protocol = backup["protocol"]
        old = {k: v for k,v in old_protocol.items() if k not in {"signature", "selection_frozen_before_results"}}
        if fingerprint(old) != old_protocol["signature"]:
            raise ValueError("original protocol signature mismatch")
        cfg, evidence = resolve_config(repo, config_path)
        current = dict(old, config=cfg, input_sha256=evidence,
            execution=execution_identity(repo, {"models": read_json(repo/".local/model_paths.json")}),
            profile_sha256={name:sha256(repo/"configs"/name) for name in old["profile_sha256"]},
            config_sha256=sha256(config_path))
        original_decoder = validate_patch(old, current, (repo/DECODER).read_text())
        new_signature = fingerprint(current)
        plan = build_plan(root, cfg, repo)
        if plan != backup["plan"]:
            raise ValueError("task plan changed")
        completed_marker = repair/"migration.json"
        if completed_marker.exists():
            if (read_json(completed_marker)["new_signature"] == new_signature
                    and read_json(root/"protocol.json")["signature"] == new_signature):
                print(f"이미 복구됐습니다: {root}")
                return
            raise ValueError("existing migration does not match this runtime")
        updated, decoder_evidence = rebind_prefix(plan, backup["receipts"], old_protocol["signature"], new_signature)
        alias = repo/"outputs"/f"quality_validation_{new_signature[:12]}"
        if (alias.exists() or alias.is_symlink()) and alias.resolve() != root:
            raise ValueError(f"output alias already belongs to another run: {alias}")
        result = {"status":"VERIFIED", "old_signature":old_protocol["signature"], "new_signature":new_signature,
                  "reused_completed_tasks":len(updated), "decoder_reuse_evidence":decoder_evidence,
                  "output_alias":str(alias), "original_output":str(root),
                  "reason":"left-pad generated clips shorter than 17 frames before overlap encoding"}
        if args.apply:
            repair.mkdir(parents=True,exist_ok=True)
            if not backup_path.exists():
                write_json(backup_path, backup)
                (repair/"decoder_before.py.txt").write_text(original_decoder)
                (repair/"decoder_after.py.txt").write_text((repo/DECODER).read_text())
                for name, receipt in backup["receipts"].items():
                    log = root/"tasks"/name/"worker.log"
                    if receipt["status"] != "PASSED" and log.exists():
                        (repair/f"{name}.worker.log").write_bytes(log.read_bytes())
            for name, receipt in updated.items():
                write_json(root/"tasks"/name/"receipt.json", receipt)
            write_json(root/"protocol.json", dict(current, signature=new_signature, selection_frozen_before_results=True))
            if not alias.exists():
                alias.symlink_to(root, target_is_directory=True)
            result["status"] = "MIGRATED"
            write_json(completed_marker,result)
        print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
