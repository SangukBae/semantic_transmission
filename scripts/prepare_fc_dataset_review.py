#!/usr/bin/env python3
"""Bind genuinely reviewed source proposals to the existing FC-LGVSC selector.

Supply observations authored after viewing the source sheets. This command does
not label unreviewed frames or substitute dataset summaries for source evidence.
"""
import argparse
import ast
from datetime import datetime, timezone
from pathlib import Path

from semantic_transmission import hybrid_selection as hybrid
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.webvid5 import fingerprint, read_json


def prepare(directory, authored):
    source=read_json(directory / "source_prepared.json")
    observations=read_json(authored)
    if observations["normalized_sha256"] != source["normalized_sha256"]:
        raise ValueError("review was authored for a different normalized source")
    inspected=observations["inspected_source_indices"]
    if not inspected or any(type(i) is not int or not 0<=i<source["frames"] for i in inspected):
        raise ValueError("invalid inspected source frames")
    candidates=observations["candidates"]
    indices=[r["frame"] for r in candidates]
    if indices != sorted(set(indices)) or indices[0]!=0 or indices[-1]!=source["frames"]-1:
        raise ValueError("invalid source proposal timeline")
    if any(i not in inspected for i in indices):
        raise ValueError("proposed frames must be visually inspected")
    for row in candidates:
        if type(row["mandatory"]) is not bool or not row["reason_ko"]:
            raise ValueError("each proposal needs a mandatory flag and observed reason")
    cfg=read_json(hybrid.BASE / "run_config.json")
    for name in ("selector_checkpoint","caption_checkpoint","hybrid_selection_root","caption_bundle"):
        cfg.pop(name,None)
    cfg.update(input=source["normalized_video"],input_sha256=source["normalized_sha256"],
        frames=source["frames"],max_frames=source["frames"],fps=24,width=576,height=320,
        preserve_input=True,semantic_clip_policy="frame_exact",concatenation_policy="endpoint_exact")
    tree=ast.parse(hybrid.SKEM.read_text())
    prompts={n.targets[0].id:ast.literal_eval(n.value) for n in ast.walk(tree)
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
        and n.targets[0].id in ("prompt_ask_image","prompt_compare_image")}
    files=[Path(hybrid.__file__),hybrid.SKEM,hybrid.REPO / "src/semantic_transmission/internvl_memory.py",
           hybrid.REPO / "src/semantic_transmission/exact_reuse.py",Path(__file__)]
    destination=directory / "extraction"
    destination.mkdir(exist_ok=True)
    if (destination / "protocol.json").exists():
        return hybrid.validate_protocol(destination)
    proposal=destination / "assistant_observations.json"
    write_json(proposal,observations)
    protocol=dict(version=1,created_utc=datetime.now(timezone.utc).isoformat(),
        source_id=source["id"],source_split="local_full_source_extraction",baseline=str(directory),
        input=source["normalized_video"],input_sha256=source["normalized_sha256"],
        normalized_video=source["normalized_video"],normalized_sha256=source["normalized_sha256"],
        raw_source=source["source_path"],raw_source_sha256=source["source_sha256"],
        frames=source["frames"],fps=24,width=576,height=320,
        source_frames=source["source_frames"],source_frame_hashes=source["source_frame_hashes"],
        proposal_path=str(proposal),proposal_sha256=sha256(proposal),candidates=candidates,
        max_gap_frames=24,threshold=.35,config=cfg,prompts=prompts,
        model_files=hybrid.model_inventory(cfg["models"]["internvl"]),
        code={str(p.relative_to(hybrid.REPO)):sha256(p) for p in files},
        scope="Offline assistant source review; sampled event observations, not independent ground truth.",
        sampling=observations["sampling"],inspected_source_indices=inspected,
        benchmark_memberships=source["benchmark_memberships"],
        reconstruction_status="NOT_STARTED",hallucination_mitigation_verified=False)
    protocol["signature"]=fingerprint(protocol)
    write_json(destination / "protocol.json",protocol)
    return hybrid.validate_protocol(destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--observations",type=Path,required=True)
    parser.add_argument("--select",action="store_true")
    args=parser.parse_args()
    directory=args.source_root.resolve()
    prepare(directory,args.observations.resolve())
    if args.select:
        hybrid.select(directory / "extraction")
        from semantic_transmission.assisted_captions import prepare as prepare_captions
        prepare_captions(directory / "extraction")


if __name__=="__main__":
    main()
