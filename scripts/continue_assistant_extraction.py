#!/usr/bin/env python3
"""Package actual assistant observations; never generate captions or mark unseen pages reviewed.

Use prepare after reviewing specified overview pages. Use freeze only after
reviewing every compact caption page and writing interval-specific source text.
The original three bundles and the independent automatic GPU run are untouched.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.webvid5 import read_json, fingerprint
from semantic_transmission import hybrid_selection as hybrid
from semantic_transmission.assisted_captions import expected_samples, validate_bundle

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/"outputs/fc_lgvsc_webvid_tvsum_full_20261001"


def directory(video):
    matches=list(ROOT.glob(f"*/{video}"))
    if len(matches)!=1:raise ValueError("unknown/ambiguous video")
    return matches[0]


def prepare(video,notes_path):
    d=directory(video);source=read_json(d/"source_prepared.json");notes=read_json(notes_path)
    dest=d/"extraction"
    if (dest/"selection_freeze.json").exists():raise ValueError("preserve existing selection")
    required_pages=list(source["evidence"])
    if sorted(notes["reviewed_overview_pages"])!=sorted(required_pages):
        raise ValueError("all full-source overview pages must actually be reviewed")
    for page,h in source["evidence"].items():
        if sha256(d/page)!=h:raise ValueError("overview evidence changed")
    candidates=notes["candidates"]
    if any(not r["mandatory"] for r in candidates):
        raise ValueError("optional proposals need genuine SKEM scoring; this CPU helper does not bypass it")
    authored=dict(normalized_sha256=source["normalized_sha256"],
        author="Codex assistant after directly viewing the recorded source overview images in this conversation",
        sampling="Full-source overview at every fourth 24-fps frame plus final frame; sampled review, not every-frame inspection",
        inspected_source_indices=source["sampled_indices"],candidates=candidates,
        reviewed_evidence=source["evidence"],notes=notes["notes"])
    write_json(d/"authored_observations.json",authored)
    spec=importlib.util.spec_from_file_location("existing_assistant_preparer",REPO/"scripts/prepare_fc_dataset_review.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # No model executes for an all-mandatory proposal set. Preserve the existing
    # model manifest after checking every recorded file's size and mtime.
    template=read_json(ROOT/"webvid/webvid_011_65f0e6cf/extraction/protocol.json")
    manifest=template["model_files"]
    for p,info in manifest.items():
        st=Path(p).stat()
        if st.st_size!=info["size"] or st.st_mtime_ns!=info["mtime_ns"]:
            raise ValueError("existing model manifest requires a fresh hash audit")
    original=hybrid.model_inventory
    try:
        hybrid.model_inventory=lambda _:manifest
        protocol=module.prepare(d,d/"authored_observations.json")
    finally:hybrid.model_inventory=original
    def forbidden(*_):raise ValueError("unscored optional candidate")
    keys,records=hybrid.choose_keys(source["frames"],[r["frame"] for r in candidates],
        [r["frame"] for r in candidates if r["mandatory"]],forbidden,max_gap=24,threshold=.35)
    selection=hybrid.export_selection(dest,protocol,keys,records,0,0)
    hybrid.verify_selection(dest)
    folder=dest/"assistant_captions";compact=folder/"compact"
    compact.mkdir(parents=True,exist_ok=True)
    samples=expected_samples(protocol,selection)
    metadata=dict(selection_freeze_sha256=sha256(dest/"selection_freeze.json"),
        input_sha256=protocol["input_sha256"],selection_root=str(dest),samples=samples,
        source_frame_hashes={str(i):protocol["source_frame_hashes"][str(i)] for r in samples for i in r["source_indices"]},
        sampling="PLLaVA four samples, half-open intervals; actual source PNGs",
        display="Each four-column group is one segment; left group then right group, top to bottom")
    write_json(folder/"samples.json",metadata)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    pages=[]
    for page,start in enumerate(range(0,len(samples),18)):
        group=samples[start:start+18]
        canvas=Image.new("RGB",(1536,((len(group)+1)//2)*151+24),"#171717")
        draw=ImageDraw.Draw(canvas);draw.text((4,3),video+" | DIRECT CAPTION REVIEW | groups of 4 frames",font=font,fill="white")
        for j,sample in enumerate(group):
            x=(j%2)*768;y=(j//2)*151+24
            draw.text((x+3,y+1),f"SEG {sample['segment']:03d} | [{sample['start']},{sample['end_exclusive']})",font=font,fill="white")
            for k,i in enumerate(sample["source_indices"]):
                with Image.open(Path(source["source_frames"])/f"{i}.png") as im:
                    canvas.paste(im.convert("RGB").resize((192,107)),(x+k*192,y+41))
                draw.text((x+k*192+3,y+21),f"f{i}",font=font,fill="white")
        path=compact/f"{page:03d}.jpg";canvas.save(path,quality=96);pages.append(str(path))
    print(json.dumps(dict(video=video,keyframes=len(keys),captions=len(samples),caption_pages=pages)),flush=True)


def freeze(video,authored_path,reviewed_pages):
    d=directory(video);dest=d/"extraction";folder=dest/"assistant_captions"
    if (folder/"captions_bundle.json").exists():raise ValueError("preserve existing caption bundle")
    pages=sorted(p.name for p in (folder/"compact").glob("*.jpg"))
    if sorted(reviewed_pages)!=pages:raise ValueError("must directly inspect every compact caption page before freezing")
    protocol,selection=hybrid.verify_selection(dest);expected=expected_samples(protocol,selection)
    authored=read_json(authored_path)
    if set(authored)!={str(r["segment"]) for r in expected}:raise ValueError("every segment requires a reviewed caption")
    records=[dict(row,text=authored[str(row["segment"])]) for row in expected]
    evidence={"samples.json":sha256(folder/"samples.json")}
    evidence.update({"compact/"+name:sha256(folder/"compact"/name) for name in pages})
    payload=dict(version=1,status="CAPTIONS_PREPARED_RECONSTRUCTION_NOT_STARTED",
        input_sha256=protocol["input_sha256"],selection_freeze_sha256=sha256(dest/"selection_freeze.json"),
        source_frame_hashes=read_json(folder/"samples.json")["source_frame_hashes"],records=records,
        author="Codex assistant in this conversation after direct source image review; no local model caption draft used",
        protocol="Visible content in the actual four recorded source samples; no invented identity or unseen detail",
        scope="Offline assistant-authored source captions. Not independent ground truth or an automatic model-only benchmark.",
        sampling="Same four source indices as PLLaVA; compact display with explicit segment/frame IDs",
        hallucination_mitigation_verified=False,reconstruction_started=False,evidence=evidence,
        reviewed_caption_pages=["compact/"+name for name in pages],
        packaging_code_sha256=sha256(Path(__file__)))
    write_json(folder/"authored.json",authored)
    write_json(folder/"captions_bundle.json",dict(payload,checksum=fingerprint(payload)))
    validate_bundle(folder/"captions_bundle.json",dest)
    print(json.dumps(dict(video=video,status="DIRECT_ASSISTANT_EXTRACTION_COMPLETE",keyframes=len(selection["indices"]),captions=len(records))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["prepare","freeze"]);p.add_argument("--video",required=True)
    p.add_argument("--authored",type=Path,required=True);p.add_argument("--reviewed-page",action="append",default=[])
    a=p.parse_args()
    if a.action=="prepare":prepare(a.video,a.authored)
    else:freeze(a.video,a.authored,a.reviewed_page)


if __name__=="__main__":main()
