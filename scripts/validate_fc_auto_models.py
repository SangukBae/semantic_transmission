#!/usr/bin/env python3
"""Run small local visual probes; does not label model quality automatically."""
import argparse
import gc
import json
from pathlib import Path
import time

from PIL import Image
import torch

from semantic_transmission.auto_extraction import (
    CAPTION_PROMPT, REVIEW_PROMPT, PROPOSAL_PROMPT, DEFAULT_PREPARED, LocalVLM,
    check_frame, read, parse_json, validate_caption, validate_events, write_json,
)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",required=True)
    parser.add_argument("--revision",required=True)
    parser.add_argument("--int8",action="store_true")
    parser.add_argument("--nf4",action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    path=Path.home()/".cache/huggingface/hub"/("models--"+args.model.replace("/","--"))/"snapshots"/args.revision
    config=dict(model_path=str(path),quantization="nf4" if args.nf4 else "int8" if args.int8 else None)
    begin=time.monotonic()
    vlm=LocalVLM(config)
    results=[]
    for rel,ids in [("webvid/webvid_011_65f0e6cf",[6,30,60,90]),
                    ("webvid/webvid_008_852d56f2",[5,30,60,90]),
                    ("tvsum/tvsum_011_0e11bb5b",[1100,1110,1120,1130])]:
        source=read(DEFAULT_PREPARED/rel/"source_prepared.json")
        images=[Image.open(check_frame(source,i)).convert("RGB") for i in ids]
        labels=[f"Source frame {i}, time {i/24:.6f} seconds" for i in ids]
        draft=vlm.generate(images,labels,CAPTION_PROMPT,384)
        parsed=validate_caption(parse_json(draft["raw_text"]))
        review=vlm.generate(images,labels,REVIEW_PROMPT+parsed["caption"],384)
        validate_caption(parse_json(review["raw_text"]),review=True)
        results.append(dict(case=rel,indices=ids,draft=draft,review=review))
        for image in images:image.close()
    source=read(DEFAULT_PREPARED/"tvsum/tvsum_011_0e11bb5b/source_prepared.json")
    ids=list(range(1072,1121,4))
    images=[Image.open(check_frame(source,i)).convert("RGB") for i in ids]
    labels=[f"SOURCE FRAME ID {i}, time {i/24:.6f} seconds" for i in ids]
    response=vlm.generate(images,labels,PROPOSAL_PROMPT,768)
    validate_events(parse_json(response["raw_text"]),ids)
    results.append(dict(case="tvsum_transition_1095",indices=ids,response=response))
    output=dict(model=args.model,revision=args.revision,quantization=config["quantization"],
                total_seconds=time.monotonic()-begin,results=results,
                quality_equivalence="NOT_ESTABLISHED",independent_quality_evaluation=False)
    write_json(args.output,output)
    print(json.dumps(dict(model=args.model,total_seconds=output["total_seconds"],output=str(args.output))),flush=True)


if __name__=="__main__":main()
