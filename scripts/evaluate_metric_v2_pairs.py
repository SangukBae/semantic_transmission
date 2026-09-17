#!/usr/bin/env python3
"""Apply MTE or OTF to any registered LGVSC/SGD pair, retaining playback delay."""
import argparse
import json
from pathlib import Path
import time

from semantic_transmission.automatic_validation import write
from semantic_transmission.artifacts import sha256
from semantic_transmission.pair_inputs import load_pair


def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("motion","object"))
    p.add_argument("--pairs",type=Path,action="append",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--models",type=Path,default=Path(".local/metric_v2_models"))
    p.add_argument("--cache",type=Path,default=Path("outputs/metric_v2_pair_cache"))
    p.add_argument("--fps",type=float,default=8.)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    import torch
    torch.set_num_threads(4);torch.manual_seed(20260914)
    if args.stage=="motion":
        from semantic_transmission.motion_metric import MotionMetric
        metric=MotionMetric(fps=args.fps)
    else:
        from semantic_transmission.object_metric import ObjectMetric
        metric=ObjectMetric(args.models,args.cache,fps=args.fps)
    results=[]
    for manifest_path in args.pairs:
        manifest=json.loads(manifest_path.read_text())
        if manifest.get("schema")!="source-reconstruction-pairs-v1":raise ValueError("invalid manifest")
        for row in manifest["pairs"]:
            source,rec,alignment=load_pair(row,width=320,height=192,sample_fps=args.fps)
            started=time.time();score=metric.evaluate(source,rec)
            results.append({**row,"alignment":alignment,"scores":score,"semantic_ground_truth":None,"elapsed_s":time.time()-started})
            print(row["source_id"],row["model"],"done",len(source),"samples",flush=True)
    write(args.output,{"status":"COMPLETED","scope":"unlabelled reconstruction diagnostics, not hallucination truth or matched-rate model comparison",
        "stage":args.stage,"fps":args.fps,"resolution":[320,192],"results":results,
        "torch":torch.__version__,"script_sha256":sha256(Path(__file__)),
        "pair_input_sha256":sha256(Path(__file__).resolve().parents[1]/"src/semantic_transmission/pair_inputs.py")})


if __name__=="__main__":main()
