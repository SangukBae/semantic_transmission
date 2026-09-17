#!/usr/bin/env python3
"""Repeat one fixed development pair with fresh feature/flow caches."""
import argparse
import json
from pathlib import Path

import numpy as np

from semantic_transmission.automatic_validation import write
from semantic_transmission.metric_v2_validation import _verify


def flatten(x,path="",values=None):
    values={} if values is None else values
    if isinstance(x,dict):
        for k,v in x.items():flatten(v,path+"/"+k,values)
    elif isinstance(x,(list,tuple)):
        for i,v in enumerate(x):flatten(v,path+"/"+str(i),values)
    else:values[path]=x
    return values


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--stage",choices=("motion","object"),required=True)
    args=p.parse_args();root=args.output;protocol=_verify(root)
    import torch
    torch.set_num_threads(4);torch.manual_seed(20260914)
    case_id="renderer__260914100__object_omission_0.25"
    with np.load(root/"inputs/renderer__260914100.npz") as f:a=f["frames"]
    with np.load(root/"cases"/(case_id+".npz")) as f:b=f["frames"]
    if args.stage=="motion":
        from semantic_transmission.motion_metric import MotionMetric
        metric=MotionMetric()
    else:
        from semantic_transmission.object_metric import ObjectMetric
        metric=ObjectMetric(protocol["models"]["root"],root/"repeat_object_cache_0")
    results=[]
    for i in range(2):
        if args.stage=="motion":metric.extract.cache.clear()
        else:
            path=root/("repeat_object_cache_"+str(i))
            path.mkdir(exist_ok=True)
            if list(path.glob("*.npz")):raise ValueError("repeat cache must start empty")
            metric.extract.cache_root=path
        results.append(metric.evaluate(a,b))
    x,y=map(flatten,results);mismatches=[];deltas=[]
    for key in set(x)|set(y):
        if key not in x or key not in y:
            mismatches.append(key);continue
        if isinstance(x[key],(float,int)) and isinstance(y[key],(float,int)):
            deltas.append(abs(x[key]-y[key]))
        elif x[key]!=y[key]:mismatches.append(key)
    write(root/(args.stage+"_repeatability.json"),{"case_id":case_id,"scope":"fixed-pair numerical repeatability; not independent reconstruction seeds",
          "fresh_caches":True,"numeric_values":len(deltas),"max_abs_difference":max(deltas) if deltas else None,
          "structural_mismatches":mismatches,"results":results})
    print(args.stage,"repeatability",len(deltas),max(deltas),mismatches,flush=True)


if __name__=="__main__":main()
