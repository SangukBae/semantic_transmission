#!/usr/bin/env python3
"""Supplemental closest-prior comparison, declared before formal OTF scoring."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from semantic_transmission.artifacts import sha256
from semantic_transmission.automatic_validation import auc, write
from semantic_transmission.metric_v2_validation import _verify
from semantic_transmission.object_metric import ObjectExtractor, associate_frames
from semantic_transmission.tracking_baselines import identity_baselines


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--declare",action="store_true")
    args=p.parse_args();root=args.output;protocol=_verify(root)
    code=Path(__file__).resolve().parents[1]/"src/semantic_transmission/tracking_baselines.py"
    locked=root/"tracking_baseline_protocol.json"
    if args.declare:
        if locked.exists() or list((root/"scores/object").glob("*.json")):raise ValueError("declare only before OTF results")
        write(locked,{"time":time.time(),"code_sha256":sha256(code),"script_sha256":sha256(Path(__file__)),
            "baseline":"IDF1 with mask IoU>=.5 and framewise detection F1; same SAM2/DINO-derived tracks",
            "timing":"MTE real-video results already seen; no OTF formal results seen; no change to candidates or original gates"})
        print("tracking baseline protocol declared");return
    metadata=json.loads(locked.read_text())
    if metadata["code_sha256"]!=sha256(code) or metadata["script_sha256"]!=sha256(Path(__file__)):raise ValueError("tracking baseline code changed")
    import torch
    torch.set_num_threads(4)
    extractor=ObjectExtractor(protocol["models"]["root"],root/"object_cache")
    rows=[];last=None
    for path in sorted((root/"scores/object").glob("*.json")):
        row=json.loads(path.read_text())
        if row["source_id"]!=last:
            with np.load(root/"inputs"/(row["source_stem"]+".npz")) as f:a=f["frames"]
            ta=associate_frames(extractor(a));last=row["source_id"]
        with np.load(root/"cases"/(row["case_id"]+".npz")) as f:b=f["frames"]
        tb=associate_frames(extractor(b))
        values=identity_baselines(ta,tb)
        rows.append({k:row[k] for k in ("case_id","source_id","domain","split","target","kind","severity","otf_distortion_error")} | values)
    write(root/"tracking_baselines.json",{"rows":rows,"scope":metadata["baseline"]})
    results={}
    for domain in ("public_video","rendered"):
        group=[r for r in rows if r["domain"]==domain]
        results[domain]={}
        for metric in ("idf1_mask_error","frame_mask_error"):
            dev=[r[metric] for r in group if r["split"]=="development" and r["target"]=="control" and r[metric] is not None]
            threshold=float(np.quantile(dev,.95,method="higher")) if dev else None
            test=[r for r in group if r["split"]=="heldout"]
            neg=[r[metric] for r in test if r["target"]=="control" and r[metric] is not None]
            by_kind={}
            for kind in sorted({r["kind"] for r in test if r["target"]=="object"}):
                rr=[r for r in test if r["kind"]==kind]
                pos=[r[metric] for r in rr if r[metric] is not None]
                by_kind[kind]={"auc":auc(pos,neg),"tpr":sum(v>threshold for v in pos)/len(rr) if threshold is not None else None,
                               "coverage":len(pos)/len(rr)}
            differences=[]
            for sid in sorted({r["source_id"] for r in test}):
                common=[r for r in test if r["source_id"]==sid and r[metric] is not None and r["otf_distortion_error"] is not None]
                pos=[r for r in common if r["target"]=="object"];nn=[r for r in common if r["target"]=="control"]
                if pos and nn:
                    differences.append(auc([r["otf_distortion_error"] for r in pos],[r["otf_distortion_error"] for r in nn])-auc([r[metric] for r in pos],[r[metric] for r in nn]))
            ci=None
            if differences:
                draws=np.random.default_rng(20260914).integers(0,len(differences),(500,len(differences)))
                ci=np.quantile(np.asarray(differences)[draws].mean(1),[.025,.975]).tolist()
            results[domain][metric]={"threshold":threshold,"fpr":float(np.mean(np.asarray(neg)>threshold)) if neg and threshold is not None else None,
                "errors":by_kind,"otf_minus_baseline_within_source_auc":float(np.mean(differences)) if differences else None,
                "otf_minus_baseline_source_bootstrap_95ci":ci,"paired_sources":len(differences)}
    write(root/"tracking_baseline_comparison.json",results)
    print(json.dumps(results,indent=2))


if __name__=="__main__":main()
