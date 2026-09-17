#!/usr/bin/env python3
"""Independent measurement-tool audit. Truth is used here, never in candidates."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from semantic_transmission.automatic_validation import write
from semantic_transmission.metric_v2_validation import _verify
from semantic_transmission.object_metric import ObjectExtractor, iou, associate_frames
from semantic_transmission.motion_metric import FlowExtractor, describe_flow, motion_events, match_events
from semantic_transmission.tracking_baselines import identity_baselines


def mask_match(detections, label):
    truth=[label==k for k in np.unique(label) if k]
    cost=np.full((len(truth),len(detections)),1e6)
    for i,mask in enumerate(truth):
        for j,det in enumerate(detections):
            overlap=iou(mask,det["mask"])
            if overlap>=.5:cost[i,j]=1-overlap
    matches=[(int(i),int(j)) for i,j in zip(*linear_sum_assignment(cost)) if cost[i,j]<1e6]
    return {"true_visible_objects":len(truth),"predicted_objects":len(detections),"matched":len(matches),
            "matched_ious":[1-cost[i,j] for i,j in matches]}


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--stage",choices=("object","motion"),required=True)
    p.add_argument("--cached-only",action="store_true")
    p.add_argument("--domain",choices=("public_video","rendered"))
    p.add_argument("--split",choices=("development","heldout"))
    args=p.parse_args();root=args.output;protocol=_verify(root)
    import torch
    torch.set_num_threads(4)
    if args.cached_only:
        if args.stage!="object":raise ValueError("only object extraction has a disk cache")
        class CachedExtractor:
            def frame(self,frame):
                signature=hashlib.sha256((protocol['code_sha256']['object_metric.py']+protocol['models']['sam2_weights_sha256']+protocol['models']['dino_weights_sha256']).encode()).hexdigest()
                key=hashlib.sha256(frame.tobytes()+str(frame.shape).encode()+signature.encode()).hexdigest()
                with np.load(root/'object_cache'/(key+'.npz')) as f:
                    return [{"mask":m.astype(bool),"feature":v} for m,v in zip(f['masks'],f['features'])]
        extractor=CachedExtractor()
    else:
        extractor=ObjectExtractor(protocol["models"]["root"],root/"object_cache") if args.stage=="object" else FlowExtractor()
    results=[]
    for item in protocol["inventory"]:
        if args.domain and item["domain"]!=args.domain:continue
        if args.split and item["split"]!=args.split:continue
        if args.stage=="motion" and item["domain"]!="rendered":continue
        stem=item["source_id"].replace("/","__")
        with np.load(root/"inputs"/(stem+".npz")) as f:frames=f["frames"]
        if item["domain"]=="rendered":
            with np.load(root/"inputs"/(stem+"_truth.npz")) as f:labels,centers=f["masks"],f["centers"]
        else:
            files=[Path(x["path"].replace("JPEGImages","Annotations")).with_suffix(".png") for x in item["files"]]
            # PIL retains palette object IDs; cv2 grayscale would map palette colors.
            from PIL import Image
            labels=np.stack([cv2.resize(np.array(Image.open(f)),(320,192),interpolation=cv2.INTER_NEAREST) for f in files])
        if args.stage=="object":
            detections=[extractor.frame(frame) for frame in frames]
            audits=[mask_match(dets,label) for dets,label in zip(detections,labels)]
            n=sum(a["true_visible_objects"] for a in audits);pred=sum(a["predicted_objects"] for a in audits);hit=sum(a["matched"] for a in audits)
            tracking=None
            if item["domain"]=="rendered":
                truth_tracks=[{t:{"mask":label==k} for t,label in enumerate(labels) if (label==k).any()} for k in np.unique(labels) if k]
                tracking=identity_baselines(truth_tracks,associate_frames(detections))
            row={"source_id":item["source_id"],"domain":item["domain"],"split":item["split"],"annotated_object_frames":n,
                 "predicted_object_frames":pred,"matched_object_frames":hit,"recall_iou50":hit/n if n else None,
                 "precision_iou50":hit/pred if pred and item["domain"]=="rendered" else None,
                 "renderer_truth_tracking":tracking,
                 "precision_scope":"DAVIS annotates selected objects; unmatched predictions there are not automatically false positives",
                 "frames":audits}
        else:
            flows=extractor(frames);errors=[];background=[]
            for t,flow in enumerate(flows):
                for k in np.unique(labels[t]):
                    if not k or not (labels[t+1]==k).any():continue
                    visible=labels[t]==k
                    expected=centers[t+1,int(k)-1]-centers[t,int(k)-1]
                    errors.extend(np.linalg.norm(flow[visible]-expected,axis=-1).tolist())
                bg=(labels[t]==0)&(labels[t+1]==0)
                background.extend(np.linalg.norm(flow[bg],axis=-1).tolist())
            # Independent renderer turning points, retaining object identity.
            true_events=[]
            for k in range(centers.shape[1]):
                velocity=np.diff(centers[:,k].astype(float),axis=0)
                valid=[t for t,v in enumerate(velocity) if np.linalg.norm(v)>0 and (labels[t]==k+1).any() and (labels[t+1]==k+1).any()]
                for left,right in zip(valid,valid[1:]):
                    if np.dot(velocity[left],velocity[right])<0:
                        true_events.append({"type":"turn","time_s":(left+right+1)/16.,"object_id":k+1})
            predicted=motion_events(describe_flow(flows),8.)
            row={"source_id":item["source_id"],"split":item["split"],"foreground_epe_px":float(np.mean(errors)),
                 "background_epe_px":float(np.mean(background)),"foreground_pixels":len(errors),
                 "renderer_turn_events":true_events,"predicted_motion_events":predicted,
                 "turn_event_audit":match_events(true_events,predicted),
                 "scope":"renderer source translation only; visible objects, no reference truth input to RAFT"}
        results.append(row);print(item["source_id"],args.stage,"audit done",flush=True)
    suffix="_"+args.domain if args.domain else ""
    if args.split:suffix+="_"+args.split
    write(root/(args.stage+"_extractor_audit"+suffix+".json"),{"stage":args.stage,"results":results,
          "candidate_input_uses_truth":False,"scope":"extractor accuracy is distinct from candidate error-detection accuracy"})


if __name__=="__main__":main()
