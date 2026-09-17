#!/usr/bin/env python3
"""Verify model provenance, frozen definitions, and (optionally) finished scores."""
import argparse
import json
from pathlib import Path
import subprocess

from semantic_transmission.artifacts import sha256
from semantic_transmission.automatic_validation import write
from semantic_transmission.metric_v2_validation import _verify


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--complete",action="store_true")
    args=p.parse_args();root=args.output;protocol=_verify(root);models=protocol["models"];mr=Path(models["root"])
    for file,key in (("sam2.1_hiera_tiny.pt","sam2_weights_sha256"),("dinov2_vits14_pretrain.pth","dino_weights_sha256")):
        if sha256(mr/file)!=models[key]:raise ValueError("model bytes changed: "+file)
    for name in ("sam2","dinov2"):
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=mr/name,text=True).strip()
        if commit!=models[name+"_commit"]:raise ValueError("model revision changed: "+name)
        subprocess.run(["git","diff","--quiet","HEAD","--"],cwd=mr/name,check=True)
    for file,key in ((Path.home()/".cache/torch/hub/checkpoints/raft_small_C_T_V2-01064c6d.pth","raft_weights_sha256"),
                     (Path.home()/".cache/clip/ViT-B-32.pt","clip_weights_sha256")):
        if sha256(file)!=models[key]:raise ValueError("baseline weights changed")
    if sha256(Path("scripts/report_metric_v2.py"))!=protocol["report_script_sha256"]:raise ValueError("frozen report/statistics script changed")
    checked={"model_hashes":True,"model_revisions":True,"metric_code_hashes":True,"frozen_statistics_script":True}
    if args.complete:
        rows=[json.loads(x) for x in (root/"cases.jsonl").read_text().splitlines()]
        counts={}
        for stage in ("motion","object"):
            expected=[r for r in rows if stage=="motion" or r["target"] in ("control","object")]
            actual=list((root/"scores"/stage).glob("*.json"))
            if {p.stem for p in actual}!={r["case_id"] for r in expected}:raise ValueError("incomplete or extra case scores")
            by_id={r["case_id"]:r for r in expected}
            for path in actual:
                result=json.loads(path.read_text());case=by_id[path.stem]
                for key in ("source_pixel_sha256","reconstruction_pixel_sha256","split","target","kind"):
                    if result[key]!=case[key]:raise ValueError("case result mismatch")
                for key in ("mte_tail","otf_error","otf_distortion_error","oor","hor","odr"):
                    value=result.get(key)
                    if value is not None and not 0<=value<=1:raise ValueError("invalid bounded score")
                if case["variant"]=="identity":
                    for key in ("mte_tail","otf_error","otf_distortion_error"):
                        if result.get(key) is not None and abs(result[key])>1e-7:raise ValueError("identity mismatch")
            counts[stage]=len(actual)
        checked.update(score_counts=counts,identity_invariants=True,score_bounds=True,case_provenance=True)
        write(root/"integrity.json",checked)
    print(json.dumps(checked,indent=2))


if __name__=="__main__":main()
