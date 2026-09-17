#!/usr/bin/env python3
"""Supplementary coverage/monotonicity diagnostics; does not change frozen gates."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from semantic_transmission.automatic_validation import write
from semantic_transmission.metric_v2_validation import _verify


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);args=p.parse_args();root=args.output
    _verify(root)
    merged={}
    for stage in ("motion","object"):
        for path in (root/"scores"/stage).glob("*.json"):
            row=json.loads(path.read_text());merged.setdefault(row["case_id"],{}).update(row)
    rows=list(merged.values());out={"scope":"supplementary diagnostics, no threshold/formula selection or gate replacement"}
    control=[r for r in rows if r["domain"]=="public_video" and r["target"]=="control"]
    controls={}
    for metric in ("otf_error","otf_distortion_error","odr"):
        dev=[r[metric] for r in control if r["split"]=="development" and r.get(metric) is not None]
        test=[r for r in control if r["split"]=="heldout"]
        measured=[r for r in test if r.get(metric) is not None]
        threshold=float(np.quantile(dev,.95,method="higher")) if dev else None
        controls[metric]={"development_controls":len(dev),"heldout_controls":len(test),"measured_controls":len(measured),
            "coverage":len(measured)/len(test),"threshold":threshold,
            "false_alarm_rate":float(np.mean([r[metric]>threshold for r in measured])) if measured and threshold is not None else None,
            "semantic_scope":"appearance nuisances only; no real-video object error truth"}
    out["public_object_controls"]=controls
    curves={}
    for domain,target,metric in (("public_video","motion","mte_tail"),("rendered","motion","mte_tail"),("rendered","object","otf_distortion_error")):
        group=[r for r in rows if r["domain"]==domain and r["target"]==target and r["split"]=="heldout"]
        curves[domain+"/"+target]={}
        for kind in sorted({r["kind"] for r in group}):
            rhos=[];eligible=0
            for sid in sorted({r["source_id"] for r in group}):
                rr=sorted([r for r in group if r["source_id"]==sid and r["kind"]==kind and r.get(metric) is not None],key=lambda r:r["severity"])
                if len(rr)==3:
                    eligible+=1
                    if np.ptp([r[metric] for r in rr])>1e-10:
                        rhos.append(float(spearmanr([r["severity"] for r in rr],[r[metric] for r in rr]).statistic))
            curves[domain+"/"+target][kind]={"evaluable_sources":len(rhos),"complete_sources":eligible,
                  "constant_response_sources":eligible-len(rhos),"median_severity_spearman":float(np.median(rhos)) if rhos else None}
    out["severity_response"]=curves
    out["null_cases"]={stage:sum(r.get(metric) is None for r in rows if r["target"] in targets) for stage,metric,targets in
        (("motion","mte_tail",("motion","control")),("object","otf_distortion_error",("object","control")))}
    write(root/"supplementary_diagnostics.json",out)
    print(json.dumps(controls,indent=2))


if __name__=="__main__":main()
