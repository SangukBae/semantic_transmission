#!/usr/bin/env python3
"""Freeze reporting code after corpus preparation and before any scores exist."""
import argparse
import json
from pathlib import Path
import shutil

from semantic_transmission.artifacts import sha256
from semantic_transmission.automatic_validation import write
from semantic_transmission.metric_v2_validation import _verify

p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);args=p.parse_args();root=args.output
protocol=_verify(root)
if list((root/"scores").rglob("*.json")):
    raise ValueError("reporting definition must be frozen before scores")
report=Path(__file__).with_name("report_metric_v2.py")
if "report_script_sha256" in protocol:
    if protocol["report_script_sha256"]!=sha256(report):raise ValueError("already frozen to another reporting definition")
else:
    protocol["report_script_sha256"]=sha256(report)
    write(root/"protocol.json",protocol)
    shutil.copy2(report,root/"frozen_source"/report.name)
    prep=json.loads((root/"preparation.json").read_text());prep["protocol_sha256"]=sha256(root/"protocol.json")
    write(root/"preparation.json",prep)
print("reporting definition frozen before scoring")
