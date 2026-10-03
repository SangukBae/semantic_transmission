#!/usr/bin/env python3
"""One original SKEM pair, compared with the preserved previously scored pair."""
import argparse
from pathlib import Path

from semantic_transmission.auto_extraction import read, write_json, DEFAULT_PREPARED, now
from semantic_transmission.hybrid_selection import InternVLScorer


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    config=read(args.config)
    source=read(DEFAULT_PREPARED/"webvid/webvid_011_65f0e6cf/source_prepared.json")
    scorer=InternVLScorer(dict(config=config["skem"]["config"],prompts=config["skem"]["prompts"],
                              source_frames=source["source_frames"]))
    result=scorer(72,92)
    old=read(DEFAULT_PREPARED/"webvid/webvid_011_65f0e6cf/extraction/scores/00072_00092.json")["score"]
    deltas={key:abs(result[key]-old[key]) for key in ("p_yes","p_no")}
    passed=all(delta<1e-6 for delta in deltas.values())
    write_json(args.output,dict(status="PASS" if passed else "CHANGED",utc=now(),
        reference=72,candidate=92,source_sha256=source["source_sha256"],result=result,
        previous_score=str(DEFAULT_PREPARED/"webvid/webvid_011_65f0e6cf/extraction/scores/00072_00092.json"),
        probability_absolute_deltas=deltas,interpretation="one-pair numeric regression, not selector quality evaluation"))
    if not passed:raise RuntimeError("SKEM pair output changed")


if __name__=="__main__":main()
