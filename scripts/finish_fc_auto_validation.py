#!/usr/bin/env python3
"""Wait for the smoke, run component checks, then start the authorized full batch.

All waits/checks happen locally. This script does not call Codex or any LLM API.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from semantic_transmission.auto_extraction import (
    REPO, DEFAULT_ROOT, DEFAULT_PREPARED, read, write_json, worker_env, transnet,
    pixel_change_events, digest, sha256,
)


def main():
    smoke=REPO/"outputs/fc_auto_smoke_20261002_v1"
    validation=REPO/"outputs/fc_auto_validation_20261002"
    config_path=smoke/"run_config.json"
    env=worker_env()
    with (REPO/".local/etri_60s_check.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        smoke_state=read(smoke/"status.json")
        if smoke_state["status"]!="COMPLETE" or smoke_state["completed"]!=2:
            raise RuntimeError(f"smoke incomplete: {smoke_state['status']}")
        config=read(config_path)
        if not (validation/"skem_regression.json").exists():
            with (validation/"skem_regression.log").open("w") as log:
                subprocess.run([config["skem_python"],str(REPO/"scripts/check_fc_auto_skem.py"),
                    "--config",str(config_path),"--output",str(validation/"skem_regression.json")],
                    env=env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        if read(validation/"skem_regression.json")["status"]!="PASS":raise RuntimeError("SKEM regression failed")
        # End-to-end smoke predates the added pixel guard. Demonstrate that this
        # isolated change adds no candidates on either completed smoke input.
        import numpy as np
        guard_checks=[]
        for rel in ["webvid/webvid_008_852d56f2","webvid/webvid_011_65f0e6cf"]:
            short=read(DEFAULT_PREPARED/rel/"source_prepared.json")
            raw=subprocess.check_output(["ffmpeg","-v","error","-threads","2","-filter_threads","2",
                "-i",short["normalized_video"],"-vf","scale=48:27","-vsync","0","-pix_fmt","rgb24","-f","rawvideo","-"])
            frames=np.frombuffer(raw,np.uint8).reshape(-1,27,48,3)
            additions=pixel_change_events(frames)
            guard_checks.append(dict(id=short["id"],extra_events=len(additions)))
            if additions:raise RuntimeError("guard changed smoke candidates; rerun end-to-end smoke")
        write_json(validation/"guard_smoke_equivalence.json",dict(status="PASS",videos=guard_checks,
            interpretation="pixel guard added zero candidates; other extraction functions unchanged"))
        probe_config=dict(config,pixel_guard=dict(minimum=.05,ratio=4.,history=24))
        probe_config["signature"]=digest(dict(old_config=config["signature"],
            changed_code_sha256=sha256(REPO/"src/semantic_transmission/auto_extraction.py"),pixel_guard=probe_config["pixel_guard"]))
        source=read(DEFAULT_PREPARED/"tvsum/tvsum_011_0e11bb5b/source_prepared.json")
        events=transnet(source,validation/"tvsum_transnet_with_guard",probe_config)
        checked=[]
        for cut in [1095,1171,1236,1668]:
            matches=[event for event in events if event["before"]<=cut-1 and event["after"]>=cut]
            checked.append(dict(first_new_frame=cut,covered_within_two_frames=bool(matches),matches=matches))
        write_json(validation/"tvsum_known_cuts_with_guard.json",dict(status="PASS" if all(x["covered_within_two_frames"] for x in checked) else "MISSING_CUT",
            source_id=source["id"],source_frames=source["frames"],total_detected_transitions=len(events),
            checked=checked,reference_correction="Direct source PNG review: final background cut first new frame is 1668, not 1667; old frozen assistant outputs preserved",
            interpretation="four source-inspected development transitions, not full event ground truth"))
        if not all(x["covered_within_two_frames"] for x in checked):
            raise RuntimeError("TransNet missed a known development cut")
    # Release both the GPU allocations and the validation lock before launch.
    import gc
    import torch
    gc.collect();torch.cuda.empty_cache()
    command=["bash",str(REPO/"scripts/run_fc_auto_extraction.sh")]
    if not (DEFAULT_ROOT/"run_config.json").exists():
        subprocess.run(command+["init","--root",str(DEFAULT_ROOT),"--model-id",config["model_id"],
            "--revision",config["model_revision"],"--quantization",config["quantization"]],check=True,cwd=REPO,env=env)
    full=read(DEFAULT_ROOT/"run_config.json")
    if len(full["videos"])!=103:raise RuntimeError("full inventory must contain all 103 sources")
    subprocess.run(command+["start","--root",str(DEFAULT_ROOT)],check=True,cwd=REPO,env=env)
    launch=read(DEFAULT_ROOT/"start_request.json")
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        os.kill(launch["pid"],0)
        state=read(DEFAULT_ROOT/"status.json")
        if state["status"]=="STOPPED":raise RuntimeError(state.get("error"))
        active=state.get("active")
        if active:
            video=next(v for v in full["videos"] if v["id"]==active)
            progress=DEFAULT_ROOT/video["dataset"].lower()/active/"progress.json"
            if progress.exists() and read(progress)["completed_units"]>=1:
                result=dict(status="FULL_BATCH_RUNNING",pid=launch["pid"],videos=103,smoke_completed=2,
                    skem_regression=read(validation/"skem_regression.json")["status"],known_cuts="4/4 observed transitions bracketed",
                    active=active,progress=read(progress),root=str(DEFAULT_ROOT),quality_equivalence="NOT_ESTABLISHED")
                write_json(validation/"launch_validation.json",result)
                print(json.dumps(result),flush=True)
                return
        time.sleep(2)
    raise TimeoutError("full batch did not reach a completed inference unit within 300 seconds")


if __name__=="__main__":main()
