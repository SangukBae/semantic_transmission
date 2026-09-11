"""Frozen-profile sequential reconstruction of a complete video dataset."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .artifacts import git_state, runtime_state, sha256, write_json
from .cli import doctor, repository, settings
from .video_io import probe


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--video", action="append", help="optional video stem filter")
    parser.add_argument("--reuse-completed-from", type=Path, action="append", default=[])
    parser.add_argument("--dry-run", action="store_true", help="verify inputs and reuse candidates without running models")
    args = parser.parse_args(argv)
    repo = repository()
    local = settings(repo)
    if doctor(repo)["status"] != "PASSED":
        raise RuntimeError("environment preflight failed")
    profile = args.profile or repo / "configs/etri_hq.json"
    cfg = json.loads(profile.read_text())
    if cfg.get("selector_environment") == "internvl_release" and not Path(local.get("internvl_python", "")).is_file():
        raise RuntimeError("Run bash scripts/bootstrap_internvl.sh to install the release's selector environment")
    cfg["models"] = json.loads((repo / ".local/model_paths.json").read_text())
    videos = sorted(args.input_dir.resolve().glob("*.mp4"))
    if args.video:
        videos = [v for v in videos if v.stem in args.video]
    if not videos:
        raise ValueError("no input videos")
    sources = []
    for path in videos:
        info = probe(path)
        expected_source = cfg.get("source_video", cfg)
        if any(info[key] != expected_source[key] for key in ("frames", "width", "height", "fps")):
            raise ValueError(f"{path.name} differs from the frozen video profile: {info}")
        sources.append({"id": path.stem, "path": str(path), "sha256": sha256(path), **info})
    root = args.output.resolve()
    from .resume import completed_runs, copy_completed
    reusable = completed_runs(args.reuse_completed_from, cfg, sources)
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_PASSED", "output": str(root),
                          "reused": list(reusable),
                          "to_generate": [s["id"] for s in sources if s["id"] not in reusable]}, indent=2))
        return
    root.mkdir(parents=True, exist_ok=False)
    state = {"status": "RUNNING", "started": utc(), "code": git_state(repo),
             "profile_sha256": sha256(profile), "inputs": sources, "runs": [],
             "expected_videos": len(videos), "completed_videos": 0, "failed_videos": 0}
    write_json(root / "profile.json", cfg)
    write_json(root / "environment.json", runtime_state(repo, local))
    state["environment_sha256"] = sha256(root / "environment.json")
    write_json(root / "batch_manifest.json", state)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("LD_LIBRARY_PATH", None)
    env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", PYTHONHASHSEED=str(cfg["seed"]),
               OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false")
    stages = [(s, "semantic_transmission.workers") for s in ("prepare", "select", "caption", "flow")]
    stages += [(s, "semantic_transmission.codec_transport") for s in ("send", "channel", "receive", "reconstruct")]
    stages += [("evaluate", "semantic_transmission.research_quality")]
    try:
        for source in sources:
            run = root / source["id"]
            if source["id"] in reusable:
                record = copy_completed(reusable[source["id"]], run)
                state["runs"].append(record)
                state["completed_videos"] += 1
                write_json(run / "run_manifest.json", record)
                write_json(root / "batch_manifest.json", state)
                print(f"Reused verified video {state['completed_videos']}/{len(sources)}: {source['id']}", flush=True)
                continue
            run.mkdir()
            (run / "logs").mkdir()
            write_json(run / "run_config.json", dict(cfg, input=source["path"]))
            record = {"id": source["id"], "status": "RUNNING", "started": utc(), "stages": []}
            state["runs"].append(record)
            for stage, module in stages:
                entry = {"stage": stage, "status": "RUNNING", "started": utc()}
                record["stages"].append(entry)
                write_json(root / "batch_manifest.json", state)
                write_json(run / "run_manifest.json", record)
                python = local["channel_python"] if stage == "channel" else local["python"]
                command = [python, "-m", module, stage, str(run)]
                entry["command"] = command
                stage_env = env.copy()
                if stage == "channel":
                    stage_env["CUDA_VISIBLE_DEVICES"] = "-1"
                print(f"{source['id']} — {stage}", flush=True)
                start = time.monotonic()
                with (run / "logs" / f"{stage}.log").open("w") as stream:
                    result = subprocess.run(command, env=stage_env, stdout=stream, stderr=subprocess.STDOUT)
                entry.update(seconds=time.monotonic() - start, returncode=result.returncode,
                             finished=utc(), status="PASSED" if result.returncode == 0 else "FAILED")
                write_json(run / "run_manifest.json", record)
                if result.returncode:
                    record["status"] = "FAILED"
                    state["failed_videos"] += 1
                    raise RuntimeError(f"{source['id']}/{stage} failed; see its stage log")
            record.update(status="PASSED", finished=utc())
            state["completed_videos"] += 1
            write_json(run / "run_manifest.json", record)
            write_json(root / "batch_manifest.json", state)
            print(f"Completed {state['completed_videos']}/{len(sources)}: {source['id']}", flush=True)
        state["status"] = "PASSED"
    except BaseException as error:
        state.update(status="FAILED", error=str(error))
        if state["runs"]:
            state["runs"][-1].update(status="FAILED", error=str(error), finished=utc())
            write_json(root / state["runs"][-1]["id"] / "run_manifest.json", state["runs"][-1])
        raise
    finally:
        state["finished"] = utc()
        write_json(root / "batch_manifest.json", state)


if __name__ == "__main__":
    main()
