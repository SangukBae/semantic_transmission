"""Command-line entry point for local LGVSC research."""

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .artifacts import git_state, sha256, write_json


def repository():
    return Path(__file__).resolve().parents[2]


def settings(repo):
    path = repo / ".local/settings.json"
    if not path.exists():
        raise FileNotFoundError("Run scripts/bootstrap.sh first; .local/settings.json is missing")
    return json.loads(path.read_text())


def doctor(repo, *, check_models=True):
    local = settings(repo)
    model_file = repo / ".local/model_paths.json"
    models = json.loads(model_file.read_text()) if model_file.exists() else {}
    checks = {"ffmpeg": shutil.which("ffmpeg") is not None,
              "core_python": Path(local["python"]).is_file(),
              "channel_python": Path(local["channel_python"]).is_file()}
    for name in ["Open-Sora", "NTSCC_JSAC22", "InternVL", "PLLaVA"]:
        checks[name] = (repo / ".local/vendor" / name / ".git").exists()
    if check_models:
        for name in ["internvl", "stdit", "vae", "t5", "pllava"]:
            checks["model_" + name] = name in models and (Path(models[name]) / "config.json").is_file()
        checks["model_vae2d"] = "vae2d" in models and (Path(models["vae2d"]) / "vae/config.json").is_file()
        for name in ["ntscc_hyperprior_quality_4_psnr.pth", "unimatch.pth"]:
            checks[name] = (repo / ".local/checkpoints" / name).is_file()
    result = {"status": "PASSED" if all(checks.values()) else "NOT_READY", "checks": checks,
              "settings": local, "code": git_state(repo)}
    print(json.dumps(result, indent=2))
    return result


def smoke(args, repo):
    local = settings(repo)
    if doctor(repo)["status"] != "PASSED":
        raise RuntimeError("environment/model preflight failed")
    if args.frames < 2 or min(args.width, args.height) < 128 or args.width % 64 or args.height % 64:
        raise ValueError("use >=2 frames and dimensions >=128 divisible by 64 for NTSCC")
    if args.steps < 1 or args.stride < 1 or not 2 <= args.skim_keyframes <= args.frames:
        raise ValueError("invalid sampling steps, stride or keyframe count")
    source = args.input or next((repo / "assets/sample").glob("*.mp4"))
    source = source.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    cfg = {"input": str(source), "frames": args.frames, "width": args.width, "height": args.height,
           "fps": 24, "seed": args.seed, "steps": args.steps, "selector": args.selector,
           "method": "key_framesinternvl_diff_0.35" if args.selector == "skem" else "key_framesbaseline_local",
           "threshold": 0.35, "selection_stride": args.stride, "skim_keyframes": args.skim_keyframes,
           "max_new_tokens": 128, "max_tiles": 1, "internvl_8bit": True, "snr_db": 10,
           "models": json.loads((repo / ".local/model_paths.json").read_text()),
           "profile": "rtx4080_local_development", "paper_reproduction": False}
    write_json(output / "run_config.json", cfg)
    state = {"status": "RUNNING", "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "code": git_state(repo), "input_sha256": sha256(source), "stages": [],
             "profile": cfg["profile"], "paper_reproduction": False}
    write_json(output / "run_manifest.json", state)
    env = os.environ.copy()
    # ROS overlays and a system CUDA 11.x can override this environment's CUDA 12.1 wheels.
    env.pop("PYTHONPATH", None)
    env.pop("LD_LIBRARY_PATH", None)
    env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", PYTHONHASHSEED=str(args.seed),
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="8",
               CUDA_VISIBLE_DEVICES=env.get("CUDA_VISIBLE_DEVICES", "0"))
    try:
        for stage in ["prepare", "select", "caption", "flow", "ntscc", "metadata_channel", "decode", "evaluate"]:
            if stage == "metadata_channel":
                command = [local["channel_python"], "-m", "semantic_transmission.metadata_channel",
                           "--input", str(output / "metadata_tx.json"), "--output", str(output / "metadata_rx.csv"),
                           "--report", str(output / "metadata_channel.json"), "--snr", "10", "--seed", str(args.seed)]
            else:
                command = [local["python"], "-m", "semantic_transmission.workers", stage, str(output)]
            stage_env = env.copy()
            if stage == "metadata_channel":
                stage_env["CUDA_VISIBLE_DEVICES"] = "-1"
            entry = {"stage": stage, "status": "RUNNING", "command": command}
            state["stages"].append(entry)
            write_json(output / "run_manifest.json", state)
            print(f"Starting {stage}", flush=True)
            start = time.monotonic()
            with (output / "logs" / f"{stage}.log").open("w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                        env=stage_env, check=False)
            entry.update(returncode=result.returncode, seconds=time.monotonic() - start)
            if result.returncode:
                entry["status"] = "FAILED"
                raise RuntimeError(f"{stage} failed with exit {result.returncode}; see logs/{stage}.log")
            entry["status"] = "PASSED"
            write_json(output / "run_manifest.json", state)
            print(f"Finished {stage}: {entry['seconds']:.1f}s", flush=True)
        state["status"] = "PASSED"
        state["quality"] = json.loads((output / "quality.json").read_text())
    except BaseException as error:
        state.update(status="FAILED", error=str(error))
        raise
    finally:
        state["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(output / "run_manifest.json", state)
    print(f"Completed real-model execution: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="check local paths and model availability")
    run = commands.add_parser("smoke", help="run real LGVSC models on one short clip")
    run.add_argument("--input", type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--selector", choices=["skim", "skem"], default="skem")
    run.add_argument("--frames", type=int, default=17)
    run.add_argument("--width", type=int, default=256)
    run.add_argument("--height", type=int, default=256)
    run.add_argument("--steps", type=int, default=10)
    run.add_argument("--stride", type=int, default=4)
    run.add_argument("--seed", type=int, default=1024)
    run.add_argument("--skim-keyframes", type=int, default=2)
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            raise SystemExit(0 if doctor(repository())["status"] == "PASSED" else 1)
        smoke(args, repository())
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
