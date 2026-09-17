"""New-video framewise SGD-JSCC reconstruction and common pair registration.

This calls real pretrained SGD-JSCC through the existing sibling package. It
does not emulate the historical int4 temporal-reuse experiment.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from .artifacts import sha256
from .automatic_validation import write
from .research_quality import read_video, read_frames
from .video_io import probe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sgd-repo", type=Path, default=Path(__file__).resolve().parents[3] / "sgdjscc_lab")
    p.add_argument("--python", default="/home/sangukbae/anaconda3/envs/ptest/bin/python")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--snr", type=float, default=10)
    p.add_argument("--seed", type=int, default=20260912)
    p.add_argument("--without-text", action="store_true", help="Explicit ablation only: disable BLIP2 captions")
    args = p.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    source, root, repo = args.input.resolve(), args.output.resolve(), args.sgd_repo.resolve()
    root.mkdir(parents=True, exist_ok=False)
    info = probe(source)
    checkpoint_names = ("JSCC_model.pth", "diffusion_backbone.pth", "diffusion_controlnet.pth", "muge-epoch-19-checkpoint.pth")
    config = {"_defaults_": [str(repo / "configs/base/default")], "model_root": str(repo / "checkpoints"),
              "input_path": str(root / "source_frames"), "output_dir": str(root / "reconstruction"),
              "snr_db": args.snr, "diffusion_step": args.steps, "use_text": not args.without_text}
    # JSON is a YAML subset and can be consumed by the sibling OmegaConf loader.
    write(root / "sgd_config.yaml", config)
    state = {"status": "RUNNING", "source": str(source), "source_sha256": sha256(source), "source_info": info,
             "config": config, "seed": args.seed, "started_unix": time.time(),
             "checkpoint_sha256": {name: sha256(repo / "checkpoints" / name) for name in checkpoint_names},
             "backend_scope": "framewise SGD-JSCC AWGN; full diffusion and ControlNet; no interframe reuse",
             "sgd_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
             "sgd_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True))}
    write(root / "manifest.json", state)
    try:
        from PIL import Image
        frames = read_video(source)
        if len(frames) != info["frames"]:
            raise ValueError("source decode incomplete")
        (root / "source_frames").mkdir()
        for i, f in enumerate(frames):
            Image.fromarray(f).save(root / "source_frames" / f"{i:06d}.png")
        request = dict(sgd_repo=str(repo), config=str(root / "sgd_config.yaml"), seed=args.seed,
                       frames=str(root / "source_frames"), output=str(root / "reconstruction"))
        write(root / "request.json", request)
        command = [args.python, str(Path(__file__).resolve().parents[2] / "scripts/sgd_bridge_worker.py"), "--request", str(root / "request.json")]
        env = os.environ.copy()
        for key in ("PYTHONPATH", "LD_LIBRARY_PATH"):
            env.pop(key, None)
        env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8")
        state["command"] = command
        write(root / "manifest.json", state)
        with (root / "worker.log").open("w") as stream:
            subprocess.run(command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        expected = {p.name for p in (root / "source_frames").glob("*.png")}
        actual = {p.name for p in (root / "reconstruction").glob("*.png")}
        if expected != actual:
            raise ValueError("SGD output frames missing or extra")
        rec = read_frames(root / "reconstruction")
        if rec.shape != frames.shape:
            raise ValueError("SGD reconstruction geometry mismatch")
        video = root / "recon.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-framerate", str(info["fps"]), "-i", str(root / "reconstruction/%06d.png"),
                        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(video)], check=True)
        delivered = probe(video)
        if any(delivered[k] != info[k] for k in ("frames", "width", "height", "fps")):
            raise ValueError("encoded SGD video contract mismatch")
        if sha256(source) != state["source_sha256"]:
            raise ValueError("source bytes changed during SGD reconstruction")
        pair = {"source_id": source.stem, "source": str(source), "source_sha256": sha256(source),
                "reconstruction": str(video), "reconstruction_sha256": sha256(video),
                "model": f"SGD_JSCC_framewise_awgn_{args.steps}" + ("_without_text" if args.without_text else ""),
                "provenance": str(root / "manifest.json")}
        write(root / "pairs.json", {"schema": "source-reconstruction-pairs-v1", "pairs": [pair]})
        state.update(status="PASSED", verified_frames=len(rec), reconstructed_sha256=sha256(video))
    except BaseException as error:
        state.update(status="FAILED", error=str(error))
        raise
    finally:
        state["elapsed_s"] = time.time() - state["started_unix"]
        write(root / "manifest.json", state)


if __name__ == "__main__":
    main()
