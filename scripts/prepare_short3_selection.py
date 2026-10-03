"""Prepare/replay the reviewed three-video selection; never reconstruct."""
import argparse
import ast
import datetime
from pathlib import Path

from semantic_transmission import hybrid_selection as hybrid
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.webvid5 import fingerprint, read_json

REPO = hybrid.REPO
ROOT = REPO / "outputs/etri_latest_short3_20261001"
AUTHORED = REPO / "configs/captions/short3_source_observations_v2.json"


def prepare(name, spec):
    root = ROOT / name
    if (root / "protocol.json").exists():
        return hybrid.validate_protocol(root)
    base = REPO / spec["baseline"]
    old = read_json(base / "run_config.json")
    cfg = dict(read_json(hybrid.BASE / "run_config.json"))
    for field in ("selector_checkpoint", "caption_checkpoint", "hybrid_selection_root", "caption_bundle"):
        cfg.pop(field, None)
    video = base / "data/normalized.mp4"
    cfg.update(input=str(video), input_sha256=sha256(video), frames=old["frames"], fps=old["fps"],
               width=old["width"], height=old["height"], max_frames=old["frames"], preserve_input=True,
               semantic_clip_policy="frame_exact", concatenation_policy="endpoint_exact", seed=2025)
    tree = ast.parse(hybrid.SKEM.read_text())
    prompts = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.walk(tree)
               if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
               and n.targets[0].id in ("prompt_ask_image", "prompt_compare_image")}
    files = ["src/semantic_transmission/hybrid_selection.py", "02_semantic_encoder/skem/MLM-keyframe-internvl.py",
             "src/semantic_transmission/internvl_memory.py", "src/semantic_transmission/exact_reuse.py",
             "scripts/prepare_short3_selection.py"]
    protocol = dict(version=1, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_id=name, source_split="existing_local_short_video_comparison", baseline=str(base),
        input=str(video), input_sha256=sha256(video), normalized_video=str(video), normalized_sha256=sha256(video),
        frames=old["frames"], fps=old["fps"], width=old["width"], height=old["height"],
        source_frames=str(base / "data/frames/sample"),
        source_frame_hashes={str(i):sha256(base / f"data/frames/sample/{i}.png") for i in range(old["frames"])},
        proposal_path=str(AUTHORED), proposal_sha256=sha256(AUTHORED),
        candidates=[dict(frame=i, mandatory=mandatory, kind="observed_event" if mandatory else "optional_change",
                         reason_ko=reason) for i,mandatory,reason in spec["candidates"]],
        max_gap_frames=24, threshold=0.35, config=cfg, prompts=prompts,
        model_files=hybrid.model_inventory(cfg["models"]["internvl"]),
        code={p:sha256(REPO / p) for p in files},
        scope="Offline assistant proposals from source frames; no reconstruction observed for proposal authoring. Not independent event ground truth.",
        sampling="Every fourth source frame at 24fps plus consecutive person frames 162-171 and 174-194.",
        evidence={str(p.relative_to(root)):sha256(p) for p in sorted((root / "observations").glob("*.jpg"))},
        reconstruction_status="NOT_STARTED", hallucination_mitigation_verified=False)
    protocol["signature"] = fingerprint(protocol)
    write_json(root / "protocol.json", protocol)
    return hybrid.validate_protocol(root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--select", action="store_true")
    args = p.parse_args()
    for name,spec in read_json(AUTHORED)["videos"].items():
        prepare(name,spec)
        if args.select:
            hybrid.select(ROOT / name)
            path = ROOT / name / "selection.html"
            path.write_text(path.read_text().replace("tv_low_08 혼합", f"{name} 혼합"))
        print(f"PREPARED SELECTION {name}; reconstruction NOT_STARTED", flush=True)


if __name__ == "__main__":
    main()
