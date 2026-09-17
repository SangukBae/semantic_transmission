"""Frozen, source-separated MTE then OTF evaluation; no new human review.

prepare -> motion (RAFT + all baselines) -> object (SAM2/DINOv2) -> summarize.
Never tunes candidate formulas on heldout results. Repeat runs use case hashes.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import cv2
import numpy as np

from .artifacts import sha256
from .automatic_validation import auc, write
from .metric_v2_cases import FPS, HEIGHT, WIDTH, COUNT, render_scene, variants

FILES = ("motion_metric.py", "object_metric.py", "metric_v2_cases.py", "metric_v2_validation.py",
         "automatic_metrics.py", "temporal_baselines.py", "automatic_validation.py")
METRICS = {"mte_tail": 1, "mte_mean_ablation": 1, "mte_vector_ablation": 1, "mte_max": 1,
           "otf_error": 1, "otf_distortion_error": 1, "odr": 1, "rte": 1, "lssd": 1,
           "psnr_db": -1, "ssim": -1, "lpips_alex": 1, "clip_cosine": -1,
           "tof_raft_small": 1, "tof_farneback": 1, "tlp_alex": 1}


def prepare(args):
    root, data = args.output, args.data / "DAVIS"
    root.mkdir(parents=True, exist_ok=False)
    inventory = []
    for official, split, count in (("train", "development", 8), ("val", "heldout", 16)):
        names = (data / "ImageSets/2017" / (official + ".txt")).read_text().split()
        names = sorted(names, key=lambda n: hashlib.sha256(("metric-v2-20260914/" + n).encode()).hexdigest())[:count]
        for name in names:
            files = sorted((data / "JPEGImages/480p" / name).glob("*.jpg"))[::3][:COUNT]
            if len(files) < 12:
                raise ValueError("not enough frames: " + name)
            inventory.append({"source_id": "davis/" + name, "split": split, "domain": "public_video",
                "files": [{"path": str(f.resolve()), "sha256": sha256(f)} for f in files],
                "source_frame_indices": [int(f.stem) for f in files], "frames": len(files),
                "timeline": "DAVIS JPEGs have no FPS metadata: every third frame, assigned 8 Hz benchmark playback; not native capture time"})
    for split, seeds in (("development", range(260914100, 260914108)), ("heldout", range(260914900, 260914916))):
        inventory.extend({"source_id": f"renderer/{seed}", "seed": seed, "split": split,
                          "domain": "rendered", "frames": COUNT} for seed in seeds)
    old = json.loads(Path("outputs/automatic_validation_20260912_v1/protocol.json").read_text())
    if any(x["source_id"] in {o["source_id"] for o in old["inventory"]} for x in inventory):
        raise ValueError("old source ID reused")
    previous_seeds = sum(old["synthetic_seeds"].values(), [])
    if any(x.get("seed") in previous_seeds for x in inventory):
        raise ValueError("old renderer seed reused")
    model_root = args.models.resolve()
    protocol = {"schema": "metric-v2-20260914", "created_unix": time.time(), "fps": FPS,
        "resolution": [WIDTH, HEIGHT], "max_frames": COUNT, "inventory": inventory,
        "code_sha256": {f: sha256(Path(__file__).with_name(f)) for f in FILES},
        "models": {"root": str(model_root), "sam2_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=model_root / "sam2", text=True).strip(),
                   "dinov2_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=model_root / "dinov2", text=True).strip(),
                   "sam2_weights_sha256": sha256(model_root / "sam2.1_hiera_tiny.pt"),
                   "dino_weights_sha256": sha256(model_root / "dinov2_vits14_pretrain.pth"),
                   "raft": "torchvision raft_small C_T_V2, 12 updates, FP32",
                   "raft_weights_sha256": sha256(Path.home() / ".cache/torch/hub/checkpoints/raft_small_C_T_V2-01064c6d.pth"),
                   "clip_weights_sha256": sha256(Path.home() / ".cache/clip/ViT-B-32.pt")},
        "human_review": False, "candidate_input_uses_annotations": False,
        "truth": "intervention index maps and independently generated renderer masks; DAVIS existing masks for extractor recall audit only",
        "independence": "new source IDs and renderer seeds relative to previous 2432-case experiment; not a guarantee of pretrained-model training independence",
        "sampling": "no content matching, duplicate removal, or duration rescaling",
        "controls": "identity, brightness+20, gamma .75, JPEG75; renderer fixed geometry and color mean with background/object texture changes",
        "control_scope": "object presence/shape and motion, not color-attribute preservation or all possible semantics",
        "threshold": "strictly greater than development-control 95th percentile, method=higher, by domain and metric",
        "gate": {"auc_min": .9, "tpr_min": .8, "fpr_max": .1, "coverage_min": .95},
        "advantage": "paired source-bootstrap 95% CI of AUC difference >0 against every predeclared baseline; FPR and coverage gates also required",
        "test_policy": "freeze implementation before scores; no formula/threshold retuning after heldout inspection",
        "limitations": ["24 public sources, 24 procedural scenes: pilot scale", "SAM2/DINO segmentation is fallible", "no matched-rate model-quality claims", "novelty remains subject to prior-art comparison"]}
    write(root / "protocol.json", protocol)
    (root / "frozen_source").mkdir()
    for f in FILES:
        shutil.copy2(Path(__file__).with_name(f), root / "frozen_source" / f)
    (root / "inputs").mkdir()
    (root / "cases").mkdir()
    (root / "scores").mkdir()
    count = 0
    with (root / "cases.jsonl").open("w") as out:
        for item in inventory:
            stem = item["source_id"].replace("/", "__")
            if item["domain"] == "rendered":
                source, truth = render_scene(item["seed"])
                np.savez_compressed(root / "inputs" / (stem + "_truth.npz"), **{k:v for k,v in truth.items() if isinstance(v, np.ndarray)})
            else:
                source = np.stack([cv2.resize(cv2.cvtColor(cv2.imread(f["path"]), cv2.COLOR_BGR2RGB),
                                   (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA) for f in item["files"]])
            np.savez_compressed(root / "inputs" / (stem + ".npz"), frames=source)
            for name, rec, truth in variants(source, item.get("seed")):
                cid = stem + "__" + name
                np.savez_compressed(root / "cases" / (cid + ".npz"), frames=rec)
                changed = np.flatnonzero(np.any(source != rec, axis=(1, 2, 3)))
                row = {"case_id": cid, "source_id": item["source_id"], "source_stem": stem,
                    "split": item["split"], "domain": item["domain"], "variant": name, **truth,
                    "frames": len(source), "fps": FPS, "changed_frames": changed.tolist(),
                    "observable": bool(len(changed)) or truth["kind"] == "control",
                    "source_pixel_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
                    "reconstruction_pixel_sha256": hashlib.sha256(rec.tobytes()).hexdigest()}
                out.write(json.dumps(row, allow_nan=False) + "\n")
                count += 1
    write(root / "preparation.json", {"cases": count, "sources": len(inventory),
          "cases_manifest_sha256": sha256(root / "cases.jsonl"), "protocol_sha256": sha256(root / "protocol.json")})
    print("prepared", count, "cases", flush=True)


def _verify(root):
    protocol = json.loads((root / "protocol.json").read_text())
    for f, expected in protocol["code_sha256"].items():
        if sha256(Path(__file__).with_name(f)) != expected:
            raise ValueError("frozen implementation changed: " + f)
    prep = json.loads((root / "preparation.json").read_text())
    if sha256(root / "cases.jsonl") != prep["cases_manifest_sha256"] or sha256(root / "protocol.json") != prep["protocol_sha256"]:
        raise ValueError("frozen protocol or cases changed")
    return protocol


def run_stage(args):
    from .motion_metric import MotionMetric
    root = args.output
    protocol = _verify(root)
    import torch
    torch.manual_seed(20260914)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    rows = [json.loads(x) for x in (root / "cases.jsonl").read_text().splitlines()]
    if args.stage == "object":
        rows = [r for r in rows if r["target"] in ("control", "object")]
        from .object_metric import ObjectMetric
        evaluator = ObjectMetric(protocol["models"]["root"], root / "object_cache", args.device)
    else:
        from .automatic_metrics import AutomaticMetrics
        from .temporal_baselines import TemporalBaselines
        evaluator = MotionMetric(device=args.device)
        baseline = AutomaticMetrics(device=args.device)
        temporal = TemporalBaselines(device=args.device, lpips_model=baseline.lpips)
    directory = root / "scores" / args.stage
    directory.mkdir(exist_ok=True)
    write(root / (args.stage + "_environment.json"), {"python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "opencv": cv2.__version__, "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "started_unix": time.time(), "protocol_sha256": sha256(root / "protocol.json")})
    source, last = None, None
    began = time.time()
    for index, row in enumerate(rows):
        dest = directory / (row["case_id"] + ".json")
        if dest.exists():
            saved = json.loads(dest.read_text())
            if saved["source_pixel_sha256"] != row["source_pixel_sha256"] or saved["reconstruction_pixel_sha256"] != row["reconstruction_pixel_sha256"]:
                raise ValueError("resume pixel mismatch")
            continue
        if last != row["source_id"]:
            with np.load(root / "inputs" / (row["source_stem"] + ".npz")) as f:
                source = f["frames"]
            if args.stage == "motion":
                temporal.cache.clear()
                evaluator.extract.cache.clear()
            last = row["source_id"]
        with np.load(root / "cases" / (row["case_id"] + ".npz")) as f:
            rec = f["frames"]
        if hashlib.sha256(source.tobytes()).hexdigest() != row["source_pixel_sha256"] or hashlib.sha256(rec.tobytes()).hexdigest() != row["reconstruction_pixel_sha256"]:
            raise ValueError("input pixels changed")
        start = time.time()
        scores = evaluator.evaluate(source, rec)
        if args.stage == "motion":
            scores.update(baseline.evaluate(source, rec))
            scores.update(temporal.evaluate(source, rec))
        write(dest, {**row, **scores, "elapsed_s": time.time() - start})
        if index % 8 == 0 or index + 1 == len(rows):
            progress = {"stage": args.stage, "completed": index + 1, "total": len(rows), "elapsed_s": time.time() - began,
                        "last_case": row["case_id"], "status": "RUNNING"}
            write(root / (args.stage + "_progress.json"), progress)
            print(json.dumps(progress), flush=True)
    write(root / (args.stage + "_progress.json"), {"stage": args.stage, "completed": len(rows), "total": len(rows),
          "elapsed_s": time.time() - began, "status": "COMPLETED"})


def _bootstrap_rate(rows, field, draws):
    ids = sorted({r["source_id"] for r in rows})
    sums = np.array([sum(r[field] for r in rows if r["source_id"] == sid) for sid in ids])
    counts = np.array([sum(r["source_id"] == sid for r in rows) for sid in ids])
    index = draws.integers(0, len(ids), size=(500, len(ids)))
    return np.quantile(sums[index].sum(1) / counts[index].sum(1), [.025, .975]).tolist()


def summarize(args):
    root = args.output
    _verify(root)
    merged = {}
    for stage in ("motion", "object"):
        progress = json.loads((root / (stage + "_progress.json")).read_text())
        if progress["status"] != "COMPLETED":
            raise ValueError(stage + " not complete")
        for path in (root / "scores" / stage).glob("*.json"):
            row = json.loads(path.read_text())
            merged.setdefault(row["case_id"], {}).update(row)
    rows = list(merged.values())
    result = {}
    for domain in ("public_video", "rendered"):
        for target in ("motion", "object"):
            group = [r for r in rows if r["domain"] == domain and r["target"] in (target, "control") and r["observable"]]
            if not any(r["target"] == target for r in group):
                continue
            key = domain + "/" + target
            result[key] = {}
            for metric, sign in METRICS.items():
                dev = [sign * r[metric] for r in group if r["split"] == "development" and r["kind"] == "control" and r.get(metric) is not None]
                test = [r for r in group if r["split"] == "heldout"]
                positive = [r for r in test if r["target"] == target]
                negative = [r for r in test if r["kind"] == "control"]
                if not dev or not any(r.get(metric) is not None for r in positive):
                    continue
                threshold = float(np.quantile(dev, .95, method="higher"))
                validneg = [r for r in negative if r.get(metric) is not None]
                negscores = [sign * r[metric] for r in validneg]
                errors = {}
                for kind in sorted({r["kind"] for r in positive}):
                    rr = [r for r in positive if r["kind"] == kind]
                    valid = [r for r in rr if r.get(metric) is not None]
                    scored = [{**r, "hit": float(r.get(metric) is not None and sign * r[metric] > threshold)} for r in rr]
                    errors[kind] = {"cases": len(rr), "coverage": len(valid) / len(rr),
                        "tpr_abstentions_as_misses": float(np.mean([r["hit"] for r in scored])),
                        "tpr_source_bootstrap_95ci": _bootstrap_rate(scored, "hit", np.random.default_rng(20260914)),
                        "auc": auc([sign * r[metric] for r in valid], negscores),
                        "by_severity": {str(s): {"cases": sum(r["severity"] == s for r in scored),
                             "tpr": float(np.mean([r["hit"] for r in scored if r["severity"] == s]))} for s in sorted({r["severity"] for r in scored})}}
                fpr = float(np.mean([s > threshold for s in negscores])) if negscores else None
                coverage = len(validneg) / len(negative)
                passed = bool(fpr is not None and fpr <= .1 and coverage >= .95 and all(
                    e["coverage"] >= .95 and e["tpr_abstentions_as_misses"] >= .8 and e["auc"] is not None and e["auc"] >= .9
                    for e in errors.values()))
                result[key][metric] = {"threshold": threshold, "direction": sign, "development_controls": len(dev),
                    "heldout_fpr": fpr, "control_coverage": coverage, "errors": errors,
                    "heldout_fpr_source_bootstrap_95ci": _bootstrap_rate([{**r,"alarm":float(sign*r[metric]>threshold)} for r in validneg], "alarm", np.random.default_rng(20260914)) if validneg else None,
                    "gate": "PASSED_CONTROLLED_PILOT" if passed else "NOT_PASSED"}
    write(root / "summary.json", {"cases": len(rows), "sources": len({r["source_id"] for r in rows}), "groups": result,
        "claim": "controlled pilot; a gate pass alone does not establish superiority, novelty or real reconstruction semantic accuracy"})
    scalar_keys = sorted(set(METRICS) | {"source_id", "case_id", "domain", "target", "kind", "variant", "split", "severity"})
    with (root / "scores.csv").open("w") as f:
        writer = csv.DictWriter(f, scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: {m: s["gate"] for m,s in v.items() if m in ("mte_tail", "otf_error", "otf_distortion_error")}
                      for k,v in result.items()}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "motion", "object", "summarize"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data/metric_v2_public"))
    p.add_argument("--models", type=Path, default=Path(".local/metric_v2_models"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "summarize":
        summarize(args)
    else:
        try:
            run_stage(args)
        except BaseException as error:
            write(args.output / (args.stage + "_failure.json"), {"error": repr(error), "time": time.time()})
            raise


if __name__ == "__main__":
    main()
