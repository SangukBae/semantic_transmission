"""Run source-held-out, human-free controlled validation and real-pair diagnostics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from scipy.stats import rankdata, spearmanr

from .artifacts import sha256
from .automatic_metrics import AutomaticMetrics
from .controlled_errors import controls, temporal_errors, object_errors, scene
from .pair_inputs import sampled_video, load_pair

METRICS = {"rte": 1, "lssd": 1, "psnr_db": -1, "ssim": -1, "lpips_alex": 1, "clip_cosine": -1,
           "temporal_pixel_mae": 1}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def auc(pos, neg):
    if not pos or not neg:
        return None
    ranks = rankdata(pos + neg)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def summarize(rows, bootstrap=500):
    result = {}
    for domain in ("real_temporal", "rendered"):
        group = [r for r in rows if r["domain"] == domain and r.get("observable_pixel_change", True)]
        metrics = {}
        for metric, direction in METRICS.items():
            dev = [direction * r[metric] for r in group if r["split"] == "development" and r["kind"] == "control" and r[metric] is not None]
            if not dev:
                continue
            threshold = float(np.quantile(dev, 0.95, method="higher"))
            test = [r for r in group if r["split"] == "heldout"]
            negative = [direction * r[metric] for r in test if r["kind"] == "control" and r[metric] is not None]
            by_kind = {}
            for kind in sorted({r["kind"] for r in test} - {"control"}):
                positive_rows = [r for r in test if r["kind"] == kind]
                valid = [r for r in positive_rows if r[metric] is not None]
                positive = [direction * r[metric] for r in valid]
                correlations = []
                for sid in sorted({r["source_id"] for r in valid}):
                    rr = [r for r in valid if r["source_id"] == sid]
                    if len(rr) >= 3 and np.ptp([r[metric] for r in rr]) > 1e-9:
                        correlations.append(float(spearmanr([r["severity"] for r in rr], [direction * r[metric] for r in rr]).statistic))
                by_kind[kind] = {"auc": auc(positive, negative), "tpr": float(np.mean(np.array(positive) > threshold)) if positive else None,
                    "coverage": len(valid) / len(positive_rows), "cases": len(positive_rows),
                    "tpr_abstentions_as_misses": sum(v > threshold for v in positive) / len(positive_rows),
                    "median_severity_spearman": float(np.median(correlations)) if correlations else None,
                    "severity_evaluable_sources": len(correlations)}
            # Resample entire source clusters, retaining all variants of each source.
            ids = sorted({r["source_id"] for r in test})
            negative_alarms, negative_counts, positive_rates = [], [], []
            for sid in ids:
                sr = [r for r in test if r["source_id"] == sid]
                neg = [r for r in sr if r["kind"] == "control"]
                pos = [r for r in sr if r["kind"] != "control"]
                measured_neg = [r for r in neg if r[metric] is not None]
                negative_alarms.append(sum(direction * r[metric] > threshold for r in measured_neg))
                negative_counts.append(len(measured_neg))
                positive_rates.append(np.mean([r[metric] is not None and direction * r[metric] > threshold for r in pos]))
            rng = np.random.default_rng(20260912)
            draws = rng.integers(0, len(ids), size=(bootstrap, len(ids)))
            def ci(values):
                return np.quantile(np.asarray(values)[draws].mean(axis=1), [0.025, 0.975]).tolist()
            denominators = np.asarray(negative_counts)[draws].sum(axis=1)
            available = denominators > 0
            negative_ci = (np.quantile(np.asarray(negative_alarms)[draws].sum(axis=1)[available] /
                           denominators[available], [0.025, 0.975]).tolist() if available.any() else None)
            metrics[metric] = {"direction": direction, "threshold_oriented": threshold, "development_controls": len(dev),
                "heldout_sources": len(ids), "heldout_fpr": float(np.mean(np.array(negative) > threshold)) if negative else None,
                "control_coverage": len(negative) / sum(r["kind"] == "control" for r in test),
                "source_bootstrap_fpr_95ci": negative_ci, "source_bootstrap_tpr_95ci_abstentions_as_misses": ci(positive_rates),
                "errors": by_kind}
        result[domain] = metrics
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pairs", type=Path)
    p.add_argument("--frames", type=int, default=12)
    p.add_argument("--synthetic-dev", type=int, default=16)
    p.add_argument("--synthetic-test", type=int, default=32)
    p.add_argument("--limit-real", type=int)
    args = p.parse_args()
    if args.frames < 8 or min(args.synthetic_dev, args.synthetic_test) < 1:
        raise ValueError("validation needs >=8 frames and nonempty independent scene splits")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    try:
        run(args, root)
    except BaseException as error:
        progress_path = root / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        progress.update(status="FAILED", error=str(error))
        write(progress_path, progress)
        raise


def run(args, root):
    repo = Path(__file__).resolve().parents[2]
    inventory = []
    for dataset, split in (("etri_video_eval", "development"), ("webvid55", "heldout"), ("kinetics400_14", "heldout")):
        paths = sorted((args.data_root / dataset / "processed").glob("*.mp4"))
        if args.limit_real:
            paths = paths[:args.limit_real]
        if not paths:
            raise ValueError(f"no inputs in {dataset}")
        inventory.extend({"source_id": dataset + "/" + f.stem, "path": str(f.resolve()), "split": split,
                          "sha256": sha256(f)} for f in paths)
    if len({r["sha256"] for r in inventory}) != len(inventory):
        raise ValueError("duplicate source bytes across validation inventory")
    protocol = {"schema": "automatic-validation-v1", "human_annotations_used": False, "frames_sampled": args.frames,
        "score_resolution": [224, 128], "source_sampling": "uniform over full source duration including endpoints",
        "definition_sha256": {f.name: sha256(f) for f in [Path(__file__), Path(__file__).with_name("automatic_metrics.py"), Path(__file__).with_name("controlled_errors.py")]},
        "inventory": inventory, "synthetic_seeds": {"development": list(range(1000, 1000 + args.synthetic_dev)),
            "heldout": list(range(9000, 9000 + args.synthetic_test))},
        "threshold": "95th percentile (higher) of development nuisance-control scores; strictly greater means alarm",
        "frozen_candidate_gate": {"heldout_auc_min": 0.9, "heldout_tpr_min": 0.8, "heldout_fpr_max": 0.1,
             "coverage_min": 0.95, "median_severity_spearman_min": 0.5},
        "claim_scope": "controlled sampled-frame interventions and renderer objects; real reconstructions have no semantic truth",
        "clip_checkpoint_sha256": sha256(Path.home() / ".cache/clip/ViT-B-32.pt"),
        "python": platform.python_version(), "started_unix": time.time()}
    write(root / "protocol.json", protocol)  # Freeze before any score is observed.
    for filename in protocol["definition_sha256"]:
        (root / "frozen_source").mkdir(exist_ok=True)
        (root / "frozen_source" / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    subprocess.run(["git", "diff", "--binary"], cwd=repo, stdout=(root / "tracked_diff.patch").open("w"), check=True)
    evaluator = AutomaticMetrics()
    import torch, clip, lpips
    write(root / "environment.json", {"torch": torch.__version__, "numpy": np.__version__, "gpu": torch.cuda.get_device_name(),
          "clip_code_sha256": sha256(Path(clip.__file__).with_name("model.py")),
          "lpips_weights_sha256": sha256(Path(lpips.__file__).parent / "weights/v0.1/alex.pth")})
    rows = []
    progress = {"status": "RUNNING", "completed_sources": 0, "expected_sources": len(inventory) + args.synthetic_dev + args.synthetic_test}
    with (root / "cases.jsonl").open("w") as stream:
        def evaluate_source(meta, frames, variants):
            for case_id, rec, truth in variants:
                values = evaluator.evaluate(frames, rec)
                row = dict(meta, case_id=case_id, **truth, **values)
                row["source_pixel_sha256"] = hashlib.sha256(frames.tobytes()).hexdigest()
                row["reconstruction_pixel_sha256"] = hashlib.sha256(rec.tobytes()).hexdigest()
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                if progress["completed_sources"] in (0, len(inventory)) and case_id in ("identity", "reverse_1.0", "object_addition_1.0", "object_omission_1.0"):
                    np.savez_compressed(root / f"example_{meta['domain']}_{case_id}.npz", source=frames, reconstructed=rec)
            progress["completed_sources"] += 1
            write(root / "progress.json", progress)
            print(f"{progress['completed_sources']}/{progress['expected_sources']} sources; {len(rows)} cases; {meta['source_id']}", flush=True)
        from itertools import chain
        for item in inventory:
            frames, indices, info = sampled_video(item["path"], args.frames)
            meta = {"source_id": item["source_id"], "split": item["split"], "domain": "real_temporal",
                    "source_sha256": item["sha256"], "sampled_source_indices": indices.tolist(), "source_fps": info["fps"]}
            evaluate_source(meta, frames, chain(controls(frames), temporal_errors(frames)))
        for split, seeds in protocol["synthetic_seeds"].items():
            for seed in seeds:
                frames, truth = scene(seed, args.frames)
                write(root / "renderer_truth" / f"{seed}.json", truth)
                evaluate_source({"source_id": f"renderer/{seed}", "split": split, "domain": "rendered", "seed": seed},
                                frames, chain(controls(frames), temporal_errors(frames), object_errors(seed, args.frames)))
    stats = summarize(rows)
    gates = {}
    for metric, domain, kinds in (("rte", "real_temporal", ("reverse", "freeze", "lag", "swap")),
                                  ("lssd", "rendered", ("object_addition", "object_omission", "object_shape"))):
        if metric not in stats[domain]:
            gates[metric] = {"status": "NOT_PASSED", "failures": ["no measurable development controls for calibration"]}
            continue
        s = stats[domain][metric]
        failures = []
        if s["heldout_fpr"] is None or s["heldout_fpr"] > 0.1:
            failures.append("heldout control FPR > 0.1 or unavailable")
        for kind in kinds:
            e = s["errors"][kind]
            for field, minimum in (("auc", 0.9), ("tpr", 0.8), ("coverage", 0.95), ("median_severity_spearman", 0.5)):
                if e[field] is None or e[field] < minimum:
                    failures.append(f"{kind}: {field} < {minimum} or unavailable")
        gates[metric] = {"status": "PASSED_CONTROLLED_GATE" if not failures else "NOT_PASSED", "failures": failures}
    real = []
    if args.pairs:
        for row in json.loads(args.pairs.read_text())["pairs"]:
            a, b, provenance = load_pair(row, args.frames)
            real.append(dict(row, alignment=provenance, **evaluator.evaluate(a, b), semantic_error_truth=None,
                             interpretation="diagnostic only; no model ranking at matched channel/rate"))
    write(root / "real_pairs.json", real)
    summary = {"status": "COMPLETED", "cases": len(rows), "sources": progress["expected_sources"], "statistics": stats,
               "candidate_gates": gates, "real_pairs": len(real), "elapsed_s": time.time() - protocol["started_unix"]}
    write(root / "summary.json", summary)
    flat = [{k: v for k, v in r.items() if isinstance(v, (str, float, int, bool)) or v is None} for r in rows]
    with (root / "scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in flat))))
        writer.writeheader(); writer.writerows(flat)
    progress["status"] = "COMPLETED"
    write(root / "progress.json", progress)
    print(json.dumps({k: v for k, v in summary.items() if k != "statistics"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
