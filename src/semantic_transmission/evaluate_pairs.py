"""Evaluate registered reconstructions from any model on the same source timeline."""
import argparse
import json
from pathlib import Path

from .automatic_metrics import AutomaticMetrics
from .automatic_validation import write
from .pair_inputs import load_pair
from .artifacts import sha256
from .temporal_baselines import TemporalBaselines


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, required=True, action="append")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=12)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pairs = []
    for path in args.pairs:
        manifest = json.loads(path.read_text())
        if manifest.get("schema") != "source-reconstruction-pairs-v1" or not manifest["pairs"]:
            raise ValueError("invalid or empty common pair manifest")
        pairs.extend(manifest["pairs"])
    evaluator = AutomaticMetrics()
    temporal = TemporalBaselines(evaluator.device, evaluator.lpips)
    results = []
    for row in pairs:
        source, rec, alignment = load_pair(row, args.frames)
        values = evaluator.evaluate(source, rec)
        temporal.cache.clear()
        values.update(temporal.evaluate(source, rec))
        # Locate evidence for subsequent investigation without declaring true errors.
        windows = sorted(((i, v) for i, v in enumerate(values["rte_transitions"]) if v is not None), key=lambda x: x[1], reverse=True)[:3]
        values["candidate_windows"] = [{"sample_interval": [i, i + 1], "rte": v,
              "source_frames": alignment["source_indices"][i:i + 2], "confirmed_semantic_error": None} for i, v in windows]
        results.append(dict(row, alignment=alignment, scores=values, semantic_ground_truth=None))
    environment = {"torch": evaluator.torch.__version__, "cuda": evaluator.torch.version.cuda,
                   "clip_compute_dtype": str(evaluator.clip.visual.conv1.weight.dtype),
                   "clip_checkpoint_sha256": sha256(Path.home() / ".cache/clip/ViT-B-32.pt"),
                   "metric_code_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                       ("automatic_metrics.py", "temporal_baselines.py", "pair_inputs.py")},
                   "score_resolution": [224, 128], "sample_intervals": "relative to the first decoded frame; not per-second motion scores"}
    write(args.output, {"status": "COMPLETED", "scope": "source-paired diagnostics; no human or detector labels treated as truth",
                         "environment": environment,
                         "frame_samples_requested": args.frames, "results": results})


if __name__ == "__main__":
    main()
