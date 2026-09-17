#!/usr/bin/env python
"""Run in the SGD-JSCC environment; fail on the first bad reconstructed frame."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request", type=Path, required=True)
    args = p.parse_args()
    request = json.loads(args.request.read_text())
    sys.path.insert(0, str(Path(request["sgd_repo"]) / "src"))
    import torch
    from sgdjscc_lab.config import load_config
    from sgdjscc_lab.runtime import build_models
    from sgdjscc_lab.pipelines.infer_pipeline import run_single_image
    from sgdjscc_lab.io import load_image_as_tensor, save_tensor_as_image
    from sgdjscc_lab.utils.preprocessing import prepare_patches, merge_patches
    from sgdjscc_lab.utils.seed import set_global_seed
    cfg = load_config(request["config"])
    set_global_seed(request["seed"])
    models = build_models(cfg, torch.device("cuda:0"))
    root = Path(request["output"])
    root.mkdir(exist_ok=False)
    records = []
    for path in sorted(Path(request["frames"]).glob("*.png")):
        start = time.monotonic()
        patches, meta = prepare_patches(load_image_as_tensor(path))
        outputs = []
        for patch in patches:
            value = run_single_image(patch[None], models, cfg)
            if not torch.isfinite(value).all():
                raise ValueError(f"nonfinite SGD output: {path.name}")
            outputs.append(value.cpu())
        rec = merge_patches(torch.cat(outputs), meta)
        if not torch.isfinite(rec).all():
            raise ValueError("nonfinite merged output")
        save_tensor_as_image(rec, root / path.name)
        records.append({"frame": path.name, "patches": len(patches), "seconds": time.monotonic() - start})
        (root.parent / "worker_progress.json").write_text(json.dumps(records, indent=2))
        print(f"reconstructed {len(records)}: {path.name}", flush=True)
    if not records:
        raise ValueError("no source frames")


if __name__ == "__main__":
    main()
