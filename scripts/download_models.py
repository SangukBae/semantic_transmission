#!/usr/bin/env python3
"""Fetch pinned, public model files into the shared Hugging Face cache."""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["internvl", "stdit", "vae", "t5", "pllava", "vae2d"])
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    specs = json.loads((repo / "configs/models.json").read_text())
    output = repo / ".local/model_paths.json"
    paths = json.loads(output.read_text()) if output.exists() else {}
    for name in args.models:
        spec = specs[name]
        print(f"Downloading {name}: {spec['repo']} @ {spec['revision']}", flush=True)
        paths[name] = snapshot_download(repo_id=spec["repo"], revision=spec["revision"],
                                       allow_patterns=spec.get("allow_patterns", ["*.json", "*.safetensors", "*.bin", "*.model", "*.py"]),
                                       max_workers=3)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(paths, indent=2) + "\n")
        temporary.replace(output)
        print(f"Ready: {name}", flush=True)


if __name__ == "__main__":
    main()
