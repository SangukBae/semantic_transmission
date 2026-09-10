#!/usr/bin/env python3
"""Install exact external source revisions without vendoring them into Git."""
import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    sources = json.loads((root / "configs/upstreams.json").read_text())
    for name, source in sources.items():
        path = root / ".local/vendor" / name
        if not path.exists():
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", source["url"], str(path)], check=True)
            subprocess.run(["git", "-C", str(path), "checkout", "--detach", source["commit"]], check=True)
        actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if actual != source["commit"]:
            raise RuntimeError(f"{name} has revision {actual}; expected {source['commit']}; preserving checkout")
        patches = {"NTSCC_JSAC22": "03_jscc_transmission/ntscc/patches/ntscc_modifications.patch",
                   "PLLaVA": "patches/pllava-local-attention.patch",
                   "Open-Sora": "patches/opensora-local-vae.patch"}
        if name in patches:
            patch = root / patches[name]
            reverse = subprocess.run(["git", "-C", str(path), "apply", "--reverse", "--check", str(patch)], capture_output=True)
            if reverse.returncode:
                subprocess.run(["git", "-C", str(path), "apply", "--check", str(patch)], check=True)
                subprocess.run(["git", "-C", str(path), "apply", str(patch)], check=True)
        print(f"Ready: {name} @ {actual}", flush=True)


if __name__ == "__main__":
    main()
