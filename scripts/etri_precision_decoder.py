"""Tail17 decoder with prepared T5 conditions and scoped, verified noise."""
import importlib.metadata
import os
from pathlib import Path
import runpy

from semantic_transmission import generation_noise as noise, precision_text_cache as text
from semantic_transmission.webvid5 import read_json


def main():
    import torch
    repo = Path(__file__).resolve().parents[1]
    run = Path(os.environ["ETRI_PRECISION_RUN"])
    policy = read_json(run / "receiver_policy.json")
    text.install(run / text.REPORT, run / text.TRACE, repo)
    expected = read_json(policy["noise_reference"]) if policy.get("noise_reference") else None
    runtime = dict(packages={n: importlib.metadata.version(n) for n in ("torch", "diffusers", "transformers")},
                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
    control = noise.NoiseControl(run / noise.TRACE, policy["noise_contract"], runtime, expected)
    noise.install(control)
    runpy.run_path(str(repo / "scripts/etri_tail17_decoder.py"), run_name="__main__")
    control.finish(policy["noise_contract"]["segments"], policy["noise_contract"]["steps"])


if __name__ == "__main__":
    main()
