#!/usr/bin/env python3
"""Check actual core imports, CUDA operations, and the cuDNN library path."""
import json
from pathlib import Path
import runpy
import sys

import torch
import torchvision
import xformers.ops
import transformers
import colossalai
import compressai

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / ".local/vendor/Open-Sora"))
import opensora.models
import opensora.schedulers
from opensora.utils.inference_utils import apply_mask_strategy

# Actual upstream masking with the local profile must retain both endpoints.
profile = runpy.run_path(str(root / "configs/rtx4080_opensora.py"))
latent = torch.zeros(1, 4, 5, 2, 2)
references = [[torch.ones(4, 1, 2, 2), torch.full((4, 1, 2, 2), 2.0)]]
mask = apply_mask_strategy(latent, references, ["0;0,1,0,-1,1"], 0, align=profile["align"])
assert mask[0].tolist() == [0, 1, 1, 1, 0], "keyframe masks lost an endpoint"
assert (latent[:, :, 0] == 1).all() and (latent[:, :, -1] == 2).all()

if not torch.cuda.is_available():
    raise RuntimeError("The local research profile requires a CUDA GPU")
with torch.inference_mode():
    convolution = torch.nn.Conv2d(3, 8, 3).cuda()
    output = convolution(torch.randn(1, 3, 64, 64, device="cuda"))
    if not torch.isfinite(output).all():
        raise RuntimeError("CUDA/cuDNN probe produced nonfinite output")
torch.cuda.synchronize()
print(json.dumps({"status": "PASSED", "gpu": torch.cuda.get_device_name(),
                  "torch": torch.__version__, "cuda": torch.version.cuda,
                  "cudnn": torch.backends.cudnn.version(),
                  "keyframe_endpoint_mask": "PASSED",
                  "transformers": transformers.__version__}, indent=2))
