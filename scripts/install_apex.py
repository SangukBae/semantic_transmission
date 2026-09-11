"""Build the Apex kernels used by the released Open-Sora configuration."""
import os
from pathlib import Path
import subprocess
import sys

import torch

repo = Path(__file__).resolve().parents[1]
source = repo / ".local/vendor/apex"
revision = "4138d31ff0acf4071d1dc001ccb7cd6e00800324"
if not source.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", "24.04.01",
                    "https://github.com/NVIDIA/apex.git", str(source)], check=True)
actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
if actual != revision:
    raise RuntimeError(f"Apex checkout differs from pinned build: {actual}")
environment = os.environ.copy()
environment.update(CUDA_HOME=str(repo / ".local/cuda-build"), TORCH_CUDA_ARCH_LIST="8.9", MAX_JOBS="4")
nvidia = Path(torch.__file__).resolve().parents[1] / "nvidia"
environment["CPATH"] = os.pathsep.join(str(path) for path in sorted(nvidia.glob("*/include")))
subprocess.run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps",
                "--config-settings", "--build-option=--cuda_ext", str(source)], env=environment, check=True)
from apex.normalization import FusedLayerNorm
with torch.inference_mode():
    layer = FusedLayerNorm(1152, eps=1e-6, elementwise_affine=False).cuda().bfloat16()
    values = torch.randn(1, 128, 1152, device="cuda", dtype=torch.bfloat16)
    result = layer(values)
    assert result.shape == values.shape and torch.isfinite(result).all()
print(f"Apex {revision}: BF16 CUDA LayerNorm probe passed")
