#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
base="$repo/.local/bsu_codec_eval_env/bin/python"
env_dir="$repo/.local/framer_env"
vendor="$repo/.local/framer_vendor"
revision=d0ba81f1f23a639396d4c6a801c53a5b21412b06
unset PYTHONPATH LD_LIBRARY_PATH HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTHONNOUSERSITE=1 HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=30
if [[ ! -d "$vendor/.git" ]]; then
  git clone --quiet https://github.com/aim-uofa/Framer.git "$vendor"
fi
git -C "$vendor" checkout --quiet --detach "$revision"
[[ "$(git -C "$vendor" rev-parse HEAD)" == "$revision" ]]
PYTHONPATH="$repo/src" python3 -m semantic_transmission.framer_compat "$vendor" --apply
if [[ ! -x "$env_dir/bin/python" ]]; then
  "$base" -m venv "$env_dir"
fi
env_site=$("$env_dir/bin/python" -c 'import site; print(site.getsitepackages()[0])')
"$base" -c 'import sys; print("\n".join(p for p in sys.path if p.endswith("site-packages")))' > "$env_site/readonly_codec_dependencies.pth"
"$env_dir/bin/python" -m pip install --disable-pip-version-check --no-deps 'diffusers==0.24.0'
"$env_dir/bin/python" - "$repo" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(sys.argv[1]) / ".local/framer_weights"
root.mkdir(exist_ok=True)
models = [
    ("wwen1997/framer_512x320", "023bd84da79faada5f27f3bafe7d8e5c8120a61f",
     "framer", ["*.json", "*.safetensors"]),
    ("stabilityai/stable-video-diffusion-img2vid-xt",
     "9e43909513c6714f1bc78bcb44d96e733cd242aa", "svd",
     ["*.json", "image_encoder/model.fp16.safetensors", "vae/diffusion_pytorch_model.fp16.safetensors"]),
]
manifest = {}
for model, revision, name, patterns in models:
    print(f"Downloading pinned public assets: {model}", flush=True)
    snapshot_download(model, revision=revision, local_dir=root / name,
                      allow_patterns=patterns, max_workers=2)
    hashes = {}
    for path in sorted((root / name).rglob("*")):
        if path.is_file() and ".cache" not in path.parts:
            h = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    h.update(block)
            hashes[str(path.relative_to(root / name))] = h.hexdigest()
    manifest[name] = dict(model_id=model, revision=revision, files=hashes)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("SETUP_COMPLETE", flush=True)
PY
