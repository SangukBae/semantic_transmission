"""Small, dependency-free artifact and provenance helpers."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=".json-")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def git_state(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def runtime_state(repo, local):
    """Record runtime identities without retaining a CUDA context in the driver."""
    import importlib.metadata
    import platform
    import sys
    import imageio_ffmpeg
    packages = {}
    for name in ("torch", "torchvision", "transformers", "flash-attn", "apex", "colossalai",
                 "diffusers", "accelerate", "peft", "numpy", "av", "moviepy", "imageio-ffmpeg"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    environment = os.environ.copy()
    for name in ("LD_LIBRARY_PATH", "PYTHONPATH"):
        environment.pop(name, None)
    def output(command):
        return subprocess.check_output(command, env=environment, text=True).strip()
    vendors = {}
    for path in sorted((Path(repo) / ".local/vendor").iterdir()):
        if (path / ".git").exists():
            vendors[path.name] = git_state(path)
            diff = subprocess.check_output(["git", "-C", str(path), "diff", "HEAD"])
            vendors[path.name]["tracked_patch_sha256"] = hashlib.sha256(diff).hexdigest()
    selector = None
    if local.get("internvl_python"):
        selector = json.loads(output([local["internvl_python"], "-c",
            "import json,platform,importlib.metadata as m;print(json.dumps(dict(python=platform.python_version(),packages={n:m.version(n) for n in ['torch','torchvision','transformers','flash-attn','numpy','accelerate']})))"]))
    return {"python": platform.python_version(), "executable": sys.executable, "packages": packages,
            "internvl_environment": selector,
            "channel_packages": json.loads(output([local["channel_python"], "-c",
                "import json,importlib.metadata as m;print(json.dumps({n:m.version(n) for n in ['tensorflow','sionna','numpy']}))"])),
            "gpu": output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
            "ffmpeg": output(["ffmpeg", "-version"]).splitlines()[0],
            "moviepy_ffmpeg": output([imageio_ffmpeg.get_ffmpeg_exe(), "-version"]).splitlines()[0],
            "vendors": vendors}
