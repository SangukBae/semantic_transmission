#!/usr/bin/env python3
"""Download and verify the public NTSCC and UniMatch checkpoints."""
import hashlib
from pathlib import Path
import urllib.request

import gdown


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1] / ".local/checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    specs = [
        ("ntscc_hyperprior_quality_4_psnr.pth", "1ab9tCYGLoJZo0xPXfu73Roms2u910Vb7",
         "b264cffde6e8a530d2a41b1794c0da8a4a9ac716a94861869de970444ba44799"),
        ("unimatch.pth", "https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth",
         "4e7b215d5a25dc3b41bd2fb55cbbfe1b715a9c279bdcc8ad30bc31afe69421ca"),
    ]
    for name, source, expected in specs:
        path = root / name
        if not path.exists():
            temporary = path.with_suffix(".partial")
            if source.startswith("https:"):
                urllib.request.urlretrieve(source, temporary)
            else:
                gdown.download(id=source, output=str(temporary), quiet=False)
            if digest(temporary) != expected:
                raise ValueError(f"checksum mismatch: {name}")
            temporary.replace(path)
        if digest(path) != expected:
            raise ValueError(f"existing checkpoint checksum mismatch: {name}")
        print(f"Verified: {name}")


if __name__ == "__main__":
    main()
