#!/usr/bin/env python3
"""Repair Decord 0.6.0's incorrect internal tag, preserving all binary/code bytes.

The upstream filename is py3-none-manylinux2010_x86_64, but WHEEL incorrectly
declares cp36-cp36m. Decord loads libdecord through ctypes, not a CPython extension.
Modern pip check rejects this contradictory metadata even when decoding works.
"""
import base64
import csv
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import zipfile


def main():
    root = Path(__file__).resolve().parents[1] / ".local/wheels"
    original = root / "upstream"
    original.mkdir(parents=True, exist_ok=True)
    filename = "decord-0.6.0-py3-none-manylinux2010_x86_64.whl"
    subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "decord==0.6.0", "-d", str(original)], check=True)
    with zipfile.ZipFile(original / filename) as source:
        contents = {name: source.read(name) for name in source.namelist()}
    name = "decord-0.6.0.dist-info/WHEEL"
    old, new = b"Tag: cp36-cp36m-manylinux2010_x86_64", b"Tag: py3-none-manylinux2010_x86_64"
    if old not in contents[name] and new not in contents[name]:
        raise ValueError("unrecognized Decord wheel metadata; refusing modification")
    contents[name] = contents[name].replace(old, new)
    record = "decord-0.6.0.dist-info/RECORD"
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    for path, data in contents.items():
        if path != record:
            encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            writer.writerow([path, "sha256=" + encoded, len(data)])
    writer.writerow([record, "", ""])
    contents[record] = stream.getvalue().encode()
    target = root / filename
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for path, data in contents.items():
            wheel.writestr(path, data)
    subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(target)], check=True)
    print("Repaired Decord wheel tag and RECORD; model and native code bytes unchanged.")


if __name__ == "__main__":
    main()
