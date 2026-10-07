"""Materialize byte-identical historical metric definitions from release data.

Public prose lives in docs/METRICS.md. Frozen experiments retain their original
definition hashes and output filenames; editing prose cannot change those bytes.
"""
import hashlib
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "configs/protocols/legacy_metric_definitions.json"
CACHE = ROOT / ".local/protocol_definitions"


def protocol_path(name):
    records = json.loads(BUNDLE.read_text(encoding="utf-8"))["records"]
    if name not in records or not name.replace("_", "").isalnum():
        raise ValueError(f"unknown protocol definition: {name}")
    record = records[name]
    raw = record["utf8"].encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError(f"protocol definition checksum mismatch: {name}")
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f"{name}-{record['sha256']}.txt"
    if dest.exists():
        if dest.read_bytes() != raw:
            raise ValueError(f"cached protocol definition changed: {name}")
        return dest
    with tempfile.NamedTemporaryFile(dir=CACHE, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
    try:
        temporary.replace(dest)
    finally:
        temporary.unlink(missing_ok=True)
    return dest
