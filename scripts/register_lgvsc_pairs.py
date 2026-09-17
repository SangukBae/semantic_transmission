#!/usr/bin/env python3
"""Export verified completed LGVSC runs to the common evaluation manifest."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from semantic_transmission.artifacts import sha256
from semantic_transmission.automatic_validation import write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.batch.resolve()
    manifest = json.loads((root / "batch_manifest.json").read_text())
    if manifest["status"] != "PASSED":
        raise ValueError("LGVSC batch is not complete")
    pairs = []
    for record in manifest["runs"]:
        run = root / record["id"]
        if record["status"] != "PASSED":
            raise ValueError("LGVSC run is not complete")
        cfg = json.loads((run / "run_config.json").read_text())
        quality = json.loads((run / "quality.json").read_text())
        videos = list((run / "receiver/reconstruction").glob("*.mp4"))
        if len(videos) != 1 or quality["status"] != "PASSED":
            raise ValueError("LGVSC quality evidence missing")
        source, rec = Path(cfg["input"]), videos[0]
        if sha256(source) != quality["source_sha256"] or sha256(rec) != quality["video_sha256"]:
            raise ValueError("LGVSC source/reconstruction bytes changed")
        pairs.append({"source_id": source.stem, "source": str(source), "source_sha256": sha256(source),
                      "reconstruction": str(rec), "reconstruction_sha256": sha256(rec),
                      "model": cfg["profile"], "provenance": str(run / "run_manifest.json")})
    if not pairs:
        raise ValueError("empty LGVSC batch")
    write(args.output, {"schema": "source-reconstruction-pairs-v1", "pairs": pairs})


if __name__ == "__main__":
    main()
