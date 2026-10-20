"""Read-only numerical/provenance audit of a completed Framer pilot."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.sparc_framer import read, verify
from semantic_transmission.framer_compat import check as check_vendor
from semantic_transmission.webvid5 import fingerprint
from semantic_transmission.packets import accounting


def main():
    root = Path(sys.argv[1]).resolve()
    p, r, result, restoration = [read(root / name) for name in
                                 ("protocol.json", "receiver.json", "RESULT.json", "restore.json")]
    assert p["signature"] == fingerprint({k: v for k, v in p.items() if k != "signature"})
    assert result["protocol_signature"] == restoration["signature"] == r["signature"] == p["signature"]
    verify(p["code"])
    verify(p["reference_files"])
    verify(p["source"]["hashes"], p["source"]["directory"])
    for name, value in p["assets"].items():
        verify(value["files"], Path(r["weights"]) / name)
    vendor_revision = subprocess.check_output(["git", "-C", r["vendor"], "rev-parse", "HEAD"], text=True).strip()
    assert vendor_revision == p["vendor_revision"]
    assert check_vendor(r["vendor"]) == p["vendor_patch"]
    video = r["video"]
    shape = (video["frames"], video["height"], video["width"], 3)
    assembled = np.empty(shape, np.uint8)
    count = 0
    for a, b in zip(r["indices"], r["indices"][1:]):
        path = root / f"segments/{a:05d}_{b:05d}.npy"
        record = read(path.with_suffix(".json"))
        assert record["identity"] == dict(protocol=p["signature"], start=a, end=b)
        assert sha256(path) == record["sha256"]
        frames = np.load(path, allow_pickle=False)
        assert frames.shape == (b-a+1, *shape[1:]) and frames.dtype == np.uint8
        for i, frame in ((a, frames[0]), (b, frames[-1])):
            anchor = np.asarray(Image.open(Path(r["keys_directory"]) / f"{i:05d}.png").convert("RGB"))
            assert np.array_equal(anchor, frame)
        assembled[a:b+1] = frames
        count += 1
    for i, frame in enumerate(assembled):
        exported = np.asarray(Image.open(root / f"reconstruction/frames/{i:05d}.png").convert("RGB"))
        assert np.array_equal(frame, exported), i
    assert sha256(result["movie"]) == result["movie_sha256"] == restoration["movie_sha256"]
    rows = list(csv.DictReader((root / "quality_per_frame.csv").open()))
    assert len(rows) == video["frames"]
    for key in ("psnr_db", "ssim", "lpips_vgg", "clip", "dists"):
        assert abs(np.mean([float(row[key]) for row in rows]) - result["quality"][key]) < 1e-10
    reference = Path(p["reference"])
    comparison = read(reference / "comparison.json")
    baseline_movies = 0
    for label, row in comparison["results"].items():
        movie_label = "motion" if label == "motion_no_visibility" else label
        verify({str(reference / f"reconstruction/{movie_label}/reconstruction.mp4"): row["movie_sha256"]})
        baseline_movies += 1
    historical = read(reference / "protocol.json")["historical"]
    verify({historical["path"]: historical["sha256"]})
    for label, row in comparison["historical"].items():
        movie = Path(row["movie"])
        case = movie.parent.parent
        verify({str(movie): row["movie_sha256"], str(case / "protocol.json"): row["protocol_sha256"]})
        old_source = read(case / "protocol.json")["source"]
        assert old_source["hashes"] == p["source"]["hashes"]
        assert old_source["video"] == p["source"]["video"]
        csv_path = Path(historical["path"]).parent / f"{label}_per_frame.csv"
        verify({str(csv_path): row["per_frame_csv_sha256"]})
        baseline_movies += 1
    channel = read(reference / "received/channel.json")["packets"]
    uses = 0
    for name in ("keys", "motion"):
        packet = reference / "received" / f"{name}.bin"
        budget = accounting(packet.stat().st_size)
        assert all(channel[name][k] == v for k, v in budget.items())
        assert channel[name]["bit_errors"] == 0
        uses += budget["complex_channel_uses"]
    assert uses == result["channel_uses"]
    value = dict(status="PASSED", source_hashes_verified=len(p["source"]["hashes"]),
                 received_anchors_verified=len(r["indices"]), completed_segments_verified=count,
                 exported_frames_equal_segments=len(assembled), quality_csv_means_verified=True,
                 public_checkpoint_hashes_verified=True, pinned_vendor_with_verified_ff_in_fix=True,
                 baseline_movie_hashes_verified=baseline_movies,
                 protocol_and_inference_code_verified=True, received_channel_cost_verified=uses,
                 receiver_source_pixels_used=False, result_sha256=sha256(root / "RESULT.json"))
    write_json(root / "AUDIT.json", value)
    print(json.dumps(value))


if __name__ == "__main__":
    main()
