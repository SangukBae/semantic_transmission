"""CSV-safe, failure-propagating driver for the original LGVSC decoder."""

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

from .artifacts import sha256, write_json


def split_rows(csv_path):
    groups = {}
    with Path(csv_path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["path", "text", "flow"]:
            raise ValueError("decoder CSV columns must be path,text,flow")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("malformed decoder CSV row")
            video = Path(row["path"]).parent.name
            if not video or video in {".", ".."}:
                raise ValueError("segment path must identify a video parent directory")
            groups.setdefault(video, []).append(row)
    if not groups:
        raise ValueError("decoder input contains no segments")
    return groups


def run(csv_path, save_dir, method, root_dir, *, decoder, config, python=sys.executable,
        extra_args=(), environment=None):
    csv_path, save_dir = Path(csv_path).resolve(), Path(save_dir).resolve()
    groups = split_rows(csv_path)
    # A pre-existing MP4 must never satisfy this invocation's success check.
    save_dir.mkdir(parents=True, exist_ok=False)
    report_path = save_dir / "decoder_run.json"
    split_dir = save_dir / "inputs"
    split_dir.mkdir(exist_ok=True)
    report = {"status": "RUNNING", "input_sha256": sha256(csv_path), "videos": []}
    write_json(report_path, report)
    try:
        for index, (video, rows) in enumerate(groups.items()):
            path = split_dir / f"{index:05d}.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["path", "text", "flow"])
                writer.writeheader()
                writer.writerows(rows)
            command = [str(python), str(Path(decoder).resolve()), "--save_dir", str(save_dir),
                       "--csv_path", str(path), "--method", method, "--root_path",
                       str(Path(root_dir).resolve()), str(Path(config).resolve()), *extra_args]
            entry = {"video": video, "command": command, "status": "RUNNING"}
            report["videos"].append(entry)
            write_json(report_path, report)
            start = time.monotonic()
            with (save_dir / f"{index:05d}.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        env=environment, check=False)
            entry.update(returncode=result.returncode, seconds=time.monotonic() - start)
            if result.returncode:
                entry["status"] = "FAILED"
                raise subprocess.CalledProcessError(result.returncode, command)
            if not any(save_dir.glob(f"{video}*.mp4")):
                entry["status"] = "FAILED"
                raise RuntimeError(f"decoder returned success but produced no video for {video}")
            entry["status"] = "PASSED"
            write_json(report_path, report)
        report["status"] = "PASSED"
    except BaseException as error:
        report.update(status="FAILED", error=str(error))
        raise
    finally:
        write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_name")
    parser.add_argument("save_name")
    parser.add_argument("method")
    parser.add_argument("root_dir", type=Path)
    parser.add_argument("--config", type=Path)
    args, extra = parser.parse_known_args()
    repo = Path(__file__).resolve().parents[2]
    opensora = Path(os.environ.get("OPENSORA_DIR", repo / ".local/vendor/Open-Sora"))
    config = args.config or opensora / "configs/opensora-v1-2/inference/sample.py"
    if args.method != args.csv_name.removesuffix("_video_paths_text_flow"):
        parser.error("method must match the decoder CSV name")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(opensora) + os.pathsep + environment.get("PYTHONPATH", "")
    run(args.root_dir / "16x24" / (args.csv_name + ".csv"), args.root_dir / args.save_name,
        args.method, args.root_dir,
        decoder=repo / "04_semantic_decoder/scripts/mydemo_new_align_sh.py",
        config=config, extra_args=extra, environment=environment)


if __name__ == "__main__":
    main()
