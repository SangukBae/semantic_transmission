"""Reuse only completely verified videos in a fresh, provenance-preserving batch."""
import copy
import json
from pathlib import Path
import shutil

from .artifacts import sha256

STAGES = ["prepare", "select", "caption", "flow", "send", "channel", "receive", "reconstruct", "evaluate"]


def completed_runs(previous_roots, cfg, sources):
    reusable = {}
    inputs = {s["id"]: s for s in sources}
    for previous in previous_roots:
        previous = Path(previous).resolve()
        manifest_path = previous / "batch_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        if json.loads((previous / "profile.json").read_text()) != cfg:
            raise ValueError(f"resume profile/model paths differ: {previous}")
        old_inputs = {s["id"]: s for s in manifest["inputs"]}
        for candidate in manifest["runs"]:
            name = candidate["id"]
            if name not in inputs or name in reusable or candidate["status"] != "PASSED":
                continue
            run = previous / name
            record = json.loads((run / "run_manifest.json").read_text())
            if record["status"] != "PASSED" or [s["stage"] for s in record["stages"]] != STAGES:
                raise ValueError(f"incomplete successful-run record: {run}")
            if any(s["status"] != "PASSED" or s["returncode"] != 0 for s in record["stages"]):
                raise ValueError(f"failed stage cannot be reused: {run}")
            source = inputs[name]
            if old_inputs[name]["sha256"] != source["sha256"]:
                raise ValueError(f"resume source changed: {name}")
            if json.loads((run / "run_config.json").read_text()) != dict(cfg, input=source["path"]):
                raise ValueError(f"resume video configuration changed: {run}")
            quality = json.loads((run / "quality.json").read_text())
            if quality["status"] != "PASSED" or quality["source_sha256"] != source["sha256"]:
                raise ValueError(f"unverified reconstruction: {run}")
            videos = list((run / "receiver/reconstruction").glob("*.mp4"))
            if len(videos) != 1 or sha256(videos[0]) != quality["video_sha256"]:
                raise ValueError(f"reconstructed video changed: {run}")
            if any(quality["video"][key] != cfg[key] for key in ("width", "height", "frames", "fps")):
                raise ValueError(f"reconstruction dimensions changed: {run}")
            frames = videos[0].parent / (videos[0].stem + "_frames")
            if len(list(frames.glob("*.png"))) != cfg["frames"]:
                raise ValueError(f"reconstruction frames missing: {run}")
            sender = json.loads((run / "sender_accounting.json").read_text())
            if set(sender["transmitter_files"]) != {"metadata.bin", "visual.c64"}:
                raise ValueError(f"transmitter files missing: {run}")
            for filename, info in sender["transmitter_files"].items():
                path = run / "transmitter" / filename
                if path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
                    raise ValueError(f"transmitter payload changed: {run}")
            for filename in ("channel_accounting.json", "receiver_accounting.json"):
                if json.loads((run / filename).read_text())["status"] != "PASSED":
                    raise ValueError(f"transport did not pass: {run}")
            reusable[name] = {"path": run, "record": record, "code": manifest["code"]}
    return reusable


def copy_completed(item, destination):
    shutil.copytree(item["path"], destination)
    record = copy.deepcopy(item["record"])
    record["reused_from"] = str(item["path"])
    record.setdefault("execution_code", item["code"])
    return record
