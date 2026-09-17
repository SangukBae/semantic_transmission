"""Reuse only completely verified videos in a fresh, provenance-preserving batch."""
import copy
import json
from pathlib import Path
import shutil

from .artifacts import sha256, write_json
from .input_contract import video_config

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
            # A dataset may move disks without changing the source bytes. Validate
            # the historical configuration against its own input manifest.
            resolved = video_config(cfg, source) if cfg.get("variable_length") else cfg
            if json.loads((run / "run_config.json").read_text()) != dict(resolved, input=old_inputs[name]["path"]):
                raise ValueError(f"resume video configuration changed: {run}")
            quality = json.loads((run / "quality.json").read_text())
            if quality["status"] != "PASSED" or quality["source_sha256"] != source["sha256"]:
                raise ValueError(f"unverified reconstruction: {run}")
            if "evaluation_profile" in cfg and quality.get("evaluation_profile") != cfg["evaluation_profile"]:
                raise ValueError(f"resume evaluation profile differs: {run}")
            videos = list((run / "receiver/reconstruction").glob("*.mp4"))
            if len(videos) != 1 or sha256(videos[0]) != quality["video_sha256"]:
                raise ValueError(f"reconstructed video changed: {run}")
            expected_frames = resolved["frames"]
            if cfg.get("decoder_policy") == "official_release":
                from .temporal import output_source_indices, resolve_concatenation_policy
                received = json.loads((run / "receiver/decoder_inputs.json").read_text())
                concatenation = resolve_concatenation_policy("official_release", cfg.get("concatenation_policy"))
                received_concat = resolve_concatenation_policy(received["decoder"].get("policy", "endpoint_exact"),
                                                               received["decoder"].get("concatenation_policy"))
                if concatenation != received_concat:
                    raise ValueError(f"received concatenation policy changed: {run}")
                mapping = output_source_indices(received["indices"], concatenation)
                if quality.get("output_source_indices") != mapping:
                    raise ValueError(f"official concatenation frame mapping changed: {run}")
                expected_frames = len(mapping)
                normalized = run / "data/normalized.mp4"
                if sha256(normalized) != quality.get("reference_sha256"):
                    raise ValueError(f"normalized reference changed: {run}")
            if quality["video"]["frames"] != expected_frames or any(quality["video"][key] != cfg[key] for key in ("width", "height", "fps")):
                raise ValueError(f"reconstruction dimensions changed: {run}")
            frames = videos[0].parent / (videos[0].stem + "_frames")
            if len(list(frames.glob("*.png"))) != expected_frames:
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
            reusable[name] = {"path": run, "record": record, "code": manifest["code"], "source": source}
    return reusable


def copy_completed(item, destination):
    shutil.copytree(item["path"], destination)
    record = copy.deepcopy(item["record"])
    record["reused_from"] = str(item["path"])
    record.setdefault("execution_code", item["code"])
    config_path = Path(destination) / "run_config.json"
    config = json.loads(config_path.read_text())
    if config["input"] != item["source"]["path"]:
        record["source_relocation"] = {"previous_path": config["input"],
                                       "current_path": item["source"]["path"],
                                       "sha256": item["source"]["sha256"]}
        config["input"] = item["source"]["path"]
        write_json(config_path, config)
    return record
