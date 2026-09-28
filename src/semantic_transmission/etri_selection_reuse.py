"""Import completed SKEM evidence without claiming it ran under new code.

The old ETRI protocol recorded whole-file hashes, including workers.py. The
single audited compatibility bridge below permits downstream worker repairs
while retaining the original selector entrypoints. Unknown historical workers
and any changed selector dependencies are rejected.
"""
import ast
import contextlib
import csv
import fcntl
import hashlib
from pathlib import Path
import shutil

from .artifacts import sha256, write_json
from .temporal import segment_lengths
from .webvid5 import fingerprint, read_json
from .webvid_ablation import snapshot


PREFIX = ("config", "prepare", "input-audit", "select", "selection-audit")
WORKERS = "src/semantic_transmission/workers.py"
LEGACY_WORKERS = "2c858adcab3aeeb6397ee298b840fa53aab632da9464062eb9cfef58e97bbbc0"
ENTRYPOINTS = {
    "csv_write": "6ebc6c68eaa4e15614b73faa05f6f804646d842d077f952739aef6c020ef19e9",
    "prepare": "6ad96cfd29acb2462adb8d58d49077508e85040db68311775bf9a6dcbb97ef52",
    "select": "0075d3218b772ab5d062baf8870e369ef1792dad67440d5363b9d802255b4203",
    "main": "53ee2c2076d5d4162d1ea28e454b8fe997e8f01654563a76d4a2d3b75130588a",
}
WORKER_STAGES = {"prepare", "select", "caption", "flow", "ntscc", "decode", "evaluate"}
DOWNSTREAM_CODE = {"src/semantic_transmission/etri_60s.py"}
# Additions have no place in the selector's unchanged import graph.
NEW_DOWNSTREAM_CODE = {
    "src/semantic_transmission/etri_selection_reuse.py",
    "src/semantic_transmission/semantic_clips.py",
    "src/semantic_transmission/caption_checkpoint.py",
}
SELECTOR_FIELDS = (
    "input", "input_sha256", "frames", "max_frames", "width", "height", "fps",
    "preserve_input", "official_preprocessing", "selection_stride", "seed",
    "selector", "selector_environment", "method", "threshold", "skim_keyframes",
    "max_new_tokens", "max_tiles", "flash_attn",
)


def _require(condition, message):
    if not condition:
        raise ValueError(f"SKEM reuse: {message}")


@contextlib.contextmanager
def _source_lock(root):
    # Never create or truncate any file in the preserved source run.
    with (root / ".lock").open("rb") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("SKEM reuse: source run is currently locked") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _safe_path(root, relative):
    relative = Path(relative)
    _require(not relative.is_absolute() and ".." not in relative.parts,
             f"unsafe artifact path: {relative}")
    path = root / relative
    _require(path.resolve().is_relative_to(root), f"artifact escapes source run: {relative}")
    return path


def _receipts(root, signature):
    receipts, artifacts = {}, {}
    for name in PREFIX:
        saved = read_json(root / "stages" / f"{name}.json")
        _require(saved.get("status") == "PASSED", f"source stage not complete: {name}")
        _require(saved["identity"] == signature, f"source stage identity differs: {name}")
        _require(saved["dependencies"] == {n: fingerprint(r) for n, r in receipts.items()},
                 f"source dependency chain changed: {name}")
        for relative in saved["required"]:
            _safe_path(root, relative)
        actual = snapshot(root, saved["required"])
        _require(actual == saved["artifacts"], f"source artifacts changed: {name}")
        for relative, digest in actual.items():
            _safe_path(root, relative)
            _require(relative not in artifacts or artifacts[relative] == digest,
                     f"inconsistent source artifacts: {relative}")
            artifacts[relative] = digest
        receipts[name] = saved
    return receipts, artifacts


def _worker_entrypoints(repo):
    nodes = {n.name: n for n in ast.parse((repo / WORKERS).read_text()).body
             if isinstance(n, ast.FunctionDef)}
    # Adding a downstream stage to the CLI is allowed; changing any existing
    # stage binding or other main() statement is not.
    main = nodes["main"]
    for node in ast.walk(main):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "choices" and isinstance(keyword.value, ast.List):
                    keyword.value.elts = [n for n in keyword.value.elts
                                          if isinstance(n, ast.Constant) and n.value in WORKER_STAGES]
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "functions"
                                                for t in node.targets):
            if isinstance(node.value, ast.Dict):
                pairs = [(k, v) for k, v in zip(node.value.keys, node.value.values)
                         if isinstance(k, ast.Constant) and k.value in WORKER_STAGES]
                node.value.keys, node.value.values = [p[0] for p in pairs], [p[1] for p in pairs]
    return {name: hashlib.sha256(ast.dump(nodes[name], include_attributes=False).encode()).hexdigest()
            for name in ENTRYPOINTS}


def _compatibility(repo, source_cfg, target_cfg, previous, current):
    fields = set(SELECTOR_FIELDS) | {k for c in (source_cfg, target_cfg) for k in c
                                    if k.startswith("internvl_")}
    old_contract = {k: source_cfg.get(k) for k in sorted(fields)}
    new_contract = {k: target_cfg.get(k) for k in sorted(fields)}
    _require(old_contract == new_contract, "selector settings/input differ")
    _require(source_cfg["models"]["internvl"] == target_cfg["models"]["internvl"], "InternVL model differs")
    for key in ("settings", "environment", "model_file_metadata"):
        _require(previous[key] == current[key], f"{key} changed")
    old, new = previous["code_sha256"], current["code_sha256"]
    _require(WORKERS in old and WORKERS in new, "worker provenance missing")
    _require(all(sha256(repo / path) == digest for path, digest in new.items()),
             "current code changed since execution identity was captured")
    changed = sorted(path for path in old if old[path] != new.get(path))
    unexpected = set(changed) - DOWNSTREAM_CODE - {WORKERS}
    _require(not unexpected, f"selector compatibility not certified for changed code: {sorted(unexpected)}")
    additions = set(new) - set(old)
    _require(not (additions - NEW_DOWNSTREAM_CODE),
             f"selector compatibility not certified for added code: {sorted(additions - NEW_DOWNSTREAM_CODE)}")
    if WORKERS in changed:
        _require(old[WORKERS] == LEGACY_WORKERS, "unknown historical worker; cannot certify selector equivalence")
        _require(_worker_entrypoints(repo) == ENTRYPOINTS, "selector worker entrypoints changed")
    return {"selector_contract": new_contract, "internvl_model": source_cfg["models"]["internvl"],
            "changed_downstream_code": changed, "added_downstream_code": sorted(additions),
            "source_execution_sha256": fingerprint(previous), "current_execution_sha256": fingerprint(current),
            "worker_compatibility": "audited_legacy_entrypoints" if WORKERS in changed else "identical_file"}


def _inspect(repo, source_root, target_cfg, current_execution):
    protocol = read_json(source_root / "protocol.json")
    signature = protocol["signature"]
    _require(fingerprint({k: v for k, v in protocol.items() if k != "signature"}) == signature,
             "source protocol signature mismatch")
    cfg = protocol["config"]
    _require(cfg["selector"] == "skem" and cfg["selection_stride"] == 1 and cfg["frames"] == 1440,
             "source must be a completed all-frame 60-second SKEM run")
    _require(not (source_root / "baseline/selection_reuse.json").exists(),
             "chained import is not supported; select the original SKEM run")
    compatibility = _compatibility(repo, cfg, target_cfg, protocol["execution"], current_execution)
    receipts, artifacts = _receipts(source_root, signature)
    run = source_root / "baseline"
    actual_cfg = read_json(run / "run_config.json")
    expected_cfg = dict(cfg, selector_checkpoint=str(source_root / "checkpoints/selector.json"))
    if "caption_checkpoint" in actual_cfg:
        expected_cfg["caption_checkpoint"] = str(source_root / "checkpoints/caption.json")
    _require(actual_cfg == expected_cfg,
             "source run configuration differs from protocol")
    input_hash = sha256(cfg["input"])
    _require(input_hash == cfg["input_sha256"] == protocol["selection"]["processed_sha256"],
             "original input hash changed")
    _require(sha256(run / "data/normalized.mp4") == input_hash, "normalized source differs")
    _require(sha256(target_cfg["input"]) == input_hash, "target source differs")
    frames = run / "data/frames/sample"
    with (frames / "frames.csv").open() as stream:
        candidates = [r["frame_path"] for r in csv.DictReader(stream)]
    expected_paths = [str(frames / f"{i}.png") for i in range(cfg["frames"])]
    _require(candidates == expected_paths, "source candidate frames are incomplete or reordered")
    _require({p.name for p in frames.glob("*.png")} == {f"{i}.png" for i in range(cfg["frames"])},
             "source PNG set is incomplete or has extra frames")
    _require(all(str(Path(p).relative_to(source_root)) in artifacts for p in candidates),
             "source frame hashes missing from receipts")
    keys_record = read_json(run / "keyframes.json")
    keys = keys_record["indices"]
    _require(keys_record["selector"] == "skem" and keys == sorted(set(keys)) and keys[0] == 0
             and keys[-1] == cfg["frames"] - 1 and all(type(k) is int and 0 <= k < cfg["frames"] for k in keys),
             "invalid keyframe sequence")
    key_dir = frames / cfg["method"]
    _require({p.name for p in key_dir.iterdir()} == {f"{i}.png" for i in keys}, "keyframe PNG set differs")
    for i in keys:
        _require(sha256(key_dir / f"{i}.png") == artifacts[f"baseline/data/frames/sample/{i}.png"],
                 f"keyframe pixels differ from source frame {i}")
    checkpoint = read_json(source_root / "checkpoints/selector.json")
    checksum = checkpoint.pop("checksum")
    _require(checksum == fingerprint(checkpoint) and checkpoint["done"] == cfg["frames"],
             "selector checkpoint is corrupt or incomplete")
    selected = [int(n) for n in checkpoint["selected"]]
    _require(selected == sorted(set(selected)) and selected[0] == 0
             and all(0 <= n < cfg["frames"] for n in selected)
             and sorted(set(selected + [cfg["frames"] - 1])) == keys,
             "checkpoint keyframes differ")
    audit = read_json(run / "selection_audit.json")
    _require(audit["keyframes"] == keys and audit["frames"] == cfg["frames"]
             and audit["comparisons"] == cfg["frames"] - 1
             and audit["generated_segment_frames_including_overlap"] == segment_lengths(keys)
             and audit["automatic_keyframe_insertion"] is False, "source selection audit differs")
    input_audit = read_json(run / "input_audit.json")
    _require(input_audit["source_sha256"] == input_hash and input_audit["frames"] == cfg["frames"]
             and input_audit["all_png_pixels_checked"] and input_audit["all_pts_checked"],
             "source input audit is incomplete")
    copy_files = ["baseline/keyframes.json", "baseline/selector_resources.json", "baseline/selector_runtime.json",
                  "checkpoints/selector.json", *[f"baseline/data/frames/sample/{cfg['method']}/{i}.png" for i in keys]]
    _require(all(p in artifacts for p in copy_files), "import artifacts missing from passed receipts")
    return {"version": 1, "kind": "verified_completed_skem_import", "source_root": str(source_root),
            "source_signature": signature, "source_protocol_sha256": sha256(source_root / "protocol.json"),
            "source_receipt_sha256": {name: sha256(source_root / "stages" / f"{name}.json") for name in PREFIX},
            "source_artifacts": artifacts, "copied_artifacts": {p: artifacts[p] for p in copy_files},
            "source_selector_seconds": receipts["select"]["seconds"], "source_checkpoint_identity": checkpoint["identity"],
            "keyframes": keys, "method": cfg["method"], "frames": cfg["frames"], "input_sha256": input_hash,
            "compatibility": compatibility,
            "meaning": "Selection computed in source run; imported files retain original command, timing and checkpoint identity."}


def inspect_source(repo, source_root, target_cfg, current_execution):
    """Read-only validation before creating a target protocol or starting models."""
    root = Path(source_root).resolve(strict=True)
    with _source_lock(root):
        return _inspect(Path(repo).resolve(), root, target_cfg, current_execution)


def import_selection(source_root, target_root, provenance):
    """Copy verified completed selection after fresh target prepare/input-audit.

    Caller records a new select-stage receipt and new import timing. Original
    selector runtime/resources are deliberately retained and labelled as reused.
    """
    source, target = Path(source_root).resolve(strict=True), Path(target_root).resolve(strict=True)
    _require(str(source) == provenance["source_root"] and source != target
             and not target.is_relative_to(source) and not source.is_relative_to(target), "invalid import paths")
    with _source_lock(source):
        _require(sha256(source / "protocol.json") == provenance["source_protocol_sha256"], "source protocol changed after inspection")
        for name, digest in provenance["source_receipt_sha256"].items():
            _require(sha256(source / "stages" / f"{name}.json") == digest, "source receipt changed after inspection")
        _, artifacts = _receipts(source, provenance["source_signature"])
        _require(artifacts == provenance["source_artifacts"], "source artifacts changed after inspection")
        target_cfg = read_json(target / "baseline/run_config.json")
        target_audit = read_json(target / "baseline/input_audit.json")
        _require(target_cfg["frames"] == provenance["frames"] and target_cfg["method"] == provenance["method"]
                 and target_audit["source_sha256"] == provenance["input_sha256"], "target input/config changed")
        for i in range(provenance["frames"]):
            relative = f"baseline/data/frames/sample/{i}.png"
            _require(sha256(target / relative) == artifacts[relative], f"fresh target frame {i} differs")
        for relative, digest in provenance["copied_artifacts"].items():
            origin = _safe_path(source, relative)
            destination = _safe_path(target, relative)
            _require(sha256(origin) == digest, f"source changed before copy: {relative}")
            if destination.exists():
                _require(sha256(destination) == digest, f"refusing to overwrite conflicting import: {relative}")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(origin, destination)
            _require(sha256(destination) == digest, f"copied artifact differs: {relative}")
        write_json(target / "baseline/selection_reuse.json", provenance)
    return provenance
