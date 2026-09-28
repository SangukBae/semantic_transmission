"""Completed SKEM imports preserve provenance and reject stale evidence."""
import copy
import csv
import fcntl
from pathlib import Path
import shutil

import pytest

from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.etri_selection_reuse import (
    LEGACY_WORKERS, WORKERS, import_selection, inspect_source,
)
from semantic_transmission.temporal import segment_lengths
from semantic_transmission.webvid5 import fingerprint, read_json
from semantic_transmission.webvid_ablation import snapshot


@pytest.fixture
def saved_run(tmp_path):
    repo, source, target = (tmp_path / n for n in ("repo", "source", "target"))
    repo.mkdir()
    source.mkdir()
    target.mkdir()
    (source / ".lock").touch()
    worker = repo / WORKERS
    worker.parent.mkdir(parents=True)
    actual_repo = Path(__file__).resolve().parents[1]
    worker.write_text((actual_repo / WORKERS).read_text() + "\n# downstream repair\n")
    dependency = repo / "src/semantic_transmission/internvl_memory.py"
    dependency.write_text("# immutable selector dependency\n")
    code = {WORKERS: LEGACY_WORKERS,
            str(dependency.relative_to(repo)): sha256(dependency)}
    execution = {"code_sha256": code, "settings": {"internvl_python": "/env/python"},
                 "environment": {"internvl_environment": {"torch": "frozen"}},
                 "model_file_metadata": {"/model": {"bytes": 100, "mtime_ns": 123}}}
    current = copy.deepcopy(execution)
    current["code_sha256"][WORKERS] = sha256(worker)
    original = tmp_path / "input.mp4"
    original.write_bytes(b"frozen source video fixture")
    cfg = {"input": str(original), "input_sha256": sha256(original), "frames": 1440,
           "max_frames": 1440, "fps": 24, "width": 576, "height": 320,
           "selector": "skem", "selection_stride": 1, "method": "key_frames_test",
           "threshold": .35, "models": {"internvl": "/model"}, "profile": "etri_60s_baseline_v1"}
    protocol = {"version": 1, "config": cfg, "selection": {"processed_sha256": sha256(original)},
                "execution": execution}
    signature = fingerprint(protocol)
    write_json(source / "protocol.json", dict(protocol, signature=signature))
    run = source / "baseline"
    frames = run / "data/frames/sample"
    frames.mkdir(parents=True)
    shutil.copyfile(original, run / "data/normalized.mp4")
    with (frames / "frames.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["frame_path"])
        writer.writeheader()
        for i in range(1440):
            path = frames / f"{i}.png"
            path.write_bytes(f"frame {i}".encode())
            writer.writerow({"frame_path": str(path)})
    write_json(run / "run_config.json", dict(cfg, selector_checkpoint=str(source / "checkpoints/selector.json")))
    write_json(run / "input_audit.json", {"source_sha256": sha256(original), "frames": 1440,
                                        "all_png_pixels_checked": True, "all_pts_checked": True})
    keys = [0, 8, 1439]
    (frames / cfg["method"]).mkdir()
    for i in keys:
        shutil.copyfile(frames / f"{i}.png", frames / cfg["method"] / f"{i}.png")
    write_json(run / "keyframes.json", {"indices": keys, "selector": "skem"})
    write_json(run / "selector_runtime.json", {"command": ["original-selector"]})
    write_json(run / "selector_resources.json", {"original_gpu": 123})
    checkpoint = {"identity": "original-checkpoint-identity", "done": 1440, "selected": ["0", "8"]}
    write_json(source / "checkpoints/selector.json", dict(checkpoint, checksum=fingerprint(checkpoint)))
    write_json(run / "selection_audit.json", {"keyframes": keys, "frames": 1440, "comparisons": 1439,
               "generated_segment_frames_including_overlap": segment_lengths(keys), "automatic_keyframe_insertion": False})
    required = {
        "config": ["baseline/run_config.json"],
        "prepare": ["baseline/data/normalized.mp4", "baseline/data/frames/sample/frames.csv",
                    *[f"baseline/data/frames/sample/{i}.png" for i in range(1440)]],
        "input-audit": ["baseline/input_audit.json"],
        "select": ["baseline/keyframes.json", "baseline/selector_runtime.json", "baseline/selector_resources.json",
                   f"baseline/data/frames/sample/{cfg['method']}", "checkpoints/selector.json"],
        "selection-audit": ["baseline/selection_audit.json"],
    }
    receipts = {}
    for name, paths in required.items():
        receipt = {"status": "PASSED", "identity": signature, "required": paths,
                   "artifacts": snapshot(source, paths), "seconds": 83913 if name == "select" else 1,
                   "dependencies": {n: fingerprint(r) for n, r in receipts.items()}}
        write_json(source / "stages" / f"{name}.json", receipt)
        receipts[name] = receipt
    new_cfg = dict(cfg, profile="etri_60s_baseline_v2", semantic_clip_policy="frame_exact")
    shutil.copytree(run / "data", target / "baseline/data")
    shutil.rmtree(target / "baseline/data/frames/sample" / cfg["method"])
    write_json(target / "baseline/run_config.json", new_cfg)
    shutil.copyfile(run / "input_audit.json", target / "baseline/input_audit.json")
    return repo, source, target, new_cfg, current


def inspect(saved):
    repo, source, _, cfg, execution = saved
    return inspect_source(repo, source, cfg, execution)


def test_import_completed_selection_preserves_original_evidence(saved_run):
    _, source, target, _, _ = saved_run
    before = snapshot(source, ["protocol.json", "stages", "baseline", "checkpoints", ".lock"])
    provenance = inspect(saved_run)
    assert provenance["compatibility"]["worker_compatibility"] == "audited_legacy_entrypoints"
    assert provenance["source_selector_seconds"] == 83913
    import_selection(source, target, provenance)
    assert read_json(target / "checkpoints/selector.json")["identity"] == "original-checkpoint-identity"
    assert read_json(target / "baseline/selection_reuse.json") == provenance
    assert read_json(target / "baseline/keyframes.json")["indices"] == [0, 8, 1439]
    assert before == snapshot(source, ["protocol.json", "stages", "baseline", "checkpoints", ".lock"])


@pytest.mark.parametrize("relative", ["baseline/data/frames/sample/900.png", "baseline/keyframes.json",
                                     "checkpoints/selector.json", "baseline/selector_runtime.json"])
def test_reject_changed_artifact(saved_run, relative):
    saved_run[1].joinpath(relative).write_text("corrupted")
    with pytest.raises(ValueError, match="source artifacts changed"):
        inspect(saved_run)


def test_reject_incomplete_stage(saved_run):
    path = saved_run[1] / "stages/select.json"
    receipt = read_json(path)
    write_json(path, dict(receipt, status="FAILED"))
    with pytest.raises(ValueError, match="source stage not complete"):
        inspect(saved_run)


def test_reject_broken_dependency_chain(saved_run):
    path = saved_run[1] / "stages/select.json"
    receipt = read_json(path)
    write_json(path, dict(receipt, dependencies={}))
    with pytest.raises(ValueError, match="dependency chain"):
        inspect(saved_run)


def test_reject_protocol_tampering(saved_run):
    path = saved_run[1] / "protocol.json"
    protocol = read_json(path)
    protocol["config"]["threshold"] = .1
    write_json(path, protocol)
    with pytest.raises(ValueError, match="protocol signature mismatch"):
        inspect(saved_run)


@pytest.mark.parametrize("field,value", [("threshold", .9), ("frames", 384), ("internvl_8bit", True),
                                         ("seed", 1), ("input", "/wrong/input.mp4")])
def test_reject_selector_contract_changes(saved_run, field, value):
    saved_run[3][field] = value
    with pytest.raises(ValueError, match="selector settings/input differ"):
        inspect(saved_run)


@pytest.mark.parametrize("key", ["model_file_metadata", "settings", "environment"])
def test_reject_environment_model_change(saved_run, key):
    saved_run[4][key]["changed"] = True
    with pytest.raises(ValueError, match=f"{key} changed"):
        inspect(saved_run)


def test_reject_selector_dependency_change(saved_run):
    repo, _, _, _, execution = saved_run
    name = "src/semantic_transmission/internvl_memory.py"
    (repo / name).write_text("# changed behavior\n")
    execution["code_sha256"][name] = sha256(repo / name)
    with pytest.raises(ValueError, match="changed code"):
        inspect(saved_run)


def test_reject_selector_entrypoint_change_inside_workers(saved_run):
    repo, _, _, _, execution = saved_run
    path = repo / WORKERS
    path.write_text(path.read_text().replace('str(cfg["threshold"])', 'str(cfg["threshold"] + 1)'))
    execution["code_sha256"][WORKERS] = sha256(path)
    with pytest.raises(ValueError, match="entrypoints changed"):
        inspect(saved_run)


def test_reject_source_changed_after_inspection(saved_run):
    _, source, target, _, _ = saved_run
    provenance = inspect(saved_run)
    (source / "baseline/data/frames/sample/5.png").write_text("modified")
    with pytest.raises(ValueError, match="source artifacts changed"):
        import_selection(source, target, provenance)


def test_reject_fresh_target_pixel_mismatch(saved_run):
    _, source, target, _, _ = saved_run
    provenance = inspect(saved_run)
    (target / "baseline/data/frames/sample/7.png").write_text("different pixels")
    with pytest.raises(ValueError, match="fresh target frame 7 differs"):
        import_selection(source, target, provenance)


def test_reject_active_source_run(saved_run):
    with (saved_run[1] / ".lock").open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="currently locked"):
            inspect(saved_run)


def test_reject_conflicting_target_selection(saved_run):
    _, source, target, _, _ = saved_run
    provenance = inspect(saved_run)
    (target / "baseline/keyframes.json").write_text("conflict")
    with pytest.raises(ValueError, match="conflicting import"):
        import_selection(source, target, provenance)
