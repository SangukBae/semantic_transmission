"""CPU contracts and real selector-loop resume tests; no long-video quality claim."""
import ast
import csv
import logging
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import types

import pytest

from semantic_transmission import etri_60s as app
from semantic_transmission.artifacts import write_json
from semantic_transmission.exact_reuse import FrameTensorCache
from semantic_transmission.selection_checkpoint import SelectionCheckpoint


def frame_files(tmp_path, n=8):
    tmp_path.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(n):
        p = tmp_path / f"{i}.png"
        p.write_bytes(str(i).encode())
        frames.append(str(p))
    return frames


def test_selection_state_resumes_latest_reference_and_rejects_changes(tmp_path):
    frames = frame_files(tmp_path / "frames")
    path = tmp_path / "state.json"
    checkpoint = SelectionCheckpoint(path, {"threshold": .35}, frames)
    checkpoint.save(5, ["0", "3"])
    resumed = SelectionCheckpoint(path, {"threshold": .35}, frames)
    assert resumed.done == 5 and resumed.selected[-1] == "3"
    with pytest.raises(ValueError, match="identity/checksum"):
        SelectionCheckpoint(path, {"threshold": .4}, frames)
    Path(frames[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity/checksum"):
        SelectionCheckpoint(path, {"threshold": .35}, frames)


@pytest.mark.parametrize("done,selected", [(0, []), (9, ["0"]), (4, ["0", "5"]), (4, ["1"]), (4, ["0", "3", "2"])])
def test_selection_state_rejects_invalid_progress(tmp_path, done, selected):
    checkpoint = SelectionCheckpoint(tmp_path / "state", {}, frame_files(tmp_path / "frames"))
    with pytest.raises(ValueError):
        checkpoint.save(done, selected)


def test_actual_skem_loop_resumes_without_repeating_completed_comparisons(tmp_path, monkeypatch):
    import torch
    import pandas as pd
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    source = Path(__file__).resolve().parents[1] / "02_semantic_encoder/skem/MLM-keyframe-internvl.py"
    tree = ast.parse(source.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = []
    fail = {"at": None}
    def chat(self, tokenizer, pixels, question, config, **kwargs):
        a, b = [int(n) for n in pixels.tolist()]
        if question == "describe":
            if b == fail["at"]:
                raise InterruptedError("test model interruption")
            return "description", [], None
        calls.append((a, b))
        scores = torch.tensor([[5., -5.]]) if b % 3 == 0 else torch.tensor([[-5., 5.]])
        return "answer", [], [scores]
    class Model:
        device = "cpu"
        def eval(self): return self
        def cuda(self): return self
    model_factory = SimpleNamespace(from_pretrained=lambda *a, **k: Model())
    tokenizer = SimpleNamespace(encode=lambda token, **k: [0 if token == "No" else 1])
    scope = dict(torch=torch, AutoModel=model_factory, AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
        custom_chat=chat, types=types, logging=logging, pd=pd, os=os, shutil=shutil,
        FrameTensorCache=FrameTensorCache, load_image=lambda p, **k: torch.tensor([int(Path(p).stem)]),
        tqdm=lambda **k: SimpleNamespace(update=lambda *a: None), validation_progress=lambda *a: None)
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(source), "exec"), scope)
    frames = frame_files(tmp_path / "frames")
    with (tmp_path / "frames/frames.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=["frame_path"]); w.writeheader()
        w.writerows({"frame_path": p} for p in frames)
    csv_path = tmp_path / "videos.csv"
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=["frame_save_dir"]); w.writeheader()
        w.writerow({"frame_save_dir": str(tmp_path / "frames")})
    args = SimpleNamespace(model_path="fixture", cpu_offload=False, load_in_8bit=False,
        flash_attn=False, cpu_static_head=False, gpu_static_head=False, max_new_tokens=8,
        method="test", threshold=.35, csv_path=str(csv_path), max_tiles=1,
        no_frame_cache=False, q1="describe", q2="compare", resume_state=None)
    scope["main"](args)
    complete = list(calls)
    keys = sorted(p.name for p in (tmp_path / "frames/key_framestest").glob("*.png"))
    shutil.rmtree(tmp_path / "frames/key_framestest")
    calls.clear()
    args.resume_state = str(tmp_path / "checkpoint.json")
    fail["at"] = 4
    with pytest.raises(InterruptedError):
        scope["main"](args)
    assert app.read_json(args.resume_state)["done"] == 4
    assert calls == complete[:3]
    shutil.rmtree(tmp_path / "frames/key_framestest")  # runner archives incomplete outputs
    fail["at"] = None
    scope["main"](args)
    assert calls == complete
    assert sorted(p.name for p in (tmp_path / "frames/key_framestest").glob("*.png")) == keys
    assert app.read_json(args.resume_state)["done"] == 8


def test_full_jobs_cover_tail_and_never_insert_keys():
    jobs = list(app.jobs({"frames": 1440, "method": "key_framesinternvl_diff_0.35"}))
    assert tuple(j[0] for j in jobs) == app.STAGES
    assert "data/frames/sample/1439.png" in jobs[0][3]
    assert len([p for p in jobs[0][3] if p.endswith(".png")]) == 1440
    assert next(j for j in jobs if j[0] == "select")[1] == "semantic_transmission.workers"


def test_stop_after_input_does_not_start_model_or_claim_completion(tmp_path, monkeypatch):
    cfg = dict(frames=1440, seed=2025, method="key_framesinternvl_diff_0.35")
    monkeypatch.setattr(app, "settings", lambda _: {"python": "core", "channel_python": "channel"})
    calls = []
    class Stages:
        def __init__(self, *a): pass
        def step(self, name, owned, required, callback): calls.append(name)
    monkeypatch.setattr(app, "Stages", Stages)
    result = app.execute(tmp_path, tmp_path, cfg, "identity", "input-audit")
    assert calls == ["config", "prepare", "input-audit"]
    assert result["status"] == "STOPPED_AFTER_STAGE" and not result["full_60s_reconstructed"]


def test_status_reports_saved_caption_segments(tmp_path, capsys):
    import json
    write_json(tmp_path / "checkpoints/caption.json", {"records": [{"segment": 0}, {"segment": 1}], "total": 109})
    app.status(tmp_path)
    result = json.loads(capsys.readouterr().out)
    assert result["caption_progress"] == {"done": 2, "total": 109}


def test_imported_selection_never_launches_skem(tmp_path, monkeypatch):
    from semantic_transmission import etri_selection_reuse
    cfg = dict(frames=1440, seed=2025, method="key_framesinternvl_diff_0.35")
    root = tmp_path / "recovered"
    root.mkdir()
    calls, receipts = [], {}
    monkeypatch.setattr(app, "settings", lambda _: {"python": "core", "channel_python": "channel"})
    class Stages:
        def __init__(self, *a): pass
        def step(self, name, owned, required, callback):
            receipts[name] = {"owned": owned, "required": required}
            if name in {"config", "select"}:
                callback(root / f"logs/{name}.log")
    monkeypatch.setattr(app, "Stages", Stages)
    monkeypatch.setattr(app, "run_command", lambda *a: pytest.fail("model worker must not launch"))
    monkeypatch.setattr(etri_selection_reuse, "import_selection", lambda *a: calls.append(a))
    provenance = {"source_root": str(tmp_path / "old")}
    result = app.execute(tmp_path, root, cfg, "new", "selection-audit", provenance)
    assert calls == [(tmp_path / "old", root, provenance)]
    assert "baseline/selection_reuse.json" in receipts["select"]["required"]
    assert "checkpoints/selector.json" not in receipts["select"]["owned"]
    assert app.read_json(root / "resources/select.json")["mode"] == "verified_selection_import"
    assert result["status"] == "STOPPED_AFTER_STAGE"


def test_selection_audit_rejects_missing_tail_and_unfinished_checkpoint(tmp_path):
    cfg = dict(frames=1440, selector_checkpoint=str(tmp_path / "checkpoint.json"))
    write_json(tmp_path / "run_config.json", cfg)
    directory = tmp_path / "data/frames/sample"
    directory.mkdir(parents=True)
    def candidates(n):
        with (directory / "frames.csv").open("w") as f:
            w=csv.DictWriter(f, fieldnames=["frame_path"]); w.writeheader()
            w.writerows({"frame_path": f"{i}.png"} for i in range(n))
    candidates(384)
    with pytest.raises(ValueError, match="every input"):
        app.selection_audit(tmp_path)
    candidates(1440)
    write_json(tmp_path / "keyframes.json", {"indices": [0, 1439]})
    state = {"done": 383, "selected": ["0"]}
    write_json(tmp_path / "checkpoint.json", dict(state, checksum=app.fingerprint(state)))
    with pytest.raises(ValueError, match="checkpoint incomplete"):
        app.selection_audit(tmp_path)


def test_finalize_rejects_short_reconstruction_even_with_passing_quality(tmp_path):
    write_json(tmp_path / "baseline/output_audit.json", {"full_60s_reconstructed": False})
    write_json(tmp_path / "baseline/quality.json", {"status": "PASSED"})
    with pytest.raises(ValueError, match="incomplete"):
        app.finalize(tmp_path, SimpleNamespace(completed={}))
    assert not (tmp_path / "RESULT.json").exists()


def test_full_1440_frame_pipeline_and_resume_with_mock_models(tmp_path, monkeypatch):
    """Real FFmpeg/PNG/time audit, mock model/metric results; never GPU evidence."""
    from semantic_transmission.workers import prepare
    from semantic_transmission.artifacts import sha256
    source = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=24",
                    "-frames:v", "1440", "-c:v", "libx264", "-crf", "0", str(source)], check=True)
    cfg = dict(frames=1440, width=64, height=64, fps=24, seed=2025, input=str(source),
               input_sha256=sha256(source), preserve_input=True, selection_stride=1,
               method="key_framestest", cpu_video_storage=True)
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setattr(app, "settings", lambda _: {"python": "core", "channel_python": "channel"})
    calls = []
    def launch(repo, command, log, env, metrics, progress):
        name = command[4] if "--worker" in command else command[3]
        calls.append(name)
        run = root / "baseline"
        config = app.read_json(run / "run_config.json")
        directory = run / "data/frames/sample"
        keys = [0, 8, 1439]
        if name == "prepare":
            prepare(config, repo, run)
        elif name == "select":
            target = directory / config["method"]; target.mkdir()
            for i in keys: shutil.copyfile(directory/f"{i}.png", target/f"{i}.png")
            write_json(run/"keyframes.json", {"indices": keys})
            for f in ["selector_resources.json", "selector_runtime.json"]: write_json(run/f, {"mock": True})
            checkpoint = SelectionCheckpoint(config["selector_checkpoint"], {}, [str(directory/f"{i}.png") for i in range(1440)])
            checkpoint.save(1440, ["0", "8"])
        elif name == "caption":
            for f in ["captions.json", "caption_sampling.json"]: write_json(run/f, {"mock": True})
            write_json(config["caption_checkpoint"], {"done": 2, "mock": True})
        elif name == "semantic-clips":
            for f in ["semantic_clips_audit.json", "data/clips/mock.json"]: write_json(run/f, {"mock": True})
        elif name == "flow":
            for f in ["metadata_tx.json", "flow_sampling.json"]: write_json(run/f, {"mock": True})
        elif name == "send":
            for f in ["transmitter/mock.json", "sender_accounting.json"]: write_json(run/f, {"mock": True})
        elif name == "channel":
            write_json(run/"received/mock.json", {"mock": True})
            write_json(run/"channel_accounting.json", {"status": "PASSED", "metadata_exact_match": True, "mock": True})
        elif name == "receive":
            for f in ["receiver/frames/mock.json", "receiver/metadata.csv", "receiver_accounting.json"]: write_json(run/f, {"mock": True})
            write_json(run/"receiver/decoder_inputs.json", {"indices": keys, "decoder": {"concatenation_policy": "endpoint_exact"}})
        elif name == "reconstruct":
            folder = run/"receiver/reconstruction"; folder.mkdir()
            shutil.copyfile(source, folder/"sample_0000.mp4")
            images=folder/"sample_0000_frames"; images.mkdir()
            for i in range(1440): shutil.copyfile(directory/f"{i}.png", images/f"{i:05d}.png")
            write_json(run/"receiver/decoder_config.py", {"mock": True})
            write_json(run/"receiver/reference_trace.json", [{"loop": 1, "references_before": [3], "references_after": [4]}])
        elif name == "evaluate":
            write_json(run/"quality.json", {"status": "PASSED", "video": {"frames": 1440},
                "video_sha256": sha256(run/"receiver/reconstruction/sample_0000.mp4"), "delivered_mp4": {"mock": True}})
            for f in ["quality_lossless_frames.csv", "quality_delivered_mp4.csv"]: (run/f).write_text("mock\n")
        else:
            app.worker(name, run)
        log.parent.mkdir(parents=True, exist_ok=True); log.write_text("mock models; real IO\n")
        write_json(metrics, {"mock": True}); metrics.with_suffix(".time.txt").write_text("mock\n")
    monkeypatch.setattr(app, "run_command", launch)
    result = app.execute(tmp_path, root, cfg, "mock-model-test")
    assert result["status"] == "PASS_60S_RECONSTRUCTION"
    assert calls == list(app.STAGES)
    app.execute(tmp_path, root, cfg, "mock-model-test")
    assert calls == list(app.STAGES)  # no model or IO worker repeated
    (root/"baseline/receiver/reconstruction/sample_0000_frames/01439.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifacts changed"):
        app.execute(tmp_path, root, cfg, "mock-model-test")
