"""Synthetic CPU tests; these do not claim real WebVid reconstruction quality."""
import copy
import csv
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from semantic_transmission import webvid_ablation as app
from semantic_transmission import webvid_ablation_report as report
from semantic_transmission.ablation_transport import densify, preserve_transport, subdivide_metadata
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.temporal import output_source_indices, unique_output_positions
from semantic_transmission.wire import pack, unpack


def test_density_preserves_non_etri_endpoints_and_interval_metadata():
    original = [0, 53, 88, 336]
    dense = densify(original, 24)
    assert set(original).issubset(dense)
    assert dense[-1] == 336 and max(b - a for a, b in zip(dense, dense[1:])) <= 24
    assert len(dense) == len(set(dense))
    rows = [dict(path=f"old/{i}", text=f"caption {i}, with comma", flow=i + .5) for i in range(3)]
    result = subdivide_metadata(original, dense, rows)
    for (a, b), row in zip(zip(dense, dense[1:]), result):
        i = next(i for i, (s, t) in enumerate(zip(original, original[1:])) if s <= a < b <= t)
        assert row["text"] == rows[i]["text"] and row["flow"] == rows[i]["flow"]
    assert original == [0, 53, 88, 336]
    with pytest.raises(ValueError):
        subdivide_metadata([0, 5, 10], [0, 10], rows[:2])
    assert densify([0, 1, 2], 24) == [0, 1, 2]


def test_stage_failure_resume_preserves_partials_and_does_not_repeat_selection(tmp_path):
    calls = []
    def select(log):
        calls.append("select")
        (tmp_path / "keys").write_text("completed expensive selection")
    def fail(log):
        (tmp_path / "caption").write_text("partial")
        raise RuntimeError("model interrupted")
    first = app.Stages(tmp_path, "same protocol")
    first.step("select", ["keys"], ["keys"], select)
    with pytest.raises(RuntimeError):
        first.step("caption", ["caption"], ["caption"], fail)
    second = app.Stages(tmp_path, "same protocol")
    second.step("select", ["keys"], ["keys"], select)
    second.step("caption", ["caption"], ["caption"], lambda _: (tmp_path / "caption").write_text("complete"))
    assert calls == ["select"]
    assert next((tmp_path / "failed_attempts").glob("caption_*/caption")).read_text() == "partial"
    assert app.read_json(tmp_path / "stages/caption.json")["status"] == "PASSED"
    (tmp_path / "keys").write_text("tampered")
    with pytest.raises(ValueError, match="artifacts changed"):
        app.Stages(tmp_path, "same protocol").step("select", ["keys"], ["keys"], select)
    assert calls == ["select"]


def test_resume_rejects_changed_identity_and_dependency(tmp_path):
    first = app.Stages(tmp_path, "v1")
    first.step("a", ["a"], ["a"], lambda _: (tmp_path / "a").write_text("a"))
    with pytest.raises(ValueError, match="identity/dependencies"):
        app.Stages(tmp_path, "v2").step("a", ["a"], ["a"], lambda _: pytest.fail("no execution"))
    fresh = app.Stages(tmp_path, "v1")
    fresh.completed["unrelated"] = {"changed": True}
    with pytest.raises(ValueError, match="identity/dependencies"):
        fresh.step("a", ["a"], ["a"], lambda _: pytest.fail("no execution"))


def test_duplicate_lock_fails(tmp_path):
    path = tmp_path / "lock"
    with app.exclusive_lock(path):
        with pytest.raises(RuntimeError, match="실행 중"):
            with app.exclusive_lock(path):
                pytest.fail("must not acquire twice")


@pytest.fixture
def selected_repo(tmp_path, monkeypatch):
    repo = tmp_path
    source_repo = Path(__file__).resolve().parents[1]
    cohort = app.read_json(source_repo / "configs/webvid5_manifest.json")
    chosen = next(v for v in cohort["videos"] if v["category"] == "single_subject")
    for kind in ("raw", "processed"):
        path = repo / "dataset" / kind / chosen["filename"]
        path.parent.mkdir(parents=True)
        path.write_bytes(kind.encode())
        chosen[f"{kind}_sha256"] = sha256(path)
    cfg = app.read_json(source_repo / "configs/webvid5.json")
    write_json(repo / "configs/webvid5.json", cfg)
    cohort["profile_sha256"] = sha256(repo / "configs/webvid5.json")
    write_json(repo / "configs/webvid5_manifest.json", cohort)
    write_json(repo / ".local/datasets.json", {"webvid55": str(repo / "dataset")})
    write_json(repo / ".local/model_paths.json", {})
    (repo / "configs/official_opensora.py").write_text("# fixture")
    monkeypatch.setattr(app, "probe", lambda _: {k: chosen[k] for k in ("width", "height", "frames", "fps")})
    monkeypatch.setattr(app, "doctor", lambda _: {"status": "PASSED"})
    monkeypatch.setattr(app, "execution_identity", lambda *args: {"fixture": True})
    return repo, chosen


def test_dry_run_selects_only_one_full_video_without_writes(selected_repo, monkeypatch):
    repo, chosen = selected_repo
    before = {str(p): sha256(p) for p in repo.rglob("*") if p.is_file()}
    monkeypatch.setattr(app, "execute", lambda *args: pytest.fail("dry-run must not infer"))
    selection, cfg = app.load_selection(repo)
    assert selection == chosen and cfg["frames"] == 337 and cfg["selection_stride"] == 1
    root = app.run(repo, dry_run=True)
    assert not root.exists()
    assert before == {str(p): sha256(p) for p in repo.rglob("*") if p.is_file()}
    # Other four videos are deliberately absent; the one-video command does
    # not depend on them or silently substitute another clip.
    (repo / "dataset/processed" / chosen["filename"]).write_text("changed")
    with pytest.raises(ValueError, match="source hash mismatch"):
        app.run(repo, dry_run=True)


def make_packet(run, indices, cfg, rows):
    tx = run / "transmitter"
    tx.mkdir()
    items = [dict(index=i, complex_offset=j * 2, complex_count=2, average_power=1.,
                  rate_offset=j, rate_bytes=1, rate_count=2) for j, i in enumerate(indices)]
    header = {"keyframes": items, "segments": rows,
              "video": {k: cfg[k] for k in ("frames", "width", "height", "fps")},
              "decoder": {"seed": cfg["seed"], "steps": cfg["steps"], "policy": "official_release",
                          "concatenation_policy": cfg["concatenation_policy"]}}
    (tx / "metadata.bin").write_bytes(pack(header, bytes(range(len(indices)))))
    np.arange(len(indices) * 2, dtype=np.float32).astype("<c8").tofile(tx / "visual.c64")
    write_json(run / "sender_accounting.json", {"transmitter_files": {
        p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in tx.iterdir()}})


def fake_channel(run):
    shutil.copytree(run / "transmitter", run / "received")
    n = (run / "received/visual.c64").stat().st_size // 8
    write_json(run / "channel_accounting.json", dict(status="PASSED", metadata_exact_match=True, bit_errors=0,
               visual_complex_channel_uses=n, digital_complex_channel_uses=64, total_complex_channel_uses=n + 64,
               received_files={p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in (run / "received").iterdir()}))


def fake_receive(run):
    header, _ = unpack((run / "received/metadata.bin").read_bytes())
    directory = run / "receiver/frames/sample/key_frames_received"
    directory.mkdir(parents=True)
    indices = [k["index"] for k in header["keyframes"]]
    for i in indices:
        shutil.copyfile(run / f"data/frames/sample/{i}.png", directory / f"{i}.png")
    with (run / "receiver/metadata.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "text", "flow"])
        writer.writeheader(); writer.writerows(header["segments"])
    write_json(run / "receiver/decoder_inputs.json", {"indices": indices, "decoder": header["decoder"], "video": header["video"]})
    write_json(run / "receiver_accounting.json", {"status": "PASSED"})


def test_paired_channel_retains_original_blocks_and_reproducible_added_noise(tmp_path):
    cfg = dict(frames=9, width=64, height=64, fps=4, seed=42, steps=30, snr_db=10, channel_seed=1024,
               concatenation_policy="endpoint_exact")
    rows = [dict(path="clips/sample/00000.mp4", text="caption", flow=.5)]
    base, run = tmp_path / "baseline", tmp_path / "dense"
    for folder, keys in ((base, [0, 8]), (run, [0, 4, 8])):
        folder.mkdir()
        write_json(folder / "run_config.json", cfg)
        make_packet(folder, keys, cfg, rows)
        if folder == base:
            fake_channel(folder)
    preserve_transport(run, base)
    fake_channel(run)
    preserve_transport(run, base, received=True)
    report.audit_wire(run, base)
    first = (run / "received/visual.c64").read_bytes()
    preserve_transport(run, base, received=True)
    assert first == (run / "received/visual.c64").read_bytes()
    values = np.fromfile(run / "received/visual.c64", dtype="<c8")
    sent = np.fromfile(run / "transmitter/visual.c64", dtype="<c8")
    assert not np.array_equal(values[2:4], sent[2:4])
    assert app.read_json(run / "channel_accounting.json")["total_complex_channel_uses"] == 70


def test_full_orchestration_and_real_ffmpeg_report_resume(tmp_path, monkeypatch):
    """Model stages are synthetic; report generation uses real FFmpeg and probes."""
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required for report contract")
    from semantic_transmission.workers import prepare
    source = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=4",
                    "-frames:v", "9", "-c:v", "libx264", "-threads", "1", str(source)], check=True)
    cfg = dict(input=str(source), frames=9, width=64, height=64, fps=4, seed=42, steps=30, snr_db=10,
               channel_seed=1024, models={}, preserve_input=True, selection_stride=1, method="keys_fixture",
               concatenation_policy="official_release")
    root = tmp_path / "result"
    root.mkdir()
    write_json(root / "protocol.json", {"signature": "fixture", "config": cfg,
        "selection": {"filename": "synthetic_test_fixture.mp4", "processed_sha256": sha256(source), "review_focus": "fixture only"}})
    monkeypatch.setattr(app, "settings", lambda _: {"python": "fake-core", "channel_python": "fake-channel"})
    calls = []
    def fake_command(repo, args, log, env):
        args = list(map(str, args))
        module = args[2]
        stage = args[args.index("--worker") + 1] if "--worker" in args else args[3]
        run = Path(args[args.index("--run-dir") + 1] if "--worker" in args else args[4])
        calls.append((run.name, stage))
        log.parent.mkdir(exist_ok=True); log.write_text("synthetic model fixture\n")
        config = app.read_json(run / "run_config.json")
        if stage == "prepare":
            prepare(config, repo, run)
        elif stage == "select":
            keys = [0, 5, 8]
            write_json(run / "keyframes.json", {"indices": keys})
            write_json(run / "selector_runtime.json", {"synthetic": True})
            (run / "data/frames/sample/keys_fixture").mkdir()
            for i in keys:
                shutil.copyfile(run / f"data/frames/sample/{i}.png", run / f"data/frames/sample/keys_fixture/{i}.png")
        elif stage == "caption":
            write_json(run / "captions.json", [dict(path=f"clips/sample/{i:05d}.mp4", text=f"auto {i}", flow=.5) for i in range(2)])
            write_json(run / "caption_sampling.json", {"synthetic": True})
            (run / "data/clips").mkdir(); (run / "data/clips/fixture").write_text("clip")
        elif stage == "flow":
            shutil.copyfile(run / "captions.json", run / "metadata_tx.json")
            write_json(run / "flow_sampling.json", {"synthetic": True})
        elif stage == "send":
            make_packet(run, app.read_json(run / "keyframes.json")["indices"], config, app.read_json(run / "metadata_tx.json"))
            if run.name == "dense_1s":
                preserve_transport(run, root / "baseline")
        elif stage == "channel":
            assert env["CUDA_VISIBLE_DEVICES"] == "-1"
            fake_channel(run)
            if run.name == "dense_1s":
                preserve_transport(run, root / "baseline", received=True)
        elif stage == "receive":
            fake_receive(run)
        elif stage == "reconstruct":
            inputs = app.read_json(run / "receiver/decoder_inputs.json")
            mapping = output_source_indices(inputs["indices"], config["concatenation_policy"])
            frames = run / "receiver/reconstruction/sample_0000_frames"
            frames.mkdir(parents=True)
            for j, i in enumerate(mapping):
                shutil.copyfile(run / f"data/frames/sample/{i}.png", frames / f"{j:05d}.png")
            subprocess.run(["ffmpeg", "-v", "error", "-framerate", "4", "-i", str(frames / "%05d.png"),
                            "-c:v", "libx264", "-threads", "1", str(frames.parent / "sample_0000.mp4")], check=True)
            (run / "receiver/decoder_config.py").write_text("# synthetic test model\n")
        elif stage == "evaluate":
            dest = Path(args[args.index("--output-dir") + 1]); dest.mkdir(parents=True)
            inputs = app.read_json(run / "receiver/decoder_inputs.json")
            positions = unique_output_positions(inputs["indices"], config["concatenation_policy"])
            metrics = dict.fromkeys(report.KEYS, .5); metrics["frames"] = 9
            write_json(dest / "quality.json", dict(status="PASSED", evaluation_profile="lgvsc_official_metrics_v1",
                source_sha256=sha256(source), reference_sha256=sha256(source),
                video_sha256=sha256(run / "receiver/reconstruction/sample_0000.mp4"),
                endpoint_exact_view=dict(kept_generated_indices=positions, output_source_indices=list(range(9)),
                                         lossless_frames=metrics, delivered_mp4=metrics)))
        elif stage == "report":
            report.build(root)
        else:
            pytest.fail(f"unexpected stage {module}/{stage}")
    monkeypatch.setattr(app, "command", fake_command)
    app.execute(tmp_path, root, cfg, "fixture")
    assert app.read_json(root / "COMPLETE.json")["status"] == "PASSED"
    assert sum(stage == "select" for _, stage in calls) == 1
    assert sum(stage == "reconstruct" for _, stage in calls) == 4
    summary = app.read_json(root / "comparison_summary.json")
    assert [r["case"] for r in summary["rows"]] == list(app.CASES)
    assert summary["rows"][2]["total_complex_channel_uses"] is None
    assert summary["manual_semantic_review"] == "PENDING"
    assert (root / "comparison_four_panel.mp4").stat().st_size > 0
    assert not any(p.name.startswith("__") for p in root.iterdir())
    before = len(calls)
    app.execute(tmp_path, root, cfg, "fixture")
    assert len(calls) == before  # report/metrics/model stages all reused
    # Missing one metric must prevent a final comparison, rather than treating
    # pixel-only results as a complete evaluation.
    path = root / "evaluations/aligned/quality.json"
    quality = app.read_json(path)
    del quality["endpoint_exact_view"]["lossless_frames"]["dists"]
    write_json(path, quality)
    with pytest.raises((ValueError, KeyError)):
        report.collect(root)
    with pytest.raises(ValueError, match="artifacts changed"):
        app.execute(tmp_path, root, cfg, "fixture")


def test_decoder_adapter_keeps_official_recipe_and_sets_alignment(tmp_path, monkeypatch):
    from semantic_transmission import codec_transport, decoder_runner
    repo = Path(__file__).resolve().parents[1]
    for case in app.CASES:
        run = tmp_path / case
        cfg = dict(models={k: f"/fixture/{k}" for k in ("stdit", "vae", "vae2d", "t5")}, flash_attn=True)
        inputs = dict(video=dict(height=320, width=576, fps=24), decoder=dict(seed=42, steps=30, policy="official_release",
                      concatenation_policy="official_release" if case == "baseline" else "endpoint_exact"))
        write_json(run / "run_config.json", cfg)
        write_json(run / "receiver/decoder_inputs.json", inputs)
        monkeypatch.setattr(app, "repository", lambda: repo)
        monkeypatch.setattr(decoder_runner, "run", lambda *args, **kwargs: None)
        app.worker("reconstruct", run)
        namespace = {}
        exec((run / "receiver/decoder_config.py").read_text(), namespace)
        assert namespace["conditioning_alignment"] == ("official_release" if case == "baseline" else "endpoint_exact")
        assert namespace["vae"]["micro_batch_size"] == 4
        assert namespace["decoder_policy"] == "official_release"
        assert namespace["cache_text_embeddings"] == (case == "dense_1s")
