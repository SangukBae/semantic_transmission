import json
from pathlib import Path

import numpy as np
import pytest

from semantic_transmission.automatic_metrics import transition_error, local_support
from semantic_transmission.controlled_errors import scene, object_errors, temporal_errors
from semantic_transmission.input_contract import video_config
from semantic_transmission.pair_inputs import load_pair
from semantic_transmission.artifacts import sha256


def test_transition_direction_and_undefined_static_state():
    t = np.linspace(0, 1, 12)
    a = np.stack([np.cos(t), np.sin(t)], axis=1)
    assert transition_error(a, a)["rte"] == 0
    assert transition_error(a, a[::-1])["rte"] > 0.9
    assert transition_error(a, np.repeat(a[:1], 12, axis=0))["rte"] == pytest.approx(1)
    static = np.ones((12, 3))
    assert transition_error(static, static)["rte"] is None
    assert transition_error(static, static)["transition_coverage"] == 0
    with pytest.raises(ValueError, match="nonfinite"):
        transition_error(static, static * np.nan)


def test_local_support_keeps_far_spatial_mismatch_visible():
    a = np.eye(9)[None]
    assert local_support(a, a)["lssd"] == 0
    b = a.copy()
    b[:, [0, 8]] = b[:, [8, 0]]
    assert local_support(a, b)["lssd"] > 0
    assert local_support(a, b, radius=2)["lssd"] == 0


def test_renderer_truth_is_separate_and_omission_does_not_change_background():
    original, truth = scene(1000)
    variants = {name: (frames, labels) for name, frames, labels in object_errors(1000)}
    removed, labels = variants["object_omission_0.25"]
    changed = np.flatnonzero(np.any(original != removed, axis=(1, 2, 3)))
    assert len(changed) == 3
    assert all(not labels["renderer_truth"][i]["moving_object"]["visible"] for i in changed)
    assert all(r["moving_object"]["visible"] for r in truth)
    np.testing.assert_array_equal(original[:, :10], removed[:, :10])


def test_temporal_ground_truth_agrees_with_actual_injected_frames():
    frames, _ = scene(1000)
    for _, altered, labels in temporal_errors(frames):
        np.testing.assert_array_equal(altered, frames[labels["source_index_map"]])
        assert labels["observable_pixel_change"]


def test_variable_length_recipe_does_not_mutate_frozen_profile():
    cfg = dict(frames=384, max_frames=384, width=576, height=320, fps=24,
               variable_length=True, preserve_input=True)
    for count in (150, 213, 241, 384):
        source = dict(frames=count, width=576, height=320, fps=24.0)
        assert video_config(cfg, source)["frames"] == count
    assert cfg["frames"] == 384
    for key, value in (("frames", 385), ("frames", 1), ("fps", 25), ("width", 512)):
        source = dict(frames=213, width=576, height=320, fps=24)
        source[key] = value
        with pytest.raises(ValueError):
            video_config(cfg, source)
    with pytest.raises(ValueError):
        video_config(dict(cfg, official_preprocessing=True), dict(frames=213, width=576, height=320, fps=24))


def test_physical_alignment_retains_delay_and_reports_missing_tail(tmp_path):
    import cv2
    def video(path, frames, fps):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 32))
        for v in frames:
            writer.write(np.full((32, 32, 3), v, dtype=np.uint8))
        writer.release()
    src, rec = tmp_path / "source.mp4", tmp_path / "rec.mp4"
    video(src, range(0, 120, 10), 4)
    video(rec, [0, 0, 10, 20, 30, 40, 50, 60], 4)
    row = dict(source=str(src), source_sha256=sha256(src), reconstruction=str(rec), reconstruction_sha256=sha256(rec))
    a, b, info = load_pair(row, count=12, width=32, height=32)
    assert info["missing_sample_count"] == 4
    assert info["sample_coverage"] == pytest.approx(2 / 3)
    assert info["source_indices"] == info["reconstruction_indices"]
    assert np.abs(a.astype(float) - b).mean() > 3
    row["source_sha256"] = "changed"
    with pytest.raises(ValueError, match="SHA256"):
        load_pair(row)


def test_failed_validation_records_failed_and_never_overwrites_an_old_run(tmp_path, monkeypatch):
    import sys
    from semantic_transmission.automatic_validation import main
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["validation", "--output", str(output), "--data-root", str(tmp_path / "missing")])
    with pytest.raises(ValueError, match="no inputs"):
        main()
    status = (output / "progress.json").read_text()
    assert json.loads(status)["status"] == "FAILED"
    with pytest.raises(FileExistsError):
        main()
    assert (output / "progress.json").read_text() == status


def test_common_frame_directory_uses_numeric_order_and_rejects_missing_frames(tmp_path, monkeypatch):
    import hashlib
    from PIL import Image
    from semantic_transmission import pair_inputs
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source byte identity fixture")
    folder = tmp_path / "frames"
    folder.mkdir()
    for i in range(12):
        Image.fromarray(np.full((32, 32, 3), i, dtype=np.uint8)).save(folder / f"frame_{i}.png")
    files = [folder / f"frame_{i}.png" for i in range(12)]
    digest = hashlib.sha256("".join(p.name + sha256(p) for p in files).encode()).hexdigest()
    monkeypatch.setattr(pair_inputs, "sampled_video", lambda *args: (
        np.zeros((3, 32, 32, 3), dtype=np.uint8), np.array([0, 9, 11]), {"frames": 12, "fps": 10}))
    row = dict(source=str(source), source_sha256=sha256(source), reconstruction=str(folder),
               reconstruction_sha256=digest, reconstruction_fps=10)
    _, rec, _ = load_pair(row, count=3, width=32, height=32)
    assert rec[:, 0, 0, 0].tolist() == [0, 9, 11]
    (folder / "frame_5.png").unlink()
    with pytest.raises(ValueError, match="gap"):
        load_pair(row, count=3)


def test_unknown_controls_are_not_counted_as_correct_negatives_in_bootstrap():
    from semantic_transmission.automatic_validation import METRICS, summarize
    def row(sid, split, kind, value):
        return dict(source_id=sid, split=split, domain="real_temporal", kind=kind, severity=0.5,
                    **{name: value for name in METRICS})
    rows = [row("dev", "development", "control", 0.1),
            row("static", "heldout", "control", None), row("moving", "heldout", "control", 0.2),
            row("static", "heldout", "freeze", None), row("moving", "heldout", "freeze", 0.4)]
    result = summarize(rows, bootstrap=50)["real_temporal"]["rte"]
    assert result["control_coverage"] == 0.5
    assert result["heldout_fpr"] == 1
    assert result["source_bootstrap_fpr_95ci"] == [1, 1]
    assert result["errors"]["freeze"]["coverage"] == 0.5
    assert result["errors"]["freeze"]["tpr_abstentions_as_misses"] == 0.5
