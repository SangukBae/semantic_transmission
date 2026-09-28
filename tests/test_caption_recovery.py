"""CPU regressions for exact clips and durable caption progress."""
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from semantic_transmission import workers
from semantic_transmission.artifacts import write_json
from semantic_transmission.caption_checkpoint import CaptionCheckpoint
from semantic_transmission.semantic_clips import prepare_frame_exact_clips, sample_clip_frames


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path):
    run = tmp_path / "run"
    frames = run / "data/frames/sample"
    frames.mkdir(parents=True)
    normalized = run / "data/normalized.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
        "testsrc2=size=64x48:rate=24", "-frames:v", "10", "-c:v", "libx264", "-crf", "0",
        "-threads", "1", "-pix_fmt", "yuv420p", str(normalized)], check=True)
    capture = cv2.VideoCapture(str(normalized))
    for i in range(10):
        ok, image = capture.read()
        assert ok
        cv2.imwrite(str(frames / f"{i}.png"), image)
    capture.release()
    write_json(run / "keyframes.json", {"indices": [0, 1, 5, 9]})
    model = tmp_path / "fake_model"
    model.mkdir()
    write_json(model / "config.json", {"unit_test": True})
    cfg = {"frames": 10, "fps": 24, "official_semantic_clips": True,
           "semantic_clip_policy": "frame_exact", "seed": 2025, "max_new_tokens": 30,
           "models": {"pllava": str(model)},
           "caption_checkpoint": str(tmp_path / "checkpoints/caption.json")}
    return run, cfg


def test_exact_clips_cover_single_frame_and_half_open_intervals(prepared):
    run, cfg = prepared
    paths = prepare_frame_exact_clips(cfg, run)
    audit = json.loads((run / "semantic_clips_audit.json").read_text())
    assert [row["frames"] for row in audit["clips"]] == [1, 4, 4]
    assert all(row["pts_verified"] and row["pixels_exact"] for row in audit["clips"])
    hashes = [row["sha256"] for row in audit["clips"]]
    assert prepare_frame_exact_clips(cfg, run) == paths
    assert hashes == [row["sha256"] for row in json.loads((run / "semantic_clips_audit.json").read_text())["clips"]]


def test_one_frame_works_in_actual_caption_reader_and_flow_sampler(prepared):
    run, cfg = prepared
    clip = prepare_frame_exact_clips(cfg, run)[0]
    sys.path.insert(0, str(REPO / ".local/vendor/Open-Sora"))
    from opensora.datasets.read_video import read_video_av
    decoded, _, _ = read_video_av(str(clip), pts_unit="sec", output_format="THWC")
    assert len(decoded) == 1
    extracted, actual_indices = sample_clip_frames(clip, [0, 10, 20, 30])
    assert actual_indices == [0, 0, 0, 0]
    source = cv2.cvtColor(cv2.imread(str(run / "data/frames/sample/0.png")), cv2.COLOR_BGR2RGB)
    assert len(extracted) == 4
    assert all(np.array_equal(np.asarray(image), source) for image in extracted)
    assert np.array_equal(decoded[0].numpy(), source)


def test_clip_and_source_corruption_are_refused(prepared):
    run, cfg = prepared
    paths = prepare_frame_exact_clips(cfg, run)
    original = paths[0].read_bytes()
    paths[0].write_bytes(b"not a video")
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        prepare_frame_exact_clips(cfg, run)
    paths[0].write_bytes(original)
    frame = run / "data/frames/sample/1.png"
    frame.write_bytes((run / "data/frames/sample/9.png").read_bytes())
    with pytest.raises(ValueError, match="identity/checksum"):
        prepare_frame_exact_clips(cfg, run)


def test_wrong_length_clip_rejected_before_model_loading(prepared, monkeypatch):
    run, cfg = prepared
    paths = prepare_frame_exact_clips(cfg, run)
    paths[0].write_bytes(paths[1].read_bytes())
    monkeypatch.setattr(workers, "_load_caption_inference", lambda *a: pytest.fail("model loaded before preflight"))
    with pytest.raises(ValueError, match="count/PTS"):
        workers.caption(cfg, REPO, run)


def fake_images(cfg, repo, run, start, end, clip):
    return start, {"clip": str(clip), "decoded_frames": end - start,
                   "indices": [0, 0, 0, 0], "resize_short_edge": 672}


def test_caption_resumes_persisted_rows_and_recreates_final_outputs(prepared, monkeypatch):
    run, cfg = prepared
    calls = []
    fail = [True]
    def infer(start, prompt, official):
        calls.append(start)
        if start == 1 and fail[0]:
            raise RuntimeError("simulated interruption")
        return f"source starts at {start}"
    monkeypatch.setattr(workers, "_caption_images", fake_images)
    monkeypatch.setattr(workers, "_load_caption_inference", lambda *a: infer)
    with pytest.raises(RuntimeError, match="interruption"):
        workers.caption(cfg, REPO, run)
    assert not (run / "captions.json").exists()
    saved = json.loads(Path(cfg["caption_checkpoint"]).read_text())
    assert len(saved["records"]) == 1
    assert calls == [0, 1]
    fail[0] = False
    workers.caption(cfg, REPO, run)
    assert calls == [0, 1, 1, 5]
    captions = json.loads((run / "captions.json").read_text())
    sampling = json.loads((run / "caption_sampling.json").read_text())
    assert [r["text"] for r in captions] == [f"source starts at {start}" for start in [0, 1, 5]]
    assert len(sampling) == 3
    (run / "captions.json").unlink()
    (run / "caption_sampling.json").unlink()
    monkeypatch.setattr(workers, "_load_caption_inference", lambda *a: pytest.fail("completed checkpoint loaded model"))
    workers.caption(cfg, REPO, run)
    assert json.loads((run / "captions.json").read_text()) == captions
    assert json.loads((run / "caption_sampling.json").read_text()) == sampling


@pytest.mark.parametrize("mutation", ["settings", "model", "checkpoint", "signature"])
def test_checkpoint_changes_refused_before_gpu(prepared, monkeypatch, mutation):
    run, cfg = prepared
    monkeypatch.setattr(workers, "_caption_images", fake_images)
    monkeypatch.setattr(workers, "_load_caption_inference", lambda *a: lambda *a: "caption")
    workers.caption(cfg, REPO, run)
    if mutation == "settings":
        cfg["caption_max_new_tokens"] = 31
    elif mutation == "model":
        write_json(Path(cfg["models"]["pllava"]) / "config.json", {"changed": True})
    elif mutation == "signature":
        monkeypatch.setenv("ETRI_RUN_SIGNATURE", "changed")
    else:
        path = Path(cfg["caption_checkpoint"])
        saved = json.loads(path.read_text())
        saved["records"][0]["row"]["text"] = "corrupted caption"
        write_json(path, saved)
    monkeypatch.setattr(workers, "_load_caption_inference", lambda *a: pytest.fail("model loaded before checkpoint check"))
    with pytest.raises(ValueError, match="identity/checksum"):
        workers.caption(cfg, REPO, run)


def test_checkpoint_rejects_gap_and_invalid_sampling(tmp_path):
    checkpoint = CaptionCheckpoint(tmp_path / "checkpoint.json", {"test": True}, 2, True)
    row = {"path": "clips/sample/00000.mp4", "text": "caption", "flow": 0.0}
    with pytest.raises(ValueError, match="advance exactly"):
        checkpoint.save(1, row, None)
    with pytest.raises(ValueError, match="sampling"):
        checkpoint.save(0, row, {"decoded_frames": 0})
