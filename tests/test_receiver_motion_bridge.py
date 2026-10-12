import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from semantic_transmission import receiver_motion_bridge as bridge


def textured(seed=1):
    image = np.random.default_rng(seed).integers(0, 256, (96, 160, 3), dtype=np.uint8)
    return cv2.GaussianBlur(image, (7, 7), 1.2)


def translate(dx, dy=0):
    value = bridge.IDENTITY.copy()
    value[:2, 2] = [dx, dy]
    return value


def test_registration_direction_and_motion_are_preserved():
    image = textured()
    target = bridge.warp(image, translate(4, 2))
    score, matrix = bridge.register(image, target)
    assert score > .97
    assert np.allclose(matrix[:2, 2], [4, 2], atol=.3)
    plan = bridge.pair_plan(image, target, 24, 24)
    assert plan["accepted"]
    # Generated frame is frozen at the start. The repaired middle must move
    # halfway along the received trajectory, rather than freeze the video.
    _, repaired, _ = bridge.repair_frame(image, image, target, matrix, .5)
    expected = bridge.warp(image, translate(2, 1))
    crop = np.s_[8:-8, 8:-8]
    assert np.mean(abs(repaired[crop].astype(float)-expected[crop])) < 2
    assert np.mean(abs(repaired[crop].astype(float)-image[crop])) > 5


@pytest.mark.parametrize("other", [textured(90), np.zeros((96, 160, 3), np.uint8)])
def test_scene_cut_or_no_texture_is_rejected(other):
    assert not bridge.pair_plan(textured(), other, 12, 24)["accepted"]


def test_long_gap_and_reflected_geometry_are_rejected():
    image = textured()
    assert not bridge.pair_plan(image, image, 25, 24)["accepted"]
    matrix = bridge.IDENTITY.copy()
    matrix[0, 0] = -1
    assert not bridge.plausible(matrix, image.shape)


def test_disagreeing_object_region_retains_generated_pixels():
    left = textured()
    right = left.copy()
    right[25:70, 40:110] = [255, 0, 0]
    generated = left.copy()
    # Stable reference registration for fallback; object mismatch must not
    # crossfade an invented double object in the center of that region.
    _, output, detail = bridge.repair_frame(generated, left, right, bridge.IDENTITY, .5)
    assert np.array_equal(output[40:55, 60:90], generated[40:55, 60:90])
    assert detail["reference_weight"] < 1


def test_rejected_neighbor_has_zero_bridge_weight_at_transition():
    assert bridge.edge_weight(12, 12, 24, False, True) == 0
    assert bridge.edge_weight(24, 12, 24, True, False) == 0
    assert bridge.edge_weight(18, 12, 24, False, False) == 1


def fixture_run(tmp_path):
    run = tmp_path / "run"
    receiver = run / "receiver"
    keys = receiver / "frames/sample/key_frames_received"
    frames = receiver / "reconstruction/sample_0000_frames"
    keys.mkdir(parents=True)
    frames.mkdir(parents=True)
    a = textured()
    b = bridge.warp(a, translate(2))
    for i, image in [(0, a), (4, b)]:
        cv2.imwrite(str(keys / f"{i}.png"), image)
    for i in range(5):
        cv2.imwrite(str(frames / f"{i:05}.png"), a)
    (receiver / "decoder_inputs.json").write_text(json.dumps(dict(
        video=dict(frames=5, fps=24, width=160, height=96), indices=[0, 4])))
    return run


def test_receiver_only_integration_resume_and_tamper_detection(tmp_path):
    # This fixture has no source frames/video, captions, sender, or models.
    run = fixture_run(tmp_path)
    output = tmp_path / "repair"
    result = bridge.execute(run, output)
    assert result["frames"] == 5 and result["accepted_pairs"] == 1
    protocol = json.loads((output / "protocol.json").read_text())
    assert not protocol["source_images_used_for_inference"]
    assert protocol["additional_channel_uses"] == 0
    first = cv2.imread(str(output / "hybrid_frames/00000.png"))
    last = cv2.imread(str(output / "hybrid_frames/00004.png"))
    assert np.array_equal(first, cv2.imread(str(run / "receiver/frames/sample/key_frames_received/0.png")))
    assert np.array_equal(last, cv2.imread(str(run / "receiver/frames/sample/key_frames_received/4.png")))
    stamp = (output / "hybrid.mp4").stat().st_mtime_ns
    assert bridge.execute(run, output) == result
    assert (output / "hybrid.mp4").stat().st_mtime_ns == stamp
    with (output / "hybrid.mp4").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="completed output changed"):
        bridge.execute(run, output)


def test_input_or_code_contract_is_not_silently_replaced(tmp_path):
    run = fixture_run(tmp_path)
    output = tmp_path / "repair"
    bridge.execute(run, output, "geometry")
    with pytest.raises(ValueError, match="frozen inputs/code/options changed"):
        bridge.execute(run, output, "hybrid")
    key = run / "receiver/frames/sample/key_frames_received/4.png"
    cv2.imwrite(str(key), textured(6))
    with pytest.raises(ValueError, match="frozen inputs/code/options changed"):
        bridge.execute(run, output, "geometry")


def test_input_output_overlap_and_missing_frames_fail(tmp_path):
    run = fixture_run(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        bridge.execute(run, run / "repair")
    (run / "receiver/reconstruction/sample_0000_frames/00002.png").unlink()
    with pytest.raises(ValueError, match="timeline is incomplete"):
        bridge.execute(run, tmp_path / "repair")
    assert not (tmp_path / "repair").exists()
