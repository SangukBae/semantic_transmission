import cv2
import numpy as np
import pytest

from semantic_transmission.receiver_motion_repair import retain_motion, warp_to_received_tracks


def pattern():
    rng = np.random.default_rng(526)
    return cv2.GaussianBlur(rng.integers(0, 256, (160, 288, 3), dtype=np.uint8), (5, 5), 0)


def shifted(image, x):
    return cv2.warpAffine(image, np.float32([[1, 0, x], [0, 1, 0]]), (288, 160),
                          borderMode=cv2.BORDER_REFLECT_101)


def test_motion_guard_recovers_real_excursion_erased_by_endpoint_interpolation():
    image = pattern()
    original = np.stack([shifted(image, x) for x in (0, 4, 8, 4, 0)])
    smooth = np.repeat(image[None], 5, axis=0)
    before = original.copy(); out, audit = retain_motion(original, smooth)
    assert np.abs(out[2].astype(float)-original[2]).mean() < .1
    assert np.array_equal(out[[0, -1]], original[[0, -1]])
    assert np.array_equal(original, before)
    assert audit['receiver_reads_source'] is False


def test_static_input_is_not_given_artificial_motion():
    frames = np.repeat(pattern()[None], 9, axis=0)
    out, _ = retain_motion(frames, frames)
    assert np.array_equal(out, frames)
    out, audit = warp_to_received_tracks(frames, [frames[0], frames[-1]], None)
    assert np.array_equal(out, frames) and audit['bypassed']


def test_paid_positions_reduce_known_generation_position_error():
    image = pattern(); n = 13
    movement = np.asarray([0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0], np.float32)
    points = np.asarray([(x, y) for x in range(36, 270, 30) for y in range(25, 145, 25)], np.float32)
    xy = np.repeat(points[None], n, axis=0); xy[:, :, 0] += movement[:, None]
    trajectory = dict(frames=n, origins=np.zeros(len(points), int), xy=xy,
                      valid=np.ones((n, len(points)), bool))
    generated = np.repeat(image[None], n, axis=0); before = generated.copy()
    out, audit = warp_to_received_tracks(generated, [image, image], trajectory)
    truth = shifted(image, 6)[20:-20, 35:-35]
    old_error = np.abs(generated[6, 20:-20, 35:-35].astype(float)-truth).mean()
    new_error = np.abs(out[6, 20:-20, 35:-35].astype(float)-truth).mean()
    assert new_error < .85*old_error
    assert np.array_equal(out[[0, -1]], generated[[0, -1]])
    assert np.array_equal(generated, before)
    assert audit['additional_transmission'] == 0


def test_invalid_trajectory_and_inputs_rejected():
    image = pattern(); frames = np.repeat(image[None], 3, axis=0)
    with pytest.raises(ValueError):
        retain_motion(frames, frames[:2])
    with pytest.raises(ValueError):
        warp_to_received_tracks(frames, [image, image], dict(frames=4, xy=np.zeros((4, 2, 2)),
                                                           valid=np.ones((4, 2), bool)))
