import runpy

import cv2
import numpy as np
import pytest


def test_sparse_packet_timing_retains_hold_and_missing_slots(tmp_path):
    adapter = runpy.run_path('scripts/diagnose_metric_v3_pairs.py')
    paths = [tmp_path / '0.png', tmp_path / '3.png']
    for path, level in zip(paths, (25, 75)):
        assert cv2.imwrite(str(path), np.full((192, 320, 3), level, np.uint8))
    frames, status, offset = adapter['sparse_video'](paths, [0, 3], 8., 5)
    assert frames[:, 0, 0, 0].tolist() == [25, 25, 25, 75, 75]
    assert status == ['ok', 'not_transmitted', 'not_transmitted', 'ok', 'not_transmitted']
    assert offset == 0
    with pytest.raises(ValueError, match='boundary mismatch'):
        adapter['captions']([{'text': 'first'}, {'text': 'extra'}], [0, 3], 8.)


def test_off_grid_keyframe_status_refers_to_its_actual_pixels(tmp_path):
    adapter = runpy.run_path('scripts/diagnose_metric_v3_pairs.py')
    paths = [tmp_path / '0.png', tmp_path / '16.png']
    for path, level in zip(paths, (25, 75)):
        assert cv2.imwrite(str(path), np.full((192, 320, 3), level, np.uint8))
    frames, status, offset = adapter['sparse_video'](paths, [0, 16], 24., 7)
    # 16/24 seconds is after sample 5 (5/8); its pixels first occur at sample 6.
    assert frames[5, 0, 0, 0] == 25 and status[5] == 'not_transmitted'
    assert frames[6, 0, 0, 0] == 75 and status[6] == 'ok'
    assert offset == pytest.approx(1 / 12)
