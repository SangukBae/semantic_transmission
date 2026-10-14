from pathlib import Path

import cv2
import numpy as np
import pytest

from semantic_transmission import bsu_key_context_experiment as experiment
from semantic_transmission import official_quality, research_quality, stutter_metrics, etri_60s_check
from semantic_transmission.artifacts import write_json


def test_receiver_rejects_evaluation_data(tmp_path):
    with pytest.raises(ValueError, match='source fields'):
        experiment.generate({'source_frames': ['original.png']}, tmp_path, 'frozen', 'context_only')


@pytest.fixture
def cache_case(tmp_path, monkeypatch):
    source = np.zeros((5, 16, 24, 3), np.uint8)
    donor = np.full_like(source, 40)
    output = donor.copy()
    output[2] = 20
    arrays = {'source': source, 'donor': donor, 'output': output, 'keys': source[[0, 4]]}
    monkeypatch.setattr(experiment.common, 'pixels', lambda name, *a: arrays[name])
    monkeypatch.setattr(research_quality, 'read_video', lambda name: arrays[name])
    monkeypatch.setattr(etri_60s_check, 'pts_audit', lambda *a: {})
    transitions = []
    def temporal(a, b, *args):
        transitions.append((a.copy(), b.copy()))
        return {'checked': len(b)-1}
    monkeypatch.setattr(stutter_metrics, 'evaluate', temporal)
    monkeypatch.setattr(experiment.common, 'boundary_diagnostics', lambda *a: [])
    def score(a, b, i):
        return dict(frame=i, psnr_db=cv2.PSNR(a, b), ssim=.7, lpips_vgg=.3, clip=.9, dists=.2)
    old = [score(a, b, i) for i, (a, b) in enumerate(zip(source, donor))]
    score_path = tmp_path/'donor.json'; write_json(score_path, old)
    quality_path = tmp_path/'donor_quality.json'
    write_json(quality_path, {b: dict(quality=official_quality.summarize(old, source.shape))
                             for b in ('lossless_frames', 'delivered_mp4')})
    calls = []
    class Metrics:
        def __init__(self, device): pass
        def evaluate(self, a, b, progress):
            calls.append((a.copy(), b.copy()))
            rows = [score(x, y, i) for i, (x, y) in enumerate(zip(a, b))]
            return official_quality.summarize(rows, a.shape), rows
    monkeypatch.setattr(official_quality, 'OfficialMetrics', Metrics)
    job = dict(arm='guard_only', source_frames='source', source_movie=None, frames=5,
               keys=[0, 4], key_paths='keys', frames_paths='output', movie='output',
               donor=dict(frames_paths='donor', movie='donor', quality=str(quality_path),
                          layout='comparison', scores={b:str(score_path) for b in ('lossless_frames', 'delivered_mp4')}))
    return job, arrays, calls, transitions


def test_only_exact_pixels_reuse_scores_and_all_transitions_recomputed(cache_case, tmp_path):
    job, arrays, calls, transitions = cache_case
    experiment.evaluate(job, tmp_path, 'frozen', 'cpu')
    assert len(calls) == 2 and all(len(a) == len(b) == 1 for a, b in calls)
    assert all(np.array_equal(b[0], arrays['output'][2]) for _, b in calls)
    assert len(transitions) == 2 and all(len(a) == len(b) == 5 for a, b in transitions)
    q = experiment.read(tmp_path/'quality.json')
    for boundary in ('lossless_frames', 'delivered_mp4'):
        assert q[boundary]['exact_pixel_metric_reuse'] == 4
        rows = experiment.csv_rows(tmp_path/f'quality_{boundary}.csv')
        assert [r['frame'] for r in rows] == list(range(5))
        assert rows[2]['psnr_db'] == cv2.PSNR(arrays['source'][2], arrays['output'][2])
    assert experiment.common.verify_receipt(tmp_path, 'evaluation', 'frozen')


def test_different_source_cannot_reuse_scores(cache_case, tmp_path):
    job, arrays, calls, _ = cache_case
    arrays['source'][0] = 10
    with pytest.raises(ValueError, match='cached source/pixels differ'):
        experiment.evaluate(job, tmp_path, 'frozen', 'cpu')
    assert not calls and not (tmp_path/'evaluation.json').exists()


def test_anchored_arm_fails_if_received_pixels_not_preserved(cache_case, tmp_path):
    job, _, _, _ = cache_case
    job['arm'] = 'context_only'
    with pytest.raises(ValueError, match='received keys not exact'):
        experiment.evaluate(job, tmp_path, 'frozen', 'cpu')
    assert not (tmp_path/'evaluation.json').exists()
