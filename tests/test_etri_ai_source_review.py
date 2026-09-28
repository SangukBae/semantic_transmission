"""Keep sampled AI observations separate from independent, complete truth."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def review_package(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / 'scripts/etri_ai_source_review.py'
    spec = importlib.util.spec_from_file_location('ai_source_review', source)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    benchmark, output = tmp_path / 'benchmark', tmp_path / 'review'
    monkeypatch.setattr(app, 'BENCHMARK', benchmark)
    monkeypatch.setattr(app, 'OUTPUT', output)
    app.save(benchmark / 'manifest.json', [{'id':'a', 'processed_sha256':'frozen', 'clip_duration_sec':60}])
    image = output / 'evidence/a/page_01.jpg'
    image.parent.mkdir(parents=True)
    image.write_bytes(b'fixture image bytes')
    path = str(image.relative_to(output))
    app.save(image.parent / 'sampling.json', {'input_sha256':'frozen', 'sampled_frames':31,
        'sheets':[{'path':path, 'sha256':app.digest(image)}]})
    item = {'id':'a', 'input_sha256':'frozen', 'reviewer_kind':'AI', 'independent_ground_truth':False,
            'viewed_evidence':[path], 'observations':[{'start_sec':0, 'end_sec':60}],
            'limitations':['2-second sampled stills only']}
    return app, item, image


def test_complete_ai_review_never_certifies_human_or_framewise_truth(review_package):
    app, item, _ = review_package
    app.save(app.OUTPUT / 'reviews.json', [item])
    app.audit()
    result = json.loads((app.OUTPUT / 'audit.json').read_text())
    assert result['status'] == 'PASS_AI_SAMPLED_REVIEW_PACKAGE'
    assert result['independent_human_ground_truth_ready'] is False
    assert result['all_frames_visually_reviewed'] is False
    assert result['reconstruction_error_evaluation_done'] is False


def test_empty_ai_reviews_remain_incomplete(review_package):
    app, _, _ = review_package
    app.save(app.OUTPUT / 'reviews.json', [])
    app.audit()
    assert json.loads((app.OUTPUT / 'audit.json').read_text())['status'] == 'INCOMPLETE'


@pytest.mark.parametrize('change', [
    {'independent_ground_truth':True}, {'reviewer_kind':'human'},
    {'input_sha256':'other'}, {'viewed_evidence':[]},
    {'observations':[{'start_sec':0, 'end_sec':61}]},
])
def test_unjustified_claim_or_mismatched_evidence_is_rejected(review_package, change):
    app, item, _ = review_package
    app.save(app.OUTPUT / 'reviews.json', [dict(item, **change)])
    with pytest.raises(SystemExit):
        app.audit()


def test_modified_evidence_is_rejected(review_package):
    app, item, image = review_package
    app.save(app.OUTPUT / 'reviews.json', [item])
    image.write_bytes(b'changed image')
    with pytest.raises(SystemExit):
        app.audit()
