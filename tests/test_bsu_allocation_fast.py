import copy
import json

import pytest

from semantic_transmission import bsu_allocation_fast as fast
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.webvid5 import fingerprint


@pytest.fixture
def execution(tmp_path, monkeypatch):
    monkeypatch.setattr(fast.labels, 'verify', lambda _: {'signature': 'original'})
    code = tmp_path/'cache.py'; code.write_text('unchanged weights\n')
    benchmark = tmp_path/'benchmark.json'; write_json(benchmark, {'status': 'COMPLETE'})
    config = dict(gpu_cache_bytes=4096, reserve_bytes=0, prompt_cache_bytes=100)
    validation = tmp_path/'validation.json'
    write_json(validation, dict(status='PASSED_EXACT_FASTER_REPLAY', cache_config=config,
        comparisons=[dict(frames_identical=True, mp4_identical=True)]*2,
        speedup=1.2, benchmark_receipts=[dict(path=str(benchmark), sha256=sha256(benchmark))]))
    receipt = dict(parent_protocol_signature='original', code={str(code):sha256(code)},
        validation=dict(path=str(validation), sha256=sha256(validation)), cache_config=config)
    receipt['signature'] = fingerprint(receipt)
    write_json(tmp_path/fast.PROFILE, receipt)
    return tmp_path, receipt


def test_execution_requires_measured_unchanged_faster_results(execution):
    root, receipt = execution
    assert fast.verify_execution(root) == receipt
    benchmark = root/'benchmark.json'; benchmark.write_text('changed')
    with pytest.raises(ValueError, match='benchmark receipt changed'):
        fast.verify_execution(root)


def test_execution_rejects_unvalidated_code_change(execution):
    root, _ = execution
    (root/'cache.py').write_text('different code')
    with pytest.raises(ValueError, match='execution code changed'):
        fast.verify_execution(root)


@pytest.mark.parametrize('change', ['frames', 'speed', 'config'])
def test_execution_rejects_failed_or_mismatched_validation(execution, change):
    root, receipt = execution
    path = root/'validation.json'; value = json.loads(path.read_text())
    if change == 'frames': value['comparisons'][1]['frames_identical'] = False
    if change == 'speed': value['speedup'] = 1.01
    if change == 'config': value['cache_config']['gpu_cache_bytes'] += 1
    write_json(path, value)
    receipt['validation']['sha256'] = sha256(path)
    receipt['signature'] = fingerprint({k:v for k,v in receipt.items() if k != 'signature'})
    write_json(root/fast.PROFILE, receipt)
    with pytest.raises(ValueError):
        fast.verify_execution(root)


def test_monitor_preserves_training_step_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(fast.labels, 'status_lines', lambda _: ['Status RUNNING', 'Stage fit predictor'])
    write_json(tmp_path/'predictor/status.json', dict(step=125, total=1000, status='RUNNING'))
    rows = fast.status_lines(tmp_path)
    assert any('125/1000 steps (12.5%)' in r for r in rows)
