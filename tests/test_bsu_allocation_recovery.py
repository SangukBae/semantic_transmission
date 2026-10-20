import json
from types import SimpleNamespace

import pytest

from semantic_transmission import auto_extraction as ae
from semantic_transmission import bsu_allocation_recovery as recovery
from semantic_transmission import qwen35_pairs_comparison as pc
from semantic_transmission.artifacts import sha256, write_json


def response(raw, **overrides):
    return dict(raw_text=raw, truncated=False, model_runtime=dict(
        model_id=pc.MODEL_ID, revision=pc.REVISION, quantization='nf4'), **overrides)


def request(tmp_path):
    path = tmp_path/'draft.json'
    identity = dict(task='draft', source_id='test')
    prompt = 'Describe source frames.'
    complete = dict(identity, prompt_sha256=ae.digest(prompt), labels=['frame 0'], max_new_tokens=384)
    for i in range(3):
        record = dict(identity=complete, error='invalid uncertainties',
            **response('{"caption":"Visible object.","uncertains":[]}'))
        write_json(tmp_path/'failed_attempts'/f'draft_{i}.json', ae.seal(record))
    return path, identity, prompt


def invoke(vlm, path, identity, prompt, original=pc.infer):
    return recovery.infer(vlm, path, identity, ['image'], ['frame 0'], prompt,
        ae.caption_validator({}), original=original, profile={'signature':'test-only'})


def test_real_validator_typo_feedback_and_original_cache_reuse(tmp_path):
    path, identity, prompt = request(tmp_path)
    old = {p:sha256(p) for p in (tmp_path/'failed_attempts').glob('*.json')}
    calls = []
    raw = '{"caption":"Visible object.","uncertainties":[]}'
    def generate(images, labels, actual_prompt, tokens, **kwargs):
        calls.append(actual_prompt)
        assert 'invalid uncertainties' in actual_prompt
        assert '"uncertains" is invalid' in actual_prompt
        assert tokens <= 1152 and kwargs['task'] == 'draft'
        return response(raw)
    vlm = SimpleNamespace(config={}, generate=generate)
    assert invoke(vlm, path, identity, prompt) == json.loads(raw)
    saved_hash = sha256(path)
    assert invoke(vlm, path, identity, prompt) == json.loads(raw)
    assert len(calls) == 1 and sha256(path) == saved_hash
    record = ae.unseal(path)
    assert record['raw_text'] == raw and record['recovery']['signature'] == 'test-only'
    assert all(sha256(p) == digest for p,digest in old.items())


def test_repeated_invalid_output_is_bounded_and_never_accepted(tmp_path):
    path, identity, prompt = request(tmp_path)
    calls = []
    def generate(*args, **kwargs):
        calls.append(1)
        return response('{"caption":"Visible object.","uncertains":[]}')
    with pytest.raises(RuntimeError, match='error-specific caption retries exhausted'):
        invoke(SimpleNamespace(config={}, generate=generate), path, identity, prompt)
    assert len(calls) == 3 and not path.exists()
    assert len(list((tmp_path/'failed_attempts').glob('*.json'))) == 6


def test_corrupt_success_cache_is_not_replaced(tmp_path):
    path, identity, prompt = request(tmp_path)
    write_json(path, dict(checksum='invalid'))
    digest = sha256(path)
    vlm = SimpleNamespace(config={}, generate=lambda *a,**kw: pytest.fail('must not regenerate'))
    with pytest.raises(ValueError, match='checksum mismatch'):
        invoke(vlm, path, identity, prompt)
    assert sha256(path) == digest


def test_runtime_errors_are_not_hidden_as_schema_failures(tmp_path):
    def original(*args):
        raise RuntimeError('CUDA out of memory')
    vlm = SimpleNamespace(config={})
    with pytest.raises(RuntimeError, match='CUDA out of memory'):
        invoke(vlm, tmp_path/'draft.json', dict(task='draft'), 'prompt', original)


def test_fresh_success_uses_original_inference_unchanged(tmp_path):
    calls = []
    def generate(images, labels, prompt, tokens, **kwargs):
        calls.append(prompt)
        return response('{"caption":"Visible object.","uncertainties":[]}')
    vlm = SimpleNamespace(config={}, generate=generate)
    path = tmp_path/'draft.json'
    invoke(vlm, path, dict(task='draft'), 'unchanged prompt')
    assert calls == ['unchanged prompt']
    assert 'recovery' not in ae.unseal(path)


def test_wrong_model_provenance_is_fatal(tmp_path):
    path, identity, prompt = request(tmp_path)
    def generate(*args, **kwargs):
        result = response('{"caption":"Visible object.","uncertainties":[]}')
        result['model_runtime']['revision'] = 'wrong'
        return result
    with pytest.raises(ValueError, match='model provenance differs'):
        invoke(SimpleNamespace(config={}, generate=generate), path, identity, prompt)
    assert not path.exists()


def test_new_failure_runs_original_attempts_then_error_specific_retry(tmp_path):
    calls = []
    def generate(images, labels, prompt, tokens, **kwargs):
        calls.append(prompt)
        if len(calls) <= 3:
            return response('{"caption":"Visible object.","uncertains":[]}')
        assert 'invalid uncertainties' in prompt
        return response('{"caption":"Visible object.","uncertainties":[]}')
    path = tmp_path/'draft.json'
    invoke(SimpleNamespace(config={}, generate=generate), path, dict(task='draft'), 'original prompt')
    assert len(calls) == 4 and calls[0] == 'original prompt'
    assert len(list((tmp_path/'failed_attempts').glob('*.json'))) == 3


def test_recovery_receipt_is_bound_to_code_and_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery.fast, 'verify_execution', lambda _: dict(
        signature='execution', parent_protocol_signature='protocol'))
    receipt = recovery.prepare(tmp_path)
    assert recovery.verify(tmp_path) == receipt
    receipt['parent_execution_signature'] = 'different'
    receipt['signature'] = recovery.fingerprint({k:v for k,v in receipt.items() if k != 'signature'})
    write_json(tmp_path/recovery.PROFILE, receipt)
    with pytest.raises(ValueError, match='another experiment'):
        recovery.verify(tmp_path)
    receipt['parent_execution_signature'] = 'execution'
    receipt['code'][str(recovery.Path(recovery.__file__).resolve())] = 'different'
    receipt['signature'] = recovery.fingerprint({k:v for k,v in receipt.items() if k != 'signature'})
    write_json(tmp_path/recovery.PROFILE, receipt)
    with pytest.raises(ValueError, match='code changed'):
        recovery.verify(tmp_path)
