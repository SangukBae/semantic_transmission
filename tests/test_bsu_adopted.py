import json
from pathlib import Path

import pytest

from semantic_transmission import bsu
from semantic_transmission.artifacts import write_json


def test_profile_matches_validated_model_and_excludes_rate_search():
    reference = json.loads((bsu.REPO / 'docs/validation/2026-10-16-bsu-predictive-final.json').read_text())
    assert {k: bsu.profile()[k] for k in reference['adopted_profile']} == reference['adopted_profile']
    assert bsu.profile()['adaptive_rate_search'] is False


def test_default_rejects_experimental_profile(tmp_path, monkeypatch):
    path = tmp_path / 'profile.json'
    write_json(path, {**bsu.profile(), 'temporal_stride': 4})
    monkeypatch.setattr(bsu, 'CONFIG', path)
    with pytest.raises(ValueError, match='adopted profile changed'):
        bsu.profile()


def test_profile_and_status_do_not_load_gpu_or_prepare_inputs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bsu, 'preflight', lambda _: pytest.fail('read-only command checked runtime'))
    bsu.main(['profile'])
    assert json.loads(capsys.readouterr().out)['qp'] == 32
    bsu.main(['status', '--output', str(tmp_path / 'missing')])
    assert json.loads(capsys.readouterr().out)['status'] == 'NOT_COMPLETE'
    assert not (tmp_path / 'missing').exists()


def test_check_is_read_only_and_run_dispatches_fixed_profile(tmp_path, monkeypatch, capsys):
    from semantic_transmission import bsu_adaptive_dense as pipeline
    root = tmp_path / 'new'
    record = dict(profile=bsu.profile(), signature='fixed')
    monkeypatch.setattr(bsu, 'preflight', lambda _: record)
    argv = ['--semantic-run', str(tmp_path / 'old'), '--source-frames', str(tmp_path / 'source'),
            '--output', str(root)]
    bsu.main(['check', *argv])
    assert json.loads(capsys.readouterr().out)['model_inference_started'] is False
    assert not root.exists()
    calls = []

    def sender(args):
        assert args.output == root
        assert (args.qp, args.temporal_stride, args.min_original_keys, args.protect_semantic_keys) == (32, 2, 5, False)
        calls.append('sender')
        write_json(root / 'sender_protocol.json', {'test': True})
        return root / 'predictive_sender'

    def execute(args):
        assert args.output == root / 'delivery'
        assert args.neural_run == root / 'predictive_sender'
        calls.append('receiver')
        write_json(args.output / 'RESULT.json', {'status': 'COMPLETE_ADAPTIVE_DENSE_BSU'})

    monkeypatch.setattr(pipeline, 'prepare_sender', sender)
    monkeypatch.setattr(pipeline, 'execute', execute)
    bsu.main(argv)
    result = json.loads((root / 'RESULT.json').read_text())
    assert result['default_profile'] == bsu.profile()['policy']
    assert calls == ['sender', 'receiver']
    # An incompatible frozen default must stop before another sender starts.
    record['signature'] = 'different'
    with pytest.raises(ValueError):
        bsu.main(argv)
    assert calls == ['sender', 'receiver']


def test_entrypoint_and_historical_commands_remain_distinct():
    shell = (bsu.REPO / 'scripts/run_bsu.sh').read_text()
    assert '-m semantic_transmission.bsu ' in shell
    assert '-m semantic_transmission.terminal_gpu_progress' in shell
    assert 'semantic_transmission.bsu_adaptive_rate' not in shell
    for script, module in [('run_bsu_webvid.sh', 'bsu_default'), ('run_bsu_kinetics.sh', 'bsu_kinetics')]:
        assert f'-m semantic_transmission.{module}' in (bsu.REPO / 'scripts' / script).read_text()
