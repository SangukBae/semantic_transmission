"""Exercise the batch launcher with a fake interpreter; never load Qwen."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(tmp_path):
    repo = tmp_path / 'repo'
    script = repo / 'scripts/train_qwen35_evidence_repair.sh'
    script.parent.mkdir(parents=True)
    script.write_text((ROOT / 'scripts/train_qwen35_evidence_repair.sh').read_text())
    fake = repo / '.local/qwen35_env/bin/python'
    fake.parent.mkdir(parents=True)
    fake.write_text(f'#!{sys.executable}\n' + '''
import json, os, resource, sys
with open(os.environ['QWEN_REPAIR_TEST_LOG'], 'a') as f:
    f.write(json.dumps({'args': sys.argv[1:], 'pid': os.getpid(),
                       'nofile': resource.getrlimit(resource.RLIMIT_NOFILE)[0]}) + '\\n')
if sys.argv[-1] == os.environ.get('QWEN_REPAIR_TEST_FAIL_ARM'):
    sys.exit(23)
''')
    fake.chmod(0o755)
    log = tmp_path / 'calls.jsonl'

    def run(*args, fail_arm='', limited=False):
        env = dict(os.environ, QWEN_REPAIR_TEST_LOG=str(log),
                   QWEN_REPAIR_TEST_FAIL_ARM=fail_arm)
        if limited:
            command = ['bash', '-c', 'ulimit -Sn 1024; exec bash "$@"', 'launcher', str(script), *args]
        else:
            command = ['bash', str(script), *args]
        result = subprocess.run(command, cwd=tmp_path, env=env, text=True,
                                capture_output=True, timeout=20)
        calls = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
        return result, calls

    return run


def test_all_runs_four_separate_processes_in_order(launcher):
    result, calls = launcher('train', 'all')
    assert result.returncode == 0, result.stderr
    assert [c['args'] for c in calls] == [
        ['-m', 'semantic_transmission.qwen35_evidence_repair', 'train', arm]
        for arm in ('caption', 'evidence', 'judgment', 'preference')]
    assert len({c['pid'] for c in calls}) == 4
    assert result.stdout.endswith('ALL_CONDITIONS_COMPLETE\n')


def test_failure_stops_before_later_conditions_and_preserves_exit_code(launcher):
    result, calls = launcher('train', 'all', fail_arm='evidence')
    assert result.returncode == 23
    assert [c['args'][-1] for c in calls] == ['caption', 'evidence']
    assert 'ALL_CONDITIONS_COMPLETE' not in result.stdout


@pytest.mark.parametrize('args, forwarded', [
    ((), ['prepare']), (('dry-run',), ['dry-run']),
    (('train', 'judgment'), ['train', 'judgment']),
])
def test_default_remains_nontraining_and_individual_commands_are_preserved(launcher, args, forwarded):
    result, calls = launcher(*args)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    assert calls[0]['args'][2:] == forwarded


@pytest.mark.parametrize('extra', ['--help', 'caption'])
def test_extra_batch_arguments_cannot_start_training(launcher, extra):
    result, calls = launcher('train', 'all', extra)
    assert result.returncode == 2
    assert calls == []
    assert 'Usage:' in result.stderr


def test_batch_children_inherit_wsl_open_file_limit_fix(launcher):
    if resource.getrlimit(resource.RLIMIT_NOFILE)[1] < 8192:
        pytest.skip('Host hard limit below training requirement')
    result, calls = launcher('train', 'all', limited=True)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 4
    assert all(c['nofile'] >= 8192 for c in calls)
