"""Reproduce interactive-shell limits without invoking GPU work."""
from pathlib import Path
import subprocess
import resource
import shutil
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
WRAPPER='scripts/train_qwen35_grounded.sh'


@pytest.fixture(autouse=True)
def isolated_launcher(tmp_path, monkeypatch):
    """Exercise the real wrapper using this test's CPU interpreter, not a local venv."""
    repo = ROOT
    target = tmp_path / WRAPPER
    target.parent.mkdir(parents=True)
    shutil.copy2(repo / WRAPPER, target)
    (tmp_path / 'src').symlink_to(repo / 'src', target_is_directory=True)
    python = tmp_path / '.local/qwen35_env/bin/python'
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    monkeypatch.setattr(sys.modules[__name__], 'ROOT', tmp_path)


def launch(soft,hard):
    # ulimit applies only in this test's child; no parent/system-wide mutation.
    command=f'ulimit -Sn {soft}; ulimit -Hn {hard}; exec bash {WRAPPER} --help'
    return subprocess.run(['bash','-c',command],cwd=ROOT,text=True,capture_output=True,timeout=30)


def test_interactive_1024_limit_is_raised_before_python():
    if resource.getrlimit(resource.RLIMIT_NOFILE)[1]<8192:pytest.skip('Host hard limit below target')
    result=launch(1024,8192)
    assert result.returncode==0,result.stderr
    assert 'open_files_soft=8192 hard=8192 (inherited_soft=1024)' in result.stderr
    assert '{prepare,freeze,verify,cache,preflight,caption,grounded,both}' in result.stdout


def test_larger_limit_is_preserved():
    if resource.getrlimit(resource.RLIMIT_NOFILE)[1]<16384:pytest.skip('Host hard limit below target')
    result=launch(16384,16384)
    assert result.returncode==0,result.stderr
    assert 'open_files_soft=16384 hard=16384 (inherited_soft=16384)' in result.stderr


def test_insufficient_hard_limit_fails_before_model_loading():
    result=launch(1024,4096)
    assert result.returncode!=0
    assert 'current soft=1024 hard=4096' in result.stderr
    assert 'usage:' not in result.stdout
