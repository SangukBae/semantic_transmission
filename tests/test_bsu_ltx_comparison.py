import os
import subprocess
import sys
import pytest
from semantic_transmission import bsu_ltx_comparison as comparison


def fixture():
    quality=dict(delivered_mp4={k:1. for k in comparison.KEYS})
    baseline=dict(id='video',frames=9,bsu_channel_uses=20,lgvsc_channel_uses=10,
                  bsu_quality=quality,lgvsc_quality=quality)
    result=dict(id='video',frames=9,total_channel_uses=20,quality=quality)
    return dict(rows=[baseline],frames=9),[result]


def test_report_keeps_lgvsc_rate_difference_explicit(tmp_path):
    audit,rows=fixture()
    comparison.report(tmp_path,audit,rows,'test')
    r=comparison.read(tmp_path/'RESULT.json')
    assert r['equal_rate_lgvsc_comparison'] is False
    assert r['arms']['LGVSC']['total_channel_uses']==10
    assert r['arms']['BSU-Gen/LTX-2.5']['total_channel_uses']==20
    assert r['independent_semantic_review']=='PENDING'


def test_no_complete_comparison_for_missing_clip(tmp_path):
    audit,_=fixture()
    with pytest.raises(ValueError,match='incomplete comparison'):
        comparison.report(tmp_path,audit,[],'test')
    assert not (tmp_path/'RESULT.json').exists()


@pytest.mark.parametrize('field,value',[('frames',8),('total_channel_uses',19)])
def test_no_equal_input_claim_after_changing_budget_or_time(tmp_path,field,value):
    audit,rows=fixture();rows[0][field]=value
    with pytest.raises(ValueError,match='complete BSU received timeline and rate'):
        comparison.report(tmp_path,audit,rows,'test')
    assert not (tmp_path/'RESULT.json').exists()


@pytest.mark.parametrize('code',[0,2])
def test_worker_progress_is_visible_and_failure_is_not_swallowed(tmp_path,capsys,code):
    cmd=[sys.executable,'-c',f'print("LTX intervals 1/2"); raise SystemExit({code})']
    log=tmp_path/'worker.log'
    if code:
        with pytest.raises(subprocess.CalledProcessError) as error:
            comparison.command(cmd,log,os.environ.copy())
        assert error.value.returncode==code
    else:
        comparison.command(cmd,log,os.environ.copy())
    assert 'LTX intervals 1/2' in capsys.readouterr().out
    assert log.read_text()=='LTX intervals 1/2\n'
