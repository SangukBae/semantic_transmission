import copy
import sys
from pathlib import Path

import pytest

from semantic_transmission import bsu_default as default
from semantic_transmission import bsu_webvid as legacy
from semantic_transmission.artifacts import write_json
from semantic_transmission.webvid5 import fingerprint
from test_bsu_webvid import args_for


def plan(tmp_path,monkeypatch):
    args=args_for(tmp_path);args.action='check';args.mode=None;args.keep_going=False
    from semantic_transmission import qwen35_pairs_video
    monkeypatch.setattr(qwen35_pairs_video,'plan',lambda *a,**kw:None)
    monkeypatch.setattr(legacy,'settings',lambda _:dict(python=sys.executable,channel_python=sys.executable))
    return args,default.load_or_plan(args)


def test_new_default_declares_context_worker_code_and_parameters(tmp_path,monkeypatch):
    args,p=plan(tmp_path,monkeypatch)
    assert p['mode']=='combined_context' and p['policy']==default.POLICY
    assert p['receiver_refinement']['sender_mode']=='combined'
    assert p['receiver_refinement']['additional_context_transmission']==0
    assert all('src/semantic_transmission/'+n in p['code'] for n in ('bsu_default.py','bsu_receiver.py','receiver_key_context.py'))
    assert not args.root.exists()
    legacy.verify_plan(p)


def test_current_batch_routes_workers_and_restores_historical_runner_on_failure(monkeypatch):
    calls=[];old_code=legacy.CODE;old_policy=legacy.POLICY
    def command(cmd,*a):calls.append(cmd);raise RuntimeError('worker failure')
    monkeypatch.setattr(legacy,'command',command)
    with pytest.raises(RuntimeError,match='worker failure'):
        with default.current_pipeline():
            legacy.command(['python','-m','semantic_transmission.bsu_sequence_refinement','refine'],None,None)
    assert calls[0][2]=='semantic_transmission.bsu_receiver'
    assert legacy.CODE==old_code and legacy.POLICY==old_policy and legacy.command is command


def test_existing_protocol_preserves_mode_and_rejects_implicit_migration(tmp_path,monkeypatch):
    args,p=plan(tmp_path,monkeypatch)
    p['mode']='combined';p['policy']=legacy.POLICY
    p['signature']=fingerprint({k:v for k,v in p.items() if k!='signature'})
    write_json(args.root/'protocol.json',p)
    monkeypatch.setattr(legacy,'make_plan',lambda *a:pytest.fail('existing batch replanned'))
    assert default.load_or_plan(args)==p
    args.mode='combined_context'
    with pytest.raises(ValueError,match='frozen mode'):default.load_or_plan(args)


@pytest.mark.parametrize('field,value',[('snr',[6.]),('extra_rate_ratio',.1),('ids',['other'])])
def test_resume_rejects_explicit_changed_conditions(tmp_path,monkeypatch,field,value):
    args,p=plan(tmp_path,monkeypatch);write_json(args.root/'protocol.json',p)
    setattr(args,field,value)
    with pytest.raises(ValueError,match='frozen'):default.load_or_plan(args)


def test_completed_context_run_uses_receipts_without_starting_workers(tmp_path,monkeypatch):
    p=dict(policy=default.POLICY,root=str(tmp_path),signature='frozen',mode='combined_context',snr_db=[10.],videos=[dict(id='v')])
    monkeypatch.setattr(legacy,'load_result',lambda *a,**kw:dict(status='COMPLETE'))
    monkeypatch.setattr(legacy,'run_one',lambda *a:pytest.fail('completed receiver restarted'))
    monkeypatch.setattr(legacy,'report',lambda *a,**kw:dict(status='COMPLETE',pending=[]))
    assert default.run(p)==0


def test_shell_entrypoint_selects_current_runner():
    shell=(default.REPO/'scripts/run_bsu_webvid.sh').read_text()
    assert '-m semantic_transmission.bsu_default' in shell
    assert default.parser().parse_args(['check']).root==default.ROOT


def test_legacy_plan_keeps_original_worker_dispatch(monkeypatch):
    command=legacy.command
    monkeypatch.setattr(legacy,'run',lambda *a:legacy.command is command)
    assert default.run(dict(policy=legacy.POLICY))
