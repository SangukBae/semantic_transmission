import copy
import json
from pathlib import Path
import sys

import pytest

from semantic_transmission import bsu_kinetics as runner, kinetics_timesformer as action
from semantic_transmission.artifacts import sha256, write_json


def inputs(tmp_path, monkeypatch):
    raw=tmp_path/'raw';raw.mkdir()
    manifest=tmp_path/'manifest.csv';lines=['filename,kinetics_label_id,sha256']
    for i in range(14):
        p=raw/f'clip{i}.mp4';p.write_bytes(str(i).encode())
        lines.append(f'{p.name},{i},{sha256(p)}')
    manifest.write_text('\n'.join(lines)+'\n')
    monkeypatch.setattr(runner.assets,'MANIFEST_SHA',sha256(manifest))
    rows=runner.originals(raw,manifest)
    inventory=tmp_path/'inputs/inventory.json'
    for row in rows:
        row['dataset']='kinetics';row['source_sha256']=row['sha256']
        write_json(inventory.parent/'kinetics'/row['id']/'source_prepared.json',dict(
            source_sha256=row['sha256'],frames=240,fps=24,width=576,height=320,
            source_duration_seconds=10,recipe=dict(duration_limit=None,source_start_seconds=0,source_end='EOF')))
    write_json(inventory,dict(videos=rows))
    return raw,manifest,inventory


def test_exact_author_manifest_hash_rejects_altered_label(tmp_path,monkeypatch):
    raw,manifest,_=inputs(tmp_path,monkeypatch)
    manifest.write_text(manifest.read_text().replace('clip0.mp4,0,','clip0.mp4,1,'))
    with pytest.raises(ValueError,match='exact 14'):runner.originals(raw,manifest)


@pytest.mark.parametrize('mutation',['raw','missing','duplicate','label','duration','truncation'])
def test_rejects_wrong_bytes_labels_or_partial_timelines(tmp_path,monkeypatch,mutation):
    raw,manifest,inventory=inputs(tmp_path,monkeypatch)
    rows=runner.read(inventory)
    if mutation=='raw':(raw/'clip0.mp4').write_bytes(b'changed')
    elif mutation=='missing':rows['videos'].pop()
    elif mutation=='duplicate':rows['videos'][-1]=rows['videos'][0]
    elif mutation=='label':rows['videos'][0]['label']=77
    else:
        p=inventory.parent/'kinetics'/rows['videos'][0]['id']/'source_prepared.json'
        data=runner.read(p)
        if mutation=='duration':data['frames']=239-2
        else:data['recipe']['duration_limit']=10
        write_json(p,data)
    write_json(inventory,rows)
    with pytest.raises(ValueError):runner.cohort(inventory,manifest)


def test_plan_uses_current_context_worker_full14_and_only10db(tmp_path,monkeypatch):
    _,manifest,inventory=inputs(tmp_path,monkeypatch)
    from semantic_transmission import qwen35_pairs_video
    monkeypatch.setattr(qwen35_pairs_video,'plan',lambda *a,**kw:None)
    monkeypatch.setattr(runner.batch,'settings',lambda _:dict(python=sys.executable,channel_python=sys.executable))
    template=tmp_path/'template.json'
    write_json(template,dict(interpreters=dict(qwen=sys.executable),receiver=dict(seed=2025,channel_seed=42,steps=30)))
    previous=(runner.batch.CODE,runner.batch.POLICY,runner.batch.cohort,runner.batch.command)
    plan=runner.plan(tmp_path/'output',inventory,template,{},manifest)
    assert len(plan['videos'])==14 and plan['snr_db']==[10.]
    assert plan['mode']=='combined_context' and plan['policy']==runner.POLICY
    assert plan['receiver_refinement']['sender_mode']=='combined'
    assert plan['receiver_refinement']['additional_context_transmission']==0
    assert 'src/semantic_transmission/bsu_receiver.py' in plan['code']
    assert 'src/semantic_transmission/decoder_memory.py' in plan['code']
    assert (runner.batch.CODE,runner.batch.POLICY,runner.batch.cohort,runner.batch.command)==previous
    runner.batch.verify_plan(plan)
    assert runner.plan(tmp_path/'output',inventory,template,{},manifest)==plan
    with pytest.raises(ValueError,match='runtime/mode'):
        runner.plan(tmp_path/'output',inventory,template,{'changed':True},manifest)


def action_fixture(tmp_path):
    config=tmp_path/'config.yaml';config.write_text('fixed configuration')
    rows=[]
    for i in range(14):
        for arm in action.ARMS:
            path=tmp_path/f'{i}_{arm}.mp4';path.write_bytes(f'{i}_{arm}'.encode())
            rows.append(dict(id=f'video_{i}',variant=arm,label=i,path=str(path),sha256=sha256(path),frames=240))
    job=action.signed(dict(policy=action.POLICY,rows=rows,config=str(config),config_sha256=sha256(config),
        code={},output=str(tmp_path/'action'),runtime={}))
    for row in rows:
        # Alternate top-1 correctness, always top-5 correct.
        top5=[row['label'],100,101,102,103] if row['label']%2==0 else [100,row['label'],101,102,103]
        logits=[-100.]*400
        for n,klass in enumerate(top5):logits[klass]=float(5-n)
        write_json(action.record_path(job,row),action.signed(dict(job_signature=job['signature'],input=row,
            summed_logits=logits,top5=top5,crops=3,top1_correct=row['label']==top5[0],top5_correct=True)))
    return job


def test_full_action_summary_uses_video_denominator_and_integrity(tmp_path):
    job=action_fixture(tmp_path);action.validate_job(job)
    records=[action.load_record(job,row) for row in job['rows']]
    result=action.summary(records)
    assert result['original']['top1_percent']==50 and result['original']['top5_percent']==100
    assert result['bsu_context']['videos']==14
    with pytest.raises(ValueError,match='partial'):action.summary(records[:-1])
    p=action.record_path(job,job['rows'][0]);value=action.read(p);value['top1_correct']=False
    write_json(p,action.signed({k:v for k,v in value.items() if k!='signature'}))
    with pytest.raises(ValueError,match='accuracy'):action.load_record(job,job['rows'][0])


@pytest.mark.parametrize('mutation',['bytes','label','duplicates','record_signature','record_job'])
def test_action_resume_rejects_changed_input_or_prediction(tmp_path,mutation):
    job=action_fixture(tmp_path)
    if mutation=='bytes':Path(job['rows'][0]['path']).write_bytes(b'changed')
    elif mutation=='label':job['rows'][0]['label']=66
    elif mutation=='duplicates':job['rows'][-1]=job['rows'][0]
    else:
        path=action.record_path(job,job['rows'][0]);value=action.read(path)
        if mutation=='record_signature':value['signature']='wrong'
        else:
            value['job_signature']='wrong';value=action.signed({k:v for k,v in value.items() if k!='signature'})
        write_json(path,value)
        with pytest.raises(ValueError):action.load_record(job,job['rows'][0])
        return
    job=action.signed({k:v for k,v in job.items() if k!='signature'})
    with pytest.raises(ValueError):action.validate_job(job)


def test_completed_action_resume_does_not_load_model(tmp_path,monkeypatch):
    job=action_fixture(tmp_path)
    rows=[action.load_record(job,row) for row in job['rows']]
    result=action.signed(dict(status='COMPLETE_ACTION_EVALUATION',job_signature=job['signature'],summary=action.summary(rows),
        artifacts={str(action.record_path(job,r).relative_to(job['output'])):sha256(action.record_path(job,r)) for r in job['rows']}))
    write_json(Path(job['output'])/'RESULT.json',result)
    monkeypatch.setattr(action,'validate_runtime',lambda _:None)
    monkeypatch.setattr(action,'load_model',lambda *args:pytest.fail('completed result must not infer again'))
    before={p:p.stat().st_mtime_ns for p in Path(job['output']).rglob('*.json')}
    assert action.evaluate(job)==result
    assert all(p.stat().st_mtime_ns==t for p,t in before.items())


def test_classification_sums_three_views_and_rejects_replacement_clip():
    import torch
    logits=torch.zeros(3,400)
    logits[0,0]=10;logits[1,1]=9;logits[2,1]=9
    class Dataset:
        def __getitem__(self,i):return torch.zeros(3,8,224,224),7,i,{}
    sums,ranking=action.classify(lambda x:logits,Dataset(),0,7,'cpu')
    assert ranking[0]==1 and sums[1]==18
    class Replaced(Dataset):
        def __getitem__(self,i):return torch.zeros(3,8,224,224),7,3,{}
    with pytest.raises(ValueError,match='replacement'):action.classify(lambda x:logits,Replaced(),0,7,'cpu')


def test_finished_partial_download_is_promoted_without_network(tmp_path,monkeypatch):
    path=tmp_path/'weights';part=tmp_path/'weights.part';part.write_bytes(b'finished download')
    digest=sha256(part)
    monkeypatch.setattr(runner.assets.urllib.request,'urlopen',lambda *a,**k:pytest.fail('unnecessary network'))
    assert runner.assets.fetch('https://unused.example',path,digest)==path
    assert not part.exists() and sha256(path)==digest


def test_reconstruction_failure_preserves_completed_clips_and_restores_adapter(tmp_path,monkeypatch):
    root=tmp_path/'benchmark';root.mkdir()
    plan=dict(root=str(root),mode='combined_context',videos=[dict(id=f'v{i}') for i in range(14)])
    monkeypatch.setattr(runner.batch,'verify_plan',lambda _:None)
    monkeypatch.setattr(runner.batch,'load_result',lambda path,_:None if path.name=='v13' else dict(id=path.name))
    attempts=[];old_command=runner.batch.command
    def fail(plan,video,snr,path):
        attempts.append((video['id'],snr))
        assert runner.batch.command is not old_command
        raise RuntimeError('generator failed')
    monkeypatch.setattr(runner.batch,'run_one',fail)
    with pytest.raises(RuntimeError,match='generator failed'):runner.run(plan)
    state=runner.read(tmp_path/'status.json')
    assert state['status']=='FAILED' and state['reconstructed']==13
    assert attempts==[('v13',10.)] and runner.batch.command is old_command


def test_partial_action_resume_infers_only_missing_views(tmp_path,monkeypatch):
    import torch
    from types import SimpleNamespace
    job=action_fixture(tmp_path)
    missing=job['rows'][9];action.record_path(job,missing).unlink()
    kept=action.record_path(job,job['rows'][0]);before=kept.read_bytes(),kept.stat().st_mtime_ns
    monkeypatch.setattr(action,'validate_runtime',lambda _:None)
    monkeypatch.setattr(action,'configure',lambda *_:SimpleNamespace(RNG_SEED=0))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(action,'load_model',lambda *_:None)
    # Isolate heavy third-party imports; exercise the production resume loop.
    module=SimpleNamespace(Kinetics=lambda *args:list(range(84)))
    monkeypatch.setitem(sys.modules,'timesformer.datasets.kinetics',module)
    calls=[]
    def classify(model,dataset,index,label,device):
        calls.append(index)
        logits=[-1.]*400;top5=[label,100,101,102,103]
        for i,k in enumerate(top5):logits[k]=5.-i
        return logits,top5
    monkeypatch.setattr(action,'classify',classify)
    result=action.evaluate(job)
    assert calls==[9] and result['summary']['original']['videos']==14
    assert (kept.read_bytes(),kept.stat().st_mtime_ns)==before
    assert action.validate_evaluation(job)==result
