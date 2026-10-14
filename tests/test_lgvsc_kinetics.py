from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from semantic_transmission import lgvsc_kinetics as run, lgvsc_kinetics_action as action
from semantic_transmission import kinetics_timesformer as base
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.packets import accounting
from semantic_transmission.wire import pack


def source_fixture(tmp_path, frames=5):
    folder=tmp_path/'source'; folder.mkdir()
    for i in range(frames): (folder/f'{i}.png').write_bytes(f'frame{i}'.encode())
    movie=tmp_path/'normalized.mp4'; movie.write_bytes(b'original bytes')
    prepared=tmp_path/'prepared.json'
    source=dict(source_frames=str(folder),normalized_video=str(movie),normalized_sha256=sha256(movie),frames=frames)
    write_json(prepared,source)
    return dict(id='v1',prepared=str(prepared),frames=frames,config={}),source


@pytest.mark.parametrize('frames,last,count',[(240,239,61),(213,212,54),(2,1,2)])
def test_candidates_preserve_full_source_without_inventing_key_budget(frames,last,count):
    c=run.candidates(frames)
    assert c[0]==0 and c[-1]==last and len(c)==count and c==sorted(set(c))


def test_config_keeps_lgvsc_models_and_matches_actual_bsu_transport(tmp_path):
    shared=dict(width=576,height=320,fps=24,steps=30,seed=2025,channel_seed=42,snr_db=10,
        models={'internvl':'InternVL','pllava':'PLLaVA'},flash_attn=True,decoder_policy='official_release',
        concatenation_policy='endpoint_exact',semantic_clip_policy='frame_exact',
        visual_channel='source_frame_seeded_numpy_awgn_v1',evaluation_profile='lgvsc_official_metrics_v1',
        caption_provider='Qwen',caption_adapter={'path':'must not inherit'})
    cfg=run.config(shared,dict(frames=213,normalized_video='same.mp4',normalized_sha256='hash'),tmp_path)
    assert cfg['selector']=='skem' and cfg['threshold']==.35 and not cfg['internvl_8bit']
    assert cfg['paper_caption'] and cfg['paper_flow'] and cfg['frames']==213
    assert 'caption_adapter' not in cfg and 'caption_provider' not in cfg
    assert all(cfg[k]==shared[k] for k in run.COMMON)
    assert cfg['benchmark_digital_bits_per_symbol']==4
    assert 'key_budget' not in cfg and 'max_gap' not in cfg


def test_initialize_links_exact_bsu_bytes_and_does_not_leave_placeholder_selection(tmp_path):
    video,source=source_fixture(tmp_path,6); dest=tmp_path/'out'
    before={p:sha256(p) for p in Path(source['source_frames']).iterdir()}
    run.initialize(dest,video)
    assert not (dest/'run/keyframes.json').exists()
    assert (dest/'run/data/normalized.mp4').resolve()==Path(source['normalized_video'])
    csv=(dest/'run/data/frames/sample/frames.csv').read_text()
    assert '/0.png' in csv and '/4.png' in csv and '/5.png' in csv and '/1.png' not in csv
    assert all(sha256(p)==s for p,s in before.items())


def transport_fixture(tmp_path):
    cfg=dict(width=576,height=320,fps=24,frames=5)
    keys=[dict(index=i,rate_bytes=1,average_power=1.,complex_count=2,complex_offset=2*n,
               rate_offset=n,rate_count=1) for n,i in enumerate([0,4])]
    header=dict(video=cfg,keyframes=keys,segments=[dict(text='A person walks.',flow=1.,path='sample/0')],decoder={})
    packet=pack(header,b'\x00\x00')
    tx=tmp_path/'transmitter'; tx.mkdir()
    (tx/'metadata.bin').write_bytes(packet); (tx/'visual.c64').write_bytes(b'\x00'*32)
    report=dict(accounting(len(packet)),metadata_exact_match=True,bit_errors=0)
    write_json(tmp_path/'channel_accounting.json',report)
    return dict(id='v1',config=cfg),header,report


def test_rate_counts_symbols_and_fec_not_complex64_storage_bytes(tmp_path):
    video,_,_=transport_fixture(tmp_path)
    run.write_rate(tmp_path,video,'signature')
    rate=run.read(tmp_path/'rate.json')
    assert rate['total_channel_uses']==4+2304
    assert rate['cbr_denominator']==3*576*320*5
    assert rate['ledger']['ldpc_bits']['padding']>0
    assert rate['ledger']['ldpc_bits']['parity']==3072


@pytest.mark.parametrize('mutation',['tail','modulation','errors','one_key','byte_tail'])
def test_rate_rejects_different_timeline_link_or_corruption(tmp_path,mutation):
    video,header,report=transport_fixture(tmp_path)
    if mutation=='tail': header['keyframes'][-1]['index']=3
    elif mutation=='one_key': header['keyframes']=header['keyframes'][:1]
    elif mutation=='modulation': report['bits_per_symbol']=2
    elif mutation=='errors': report['bit_errors']=1
    else: (tmp_path/'transmitter/visual.c64').write_bytes(b'bad')
    (tmp_path/'transmitter/metadata.bin').write_bytes(pack(header,b'\x00\x00'))
    write_json(tmp_path/'channel_accounting.json',report)
    with pytest.raises(ValueError): run.write_rate(tmp_path,video,'signature')
    assert not (tmp_path/'rate.json').exists()


def test_pooled_comparison_uses_all_frames_and_rejects_partial_cohort():
    videos=[dict(id=f'v{i}',frames=213 if i==9 else 240,width=576,height=320,bsu_channel_uses=20) for i in range(14)]
    rows=[dict(id=v['id'],frames=v['frames'],cbr_denominator=3*576*320*v['frames'],total_channel_uses=10) for v in videos]
    value=run.aggregate_rate(dict(videos=videos),rows)
    assert value['bsu_over_lgvsc']==2 and value['bsu_change_percent']==100
    assert value['denominator']==3*576*320*3333
    with pytest.raises(ValueError): run.aggregate_rate(dict(videos=videos),rows[:-1])
    rows[0]['frames']-=1
    with pytest.raises(ValueError): run.aggregate_rate(dict(videos=videos),rows)


def test_output_cannot_overlap_bsu_sources(tmp_path):
    old=tmp_path/'bsu'; old.mkdir()
    for dest in (old,old/'child',tmp_path):
        with pytest.raises(ValueError): run.protect(dest,old)
    run.protect(tmp_path/'lgvsc',old)


def test_rate_stages_resume_and_reject_changed_completed_source(tmp_path,monkeypatch):
    video,source=source_fixture(tmp_path)
    root=tmp_path/'out'; p=dict(signature='fixed',interpreters=dict(python=sys.executable,channel_python=sys.executable))
    video['config']['seed']=2025
    monkeypatch.setattr(run,'jobs',lambda:[('rate',['rate.json'])])
    calls=[]
    def launch(command,log,env,resource,**kw):
        calls.append(command)
        write_json(root/'videos/v1/run/rate.json',dict(id='v1',status='COMPLETE_TRANSMISSION'))
        write_json(resource,dict(returncode=0))
    monkeypatch.setattr(run.legacy,'launch',launch)
    run.run_one(root,p,video,stop_after_rate=True)
    run.run_one(root,p,video,stop_after_rate=True)
    assert len(calls)==1
    (Path(source['source_frames'])/'2.png').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='artifacts changed'):run.run_one(root,p,video,stop_after_rate=True)


def test_failed_stage_preserves_evidence_and_reuses_completed_initialization(tmp_path,monkeypatch):
    video,_=source_fixture(tmp_path);video['config']['seed']=2025
    root=tmp_path/'out';p=dict(signature='fixed',interpreters=dict(python=sys.executable,channel_python=sys.executable))
    monkeypatch.setattr(run,'jobs',lambda:[('rate',['rate.json'])])
    calls=[]
    def launch(command,log,env,resource,**kw):
        calls.append(command); dest=root/'videos/v1/run/rate.json'
        write_json(dest,dict(status='partial' if len(calls)==1 else 'COMPLETE_TRANSMISSION'))
        if len(calls)==1:raise RuntimeError('failed worker')
        write_json(resource,dict(returncode=0))
    monkeypatch.setattr(run.legacy,'launch',launch)
    with pytest.raises(RuntimeError):run.run_one(root,p,video,stop_after_rate=True)
    init=root/'videos/v1/stages/initialize.json'; before=init.read_bytes()
    run.run_one(root,p,video,stop_after_rate=True)
    assert init.read_bytes()==before
    archived=list((root/'videos/v1/failed_attempts').glob('rate_*/run/rate.json'))
    assert len(archived)==1 and run.read(archived[0])['status']=='partial'


def action_fixture(tmp_path,monkeypatch):
    old=tmp_path/'old.json';old.write_text('{}')
    rows=[]
    for i in range(14):
        p=tmp_path/f'v{i}.mp4';p.write_bytes(str(i).encode())
        rows.append(dict(id=f'v{i}',label=i,frames=240,variant='lgvsc',path=str(p),sha256=sha256(p)))
    originals=[dict(r,variant='original') for r in rows]
    original_read=action.read
    monkeypatch.setattr(action,'read',lambda p:dict(rows=originals,runtime={},evaluation={}) if Path(p)==old else original_read(p))
    monkeypatch.setattr(base,'validate_evaluation',lambda _:None)
    monkeypatch.setattr(base,'validate_runtime',lambda _:None)
    job=base.signed(dict(policy=action.POLICY,rows=rows,runtime={},evaluation={'reconstruction':'delivered LGVSC matched MP4'},code={},output=str(tmp_path/'action'),
        bsu_action_job=str(old),bsu_action_job_sha256=sha256(old),config=str(base.CONFIG),config_sha256=sha256(base.CONFIG)))
    for row in rows:
        ranking=[row['label'],100,101,102,103];logits=[-10.]*400
        for i,k in enumerate(ranking):logits[k]=5.-i
        write_json(base.record_path(job,row),base.signed(dict(job_signature=job['signature'],input=row,crops=3,
            summed_logits=logits,top5=ranking,top1_correct=True,top5_correct=True)))
    return job


def test_complete_action_resume_verifies_results_without_model(tmp_path,monkeypatch):
    job=action_fixture(tmp_path,monkeypatch)
    records=[base.load_record(job,r) for r in job['rows']]
    result=base.signed(dict(status='COMPLETE_ACTION_EVALUATION',job_signature=job['signature'],summary=action.summary(records),
        artifacts={str(base.record_path(job,r).relative_to(job['output'])):sha256(base.record_path(job,r)) for r in job['rows']}))
    write_json(Path(job['output'])/'RESULT.json',result)
    monkeypatch.setattr(base,'load_model',lambda *_:pytest.fail('must reuse verified completed inference'))
    assert action.evaluate(job)==result
    job['rows'][0]['label']=99
    with pytest.raises(ValueError):action.validate_job(job)


def test_action_partial_resume_infers_only_missing_video(tmp_path,monkeypatch):
    import torch
    job=action_fixture(tmp_path,monkeypatch);missing=job['rows'][7]
    base.record_path(job,missing).unlink()
    kept=base.record_path(job,job['rows'][0]);before=kept.read_bytes(),kept.stat().st_mtime_ns
    monkeypatch.setattr(base,'configure',lambda *_:SimpleNamespace(RNG_SEED=0))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(base,'load_model',lambda *_:None)
    monkeypatch.setitem(sys.modules,'timesformer.datasets.kinetics',SimpleNamespace(Kinetics=lambda *_:list(range(42))))
    calls=[]
    def classify(model,dataset,index,label,device):
        calls.append(index);logits=[-10.]*400;ranking=[label,100,101,102,103]
        for i,k in enumerate(ranking):logits[k]=5.-i
        return logits,ranking
    monkeypatch.setattr(base,'classify',classify)
    assert action.evaluate(job)['summary']['top1_percent']==100
    assert calls==[7] and (kept.read_bytes(),kept.stat().st_mtime_ns)==before


@pytest.mark.parametrize('mutation',['label','frames','duplicate','input','variant'])
def test_action_rejects_mismatched_bsu_comparison(tmp_path,monkeypatch,mutation):
    job=action_fixture(tmp_path,monkeypatch)
    if mutation in ('label','frames'):job['rows'][0][mutation]+=1
    elif mutation=='duplicate':job['rows'][-1]=job['rows'][0]
    elif mutation=='variant':job['rows'][0]['variant']='bsu_context'
    else:Path(job['rows'][0]['path']).write_bytes(b'changed')
    job=base.signed({k:v for k,v in job.items() if k!='signature'})
    with pytest.raises(ValueError):action.validate_job(job)
