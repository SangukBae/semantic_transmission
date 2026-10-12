import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from semantic_transmission import bsu_webvid as batch
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.official_quality import KEYS


def inputs(tmp_path):
    manifest=tmp_path/'manifest.csv'
    manifest.write_text('filename,sha256\na.mp4,'+'a'*64+'\nb.mp4,'+'b'*64+'\n')
    inventory=tmp_path/'inventory.json'
    row=dict(id='webvid_001',dataset='WebVid',source_sha256='a'*64,source_path='/raw/a.mp4')
    write_json(inventory,dict(videos=[row]))
    prepared=tmp_path/'webvid/webvid_001/source_prepared.json'
    write_json(prepared,dict(source_sha256='a'*64,frames=48,fps=24,width=576,height=320,
        source_duration_seconds=2,recipe=dict(duration_limit=None,source_start_seconds=0,source_end='EOF')))
    return inventory,manifest,prepared


def test_manifest_subset_is_explicit_and_strict_mode_fails(tmp_path):
    inventory,manifest,_=inputs(tmp_path)
    videos,coverage=batch.cohort(inventory,manifest)
    assert len(videos)==1 and coverage['expected']==2 and coverage['missing']==['b.mp4']
    assert not coverage['full_manifest_selected'] and not coverage['exact_paper_reproduction']
    with pytest.raises(ValueError,match='full manifest'):batch.cohort(inventory,manifest,require_complete=True)


@pytest.mark.parametrize('mutation',['clip','duration','identity','duplicate','path','ids'])
def test_rejects_partial_or_wrong_dataset_membership(tmp_path,mutation):
    inventory,manifest,prepared=inputs(tmp_path)
    value=json.loads(prepared.read_text());rows=json.loads(inventory.read_text())
    if mutation=='clip':value['recipe']['duration_limit']=2
    elif mutation=='duration':value['frames']=24
    elif mutation=='identity':rows['videos'][0]['source_sha256']='c'*64
    elif mutation=='duplicate':rows['videos']*=2
    elif mutation=='path':rows['videos'][0]['id']='../escape'
    write_json(prepared,value);write_json(inventory,rows)
    with pytest.raises(ValueError):batch.cohort(inventory,manifest,ids=['missing'] if mutation=='ids' else None)


def args_for(tmp_path):
    inventory,manifest,_=inputs(tmp_path)
    template=tmp_path/'template.json'
    write_json(template,dict(interpreters=dict(qwen=sys.executable),receiver=dict(seed=2025,channel_seed=42,steps=30)))
    return SimpleNamespace(inventory=inventory,manifest=manifest,template=template,root=tmp_path/'output',
        ids=None,require_complete_manifest=False,reuse_base=None,snr=[6.,8.,10.],mode='combined',extra_rate_ratio=.05)


def test_check_calls_actual_source_preflight_without_creating_outputs(tmp_path,monkeypatch):
    args=args_for(tmp_path);calls=[]
    from semantic_transmission import qwen35_pairs_video
    monkeypatch.setattr(qwen35_pairs_video,'plan',lambda *a,**kw:calls.append((a,kw)))
    monkeypatch.setattr(batch,'settings',lambda _:dict(python=sys.executable,channel_python=sys.executable))
    plan=batch.make_plan(args)
    assert len(calls)==1 and not args.root.exists()
    assert plan['mode']=='combined' and plan['snr_db']==[6.,8.,10.]


@pytest.mark.parametrize('snr',[[float('nan')],[float('inf')],[10.,10.],[6.,6.0000001]])
def test_rejects_ambiguous_snr_sweeps(tmp_path,snr):
    args=args_for(tmp_path);args.snr=snr
    with pytest.raises(ValueError):batch.make_plan(args)


def row(frames,score,uses):
    quality={b:dict(frames=frames,**{k:score for k in KEYS}) for b in ['lossless_frames','delivered_mp4']}
    return dict(frames=frames,base_quality=quality,refined_quality=copy.deepcopy(quality),
        base_channel_uses=uses,total_channel_uses=uses+2,cbr_denominator=frames*100)


def test_aggregation_does_not_confuse_video_frame_or_rate_weighting():
    result=batch.aggregate([row(10,1,100),row(30,3,200)])
    q=result['refined']['delivered_mp4']
    assert q['video_mean']['psnr_db']==2
    assert q['frame_mean']['psnr_db']==2.5
    assert result['refined']['pooled_cbr']==304/4000
    assert result['refined']['video_mean_cbr']==pytest.approx((102/1000+202/3000)/2)
    assert batch.aggregate([]) is None


def test_nonfinite_or_partial_frame_scores_cannot_enter_table():
    r=row(10,1,100);r['refined_quality']['delivered_mp4']['frames']=9
    with pytest.raises(ValueError):batch.aggregate([r])
    r=row(10,float('nan'),100)
    with pytest.raises(ValueError):batch.aggregate([r])


def test_resume_verifies_output_and_base_hashes(tmp_path):
    artifact=tmp_path/'output.mp4';artifact.write_bytes(b'video')
    base=tmp_path/'base.mp4';base.write_bytes(b'base')
    write_json(tmp_path/'job.json',dict(base_artifacts={str(base):sha256(base)}))
    r=dict(status='COMPLETE_RECONSTRUCTION_METRICS_REVIEW_PENDING',signature='id',artifacts={'output.mp4':sha256(artifact)})
    write_json(tmp_path/'RESULT.json',r)
    assert batch.load_result(tmp_path,dict(signature='id'))==r
    base.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='base artifact'):batch.load_result(tmp_path,dict(signature='id'))


def test_report_incomplete_sweep_keeps_missing_jobs_out_of_means(tmp_path):
    plan=dict(root=str(tmp_path),signature='id',snr_db=[6.,10.],videos=[dict(id='a'),dict(id='b')],mode='combined',
        coverage=dict(selected=2,expected=55),paper=dict(skem_dsa_cbr_table_i={'6':.00064,'10':.00065},
        comparison_limits=['subset'],quality_reference_note='no exact published quality table',source='https://arxiv.org/html/2606.12899v2'))
    summary=batch.report(plan)
    assert summary['status']=='PARTIAL' and len(summary['pending'])==4
    assert summary['by_snr']=={'6':None,'10':None}
    assert len((tmp_path/'per_video.csv').read_text().splitlines())==1


def test_batch_resume_does_not_start_completed_jobs(tmp_path,monkeypatch):
    plan=dict(root=str(tmp_path),signature='id',snr_db=[10.],videos=[dict(id='a')],mode='combined')
    monkeypatch.setattr(batch,'load_result',lambda *a,**kw:dict(status='COMPLETE'))
    monkeypatch.setattr(batch,'run_one',lambda *a:pytest.fail('completed run was restarted'))
    monkeypatch.setattr(batch,'report',lambda *a,**kw:dict(status='COMPLETE',pending=[]))
    assert batch.run(plan)==0
    assert json.loads((tmp_path/'status.json').read_text())['completed']==1


def test_batch_failure_is_visible_and_completed_work_remains_resumable(tmp_path,monkeypatch):
    plan=dict(root=str(tmp_path),signature='id',snr_db=[10.],videos=[dict(id='a')],mode='combined',
        coverage=dict(selected=1,expected=55),paper=batch.read(batch.REPO/'configs/bsu_webvid_paper.json'))
    monkeypatch.setattr(batch,'load_result',lambda *a,**kw:None)
    def fail(*args):raise RuntimeError('GPU worker failed')
    monkeypatch.setattr(batch,'run_one',fail)
    with pytest.raises(RuntimeError,match='GPU worker failed'):batch.run(plan)
    status=json.loads((tmp_path/'status.json').read_text())
    assert status['status']=='FAILED' and status['completed']==0
    assert 'GPU worker failed' in status['failures'][0]['error']
    assert json.loads((tmp_path/'SUMMARY.json').read_text())['status']=='PARTIAL'


@pytest.mark.parametrize('snr,bits',[(6,2),(8,2),(10,4)])
def test_link_adapter_keeps_snr_seed_and_restores_frozen_transport(tmp_path,monkeypatch,snr,bits):
    from semantic_transmission import bsu_webvid_base as adapter
    from semantic_transmission import metadata_channel, qwen35_pairs_comparison as base
    write_json(tmp_path/'run_config.json',dict(snr_db=snr,channel_seed=42,benchmark_digital_bits_per_symbol=bits))
    calls=[]
    def transmit(packet,snr_db,seed,**kw):
        calls.append((packet,snr_db,seed,kw));return 'receipt'
    monkeypatch.setattr(metadata_channel,'transmit',transmit)
    monkeypatch.setattr(base,'paired_channel',lambda run:metadata_channel.transmit(b'paid',snr,42))
    assert adapter.paired_channel(tmp_path)=='receipt'
    assert calls==[(b'paid',snr,42,dict(bits_per_symbol=bits))]
    assert metadata_channel.transmit is transmit


def test_benchmark_wrapper_routes_every_worker_through_declared_adapter(monkeypatch):
    from semantic_transmission import bsu_webvid_base as adapter, qwen35_pairs_video as video
    old_module,old_worker=video.MODULE,video.worker
    observed=[]
    monkeypatch.setattr(adapter,'paired_channel',lambda path:observed.append(path))
    def main():
        assert video.MODULE==adapter.MODULE
        video.worker(Path('/run'),'channel')
    monkeypatch.setattr(video,'main',main)
    try:adapter.main()
    finally:video.MODULE=old_module;video.worker=old_worker
    assert observed==[Path('/run/reconstruction/event_pairs/run')]
