"""Regression against the actual released aggregation function and real MP4 I/O."""
import argparse
import ast
import csv
import os
from pathlib import Path

import numpy as np
import pytest

from semantic_transmission import lgvsc_official as baseline, official_baseline_workers as workers
from semantic_transmission import official_quality, official_release as release
from semantic_transmission import official_reporting as report
from semantic_transmission.artifacts import sha256, write_json


def author_function(name):
    source=release.source_path('06_evaluation/final_score_calc_100.py').read_text()
    return next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)


@pytest.mark.parametrize('reference_count,generated_count',[(241,244),(241,238),(250,80),(80,90)])
def test_windows_and_same_index_intersection_match_executed_author_function(tmp_path,reference_count,generated_count):
    source=tmp_path/'source';gen=tmp_path/'gen';source.mkdir();gen.mkdir()
    for folder,count in ((source,reference_count),(gen,generated_count)):
        for i in range(count):(folder/f'{i}.png').touch()
    rows=[dict(frame=i,**{k:float((i+j)%17) for j,k in enumerate(official_quality.KEYS)})
          for i in range(min(reference_count,generated_count))]
    scope=dict(os=os,np=np)
    for name,key in report.METRIC_NAMES.items():
        fn='calculate_clip_similarity' if name=='clip' else f'calculate_{name}'
        scope[fn]=lambda a,b,key=key:rows[int(Path(a).stem)][key]
    node=author_function('calculate_average_scores')
    exec(compile(ast.Module(body=[node],type_ignores=[]),'pinned_author','exec'),scope)
    expected=scope['calculate_average_scores'](source,gen)
    assert report.author_summary(rows,reference_count)==expected
    if generated_count==80 and reference_count==250:
        assert expected['psnr_last100']==0 and expected['ssim_last100']==0


class PixelMetrics:
    """Actual PSNR/SSIM plus fixed neural scores; tests pairing without model loads."""
    def evaluate(self,source,reconstructed,progress=None):
        rows=[dict(frame=i,**official_quality.pixel_scores(a,b),lpips_vgg=.2,clip=.8,dists=.1)
              for i,(a,b) in enumerate(zip(source,reconstructed))]
        return {},rows


def test_evaluation_worker_uses_mp4_same_number_frames_not_boundary_remapping(tmp_path,monkeypatch):
    import cv2
    from semantic_transmission.research_quality import read_video
    source=tmp_path/'data/normalized.mp4';movie=tmp_path/'receiver/reconstruction/sample_0000.mp4'
    for path,indices in ((source,list(range(9))), (movie,[0,1,2,3,4,4,5,6,7,8])):
        path.parent.mkdir(parents=True)
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),24,(32,32))
        assert writer.isOpened()
        for i in indices:writer.write(np.full((32,32,3),i*25,dtype=np.uint8))
        writer.release()
    write_json(tmp_path/'keyframes.json',{'indices':[0,4,8]})
    monkeypatch.setattr(official_quality,'OfficialMetrics',PixelMetrics)
    workers.evaluate(dict(frames=9,width=32,height=32,fps=24),baseline.REPO,tmp_path)
    result=release.read(tmp_path/'quality.json')
    a,b=read_video(source),read_video(movie)
    _,expected_rows=PixelMetrics().evaluate(a,b[:9])
    assert result['official']==report.author_summary(expected_rows,9)
    assert result['compared_frames']==9 and result['excluded_output_frames']==1
    assert result['evaluation_profile']==report.EVALUATION_PROFILE
    with (tmp_path/'quality_delivered_mp4.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert float(rows[5]['psnr_db'])==official_quality.pixel_scores(a[5],b[5])['psnr_db']
    assert float(rows[5]['psnr_db'])!=official_quality.pixel_scores(a[4],b[5])['psnr_db']


def test_quality_table_uses_equal_video_weight_and_rejects_incomplete(tmp_path):
    plan=dict(videos=[dict(id='short'),dict(id='long')],cohort={'complete':False})
    for name,value,count in (('short',10.,10),('long',20.,200)):
        write_json(tmp_path/f'videos/{name}/run/quality.json',dict(status='PASSED',
                   evaluation_profile=report.EVALUATION_PROFILE,compared_frames=count,
                   official={k:value for k in report.AUTHOR_KEYS}))
    result=baseline.report_quality(tmp_path,plan)
    assert result['average']['psnr']==15 # not a frame-weighted mean
    with (tmp_path/'comparison_scores.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert [r['name'] for r in rows]==['short','long','Average']
    write_json(tmp_path/'videos/long/run/quality.json',dict(status='FAILED'))
    with pytest.raises(ValueError,match='quality evidence'):baseline.report_quality(tmp_path,plan)


def test_accounting_separates_measured_symbols_capacity_estimate_and_denominators():
    ntscc=dict(status='COMPLETE_OFFICIAL_NTSCC',visual_complex_channel_uses=100,
        rate_index_capacity_estimate_channel_uses=25.5,official_denominator=1000,
        normalized_source_denominator=2000,official_reference_cbr=.1255)
    result=report.rate_breakdown(ntscc,dict(status='PASSED',complex_channel_uses=2304))
    assert result['measured_simulation_channel_uses']==2404
    assert result['estimated_semantic_total_channel_uses']==2429.5
    assert result['measured_simulation_cbr']==2404/2000
    assert result['estimated_semantic_total_cbr_official_denominator']==2429.5/1000
    assert result['actual_complete_transmission_cbr'] is None
    with pytest.raises(ValueError,match='disagree'):
        report.rate_breakdown(dict(ntscc,official_reference_cbr=.13),dict(status='PASSED',complex_channel_uses=2304))


def cohort_args(**kwargs):
    return argparse.Namespace(**dict(dict(input=None,input_dir=None,cohort=None,allow_incomplete_cohort=False),**kwargs))


@pytest.mark.parametrize('name,count',[('webvid55',55),('kinetics14',14)])
def test_named_cohorts_require_every_manifest_hash_and_label_explicit_subsets(tmp_path,monkeypatch,name,count):
    raw=tmp_path/'raw';raw.mkdir()
    rows=[]
    for i in range(count):
        p=raw/f'{i}.mp4';p.write_bytes(f'raw {i}'.encode());rows.append(dict(filename=p.name,sha256=sha256(p)))
    manifest=tmp_path/'manifest.csv'
    baseline.write_csv(manifest,rows)
    monkeypatch.setattr(baseline,'source_path',lambda _:manifest)
    args=cohort_args(input_dir=raw,cohort=name)
    paths,cohort=baseline.resolve_inputs(args)
    assert len(paths)==count and cohort['complete'] and cohort['source_hashes_verified']
    paths[-1].unlink()
    with pytest.raises(ValueError,match='verified raw videos'):baseline.resolve_inputs(args)
    args.allow_incomplete_cohort=True
    paths,cohort=baseline.resolve_inputs(args)
    assert len(paths)==count-1 and not cohort['complete'] and cohort['expected_videos']==count
    assert cohort['missing_files']==[f'{count-1}.mp4']
    paths[0].write_bytes(b'wrong original')
    with pytest.raises(ValueError,match='source hash mismatch'):baseline.resolve_inputs(args)


def test_webvid_is_default_but_custom_inputs_never_claim_complete_cohort(tmp_path,monkeypatch):
    expected=release.source_path('docs/eval_manifest_webvid.csv')
    seen=[]
    def source(name):seen.append(name);return expected
    monkeypatch.setattr(baseline,'source_path',source)
    with pytest.raises(ValueError,match='webvid55'):baseline.resolve_inputs(cohort_args(input_dir=tmp_path))
    assert seen==['docs/eval_manifest_webvid.csv']
    raw=tmp_path/'raw.mp4';raw.touch()
    _,cohort=baseline.resolve_inputs(cohort_args(input=[raw]))
    assert cohort['name']=='custom' and not cohort['complete']
    with pytest.raises(ValueError,match='custom cohort'):
        baseline.resolve_inputs(cohort_args(input=[raw],cohort='webvid55'))
