import copy
import pytest

from semantic_transmission import adaptive_rate as a


def fixture(n=49):
    row=dict(psnr_db=30.,ssim=.9,clip=.98,lpips_vgg=.1,dists=.08)
    rows=[dict(frame=i,**row) for i in range(n)]
    q=dict(quality=row.copy(),nonkeys=row.copy(),boundary_error=.2,channel_uses=100000,
        temporal=dict(flow_error_mean=.5,cadence_error_mean=.4,false_freeze_rate=0.,active_motion_cells=1000,
            per_transition=dict(flow_error=[.5]*(n-1),cadence_error=[None]+[.4]*(n-2),false_freeze_cells=[0]*(n-1))))
    return q,rows


def candidate(q,rows,stride=4,qp=32):
    q=copy.deepcopy(q);q['channel_uses']=50000
    return dict(stride=stride,qp=qp,quality=q,frame_rows=copy.deepcopy(rows))


def test_continuous_schedule_boundaries_cut_and_variable_quantization():
    blocks=[dict(start=0,end=24,stride=4,qp=28),dict(start=24,end=48,stride=3,qp=31)]
    indices,qps=a.schedule(49,blocks,[9,10])
    assert indices[0]==0 and indices[-1]==48 and {9,10,24}<=set(indices)
    assert all(y-x<=4 for x,y in zip(indices,indices[1:]))
    assert dict(zip(indices,qps))[0]==32 and dict(zip(indices,qps))[24]==31
    assert {28,31,32}<=set(qps)


@pytest.mark.parametrize('blocks',[
    [dict(start=1,end=48,stride=2,qp=32)],
    [dict(start=0,end=48,stride=5,qp=32)],
    [dict(start=0,end=48,stride=2,qp=64)],
    [dict(start=0,end=24,stride=2,qp=32),dict(start=25,end=48,stride=2,qp=32)],
])
def test_invalid_schedules_fail_closed(blocks):
    with pytest.raises(ValueError):a.schedule(49,blocks)


def test_temporal_search_rejects_damaged_block_preserves_easy_block():
    q,rows=fixture();c=candidate(q,rows)
    for row in c['frame_rows'][24:]:row['dists']=.8
    blocks=a.allocate(q,rows,{'s4':c},49,[0,48],'temporal')
    assert [b['stride'] for b in blocks]==[4,2]


def test_ablations_do_not_mix_controls_and_rate_must_fall():
    q,rows=fixture();c=candidate(q,rows,stride=2,qp=28)
    assert all(b['qp']==32 for b in a.allocate(q,rows,{'qp':c},49,[0,48],'temporal'))
    assert all(b['qp']==28 for b in a.allocate(q,rows,{'qp':c},49,[0,48],'quantization'))
    c['quality']['channel_uses']=110000
    assert all(b['qp']==32 for b in a.allocate(q,rows,{'qp':c},49,[0,48],'joint'))


def test_motion_regression_rejected_despite_identical_static_quality():
    q,rows=fixture();c=candidate(q,rows)
    c['quality']['temporal']['per_transition']['flow_error']=[3.] * 48
    assert all(b['stride']==2 for b in a.allocate(q,rows,{'s4':c},49,[0,48],'joint'))


def test_rollback_terminates_without_mutating_previous_plan():
    blocks=[dict(start=x,end=x+24,stride=4,qp=28) for x in range(0,120,24)]
    original=copy.deepcopy(blocks);result=blocks
    for _ in range(4):result=a.rollback(result)
    assert all(b['stride']==2 and b['qp']==32 for b in result)
    assert blocks==original


def test_paid_wrapper_roundtrip_and_receiver_has_only_inner_stream():
    from semantic_transmission.bsu_adaptive_rate import wrapped
    from semantic_transmission.bsu_dense_neural import packet
    from semantic_transmission.bsu_dense_router import receive
    c=dict(video=dict(frames=4,width=64,height=32,fps=24),indices=[0,3],
        segments=[dict(path='sample/0_3.mp4',text='Object moves.',flow=1.)],decoder={})
    inner=packet(c,b'actual compressed bytes')
    mode,payload=receive(wrapped(inner))
    assert mode=='neural' and payload=={'neural':inner}


def test_trial_cache_validates_artifacts_and_source_before_reuse(tmp_path):
    import shutil
    from pathlib import Path
    from semantic_transmission import bsu_adaptive_rate as run
    from semantic_transmission.artifacts import write_json,sha256
    from semantic_transmission.webvid_ablation import snapshot
    source=tmp_path/'old';out=source/'clip/trials/s2_q31';out.mkdir(parents=True)
    saved=source/'validated_source';saved.mkdir()
    shutil.copy2(run.__file__,saved/'bsu_adaptive_rate.py')
    case=dict(id='clip',source=dict(normalized_sha256='source-hash'))
    blocks=[dict(start=0,end=2,stride=2,qp=31)]
    write_json(source/'protocol.json',dict(cases=[case],code={'codec':'pinned'}))
    write_json(out/'sender_job.json',dict(blocks=blocks))
    (out/'sender').mkdir();write_json(out/'sender/sender.json',{})
    (out/'reconstruction.mp4').write_bytes(b'completed video')
    write_json(out/'quality.json',dict(movie=str(out/'reconstruction.mp4'),
        movie_sha256=sha256(out/'reconstruction.mp4'),source_movie_sha256='source-hash'))
    (out/'quality.csv').write_text('frame,psnr_db\n0,30.0\n')
    for stage in ('sender','decode','restore','evaluate'):
        required=['reconstruction.mp4','quality.json','quality.csv']
        write_json(out/f'stages/{stage}.json',dict(status='PASSED',required=required,artifacts=snapshot(out,required)))
    experiment=run.Experiment.__new__(run.Experiment)
    experiment.reuse_trials=[source];experiment.p={'code':{'codec':'pinned'}}
    assert experiment.cached_trial(case,'s2_q31',blocks)['reused']
    (out/'reconstruction.mp4').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='artifacts changed'):
        experiment.cached_trial(case,'s2_q31',blocks)
    changed=copy.deepcopy(case);changed['source']['normalized_sha256']='other-source'
    with pytest.raises(ValueError,match='source contract differs'):
        experiment.cached_trial(changed,'s2_q31',blocks)
