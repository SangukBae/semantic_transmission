import ast
import csv
from pathlib import Path
import sys

import pytest

from semantic_transmission import lgvsc_official as baseline, official_release as release
from semantic_transmission import official_baseline_workers as workers
from semantic_transmission.artifacts import write_json, sha256


@pytest.fixture
def cfg():
    return release.read(baseline.PROFILE)


@pytest.mark.parametrize('key,value', [
    ('append_actual_eof', True), ('short_reference_padding', True), ('condition_collision_repair', True),
    ('short_schedule_repair', True), ('reference_policy','last_17_frames_before_vae'),
    ('semantic_clip_policy','frame_exact'), ('concatenation_policy','endpoint_exact'),
    ('snr_db',8), ('internvl_8bit',True), ('candidate_interval_seconds',1),
    ('caption_provider','Qwen'), ('visual_codec','NTSCC+'), ('steps',10), ('paper_reproduction',True),
    ('evaluation_profile','lgvsc_official_metrics_v1'), ('caption_attention','eager'),
])
def test_research_settings_cannot_enter_official_baseline(cfg, key, value):
    baseline.validate_profile(cfg)
    cfg[key] = value
    with pytest.raises(ValueError):
        baseline.validate_profile(cfg)


def upstream(name):
    try:
        return release.source_path(name).read_text()
    except FileNotFoundError:
        pytest.skip('Run scripts/run_lgvsc_official.sh fetch for pinned-source regression tests')


def calls(source, name):
    return [ast.dump(node) for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name]


def test_decoder_uses_original_masks_schedule_and_concatenation(tmp_path):
    original = upstream(release.DECODER)
    modified = release.decoder_source(original)
    for function in ('apply_mask_strategy','prepare_multi_resolution_info','set_random_seed','fully_control_random_seed'):
        assert calls(original,function) == calls(modified,function)
    for forbidden in ('tail_reference','short_schedule','condition_collision','conditioning_indices','concatenate_segments','DiffusionModelReuse'):
        assert forbidden not in modified
    assert 'video[i] = video[i][:, dframe_to_frame(condition_frame_length) :]' in modified
    assert 'vae, video_clips[-1].to(device=device, dtype=dtype), refs' in modified
    assert 'model = (' in modified
    release.materialize(tmp_path/'release')
    assert (tmp_path/'release/mydemo_new_align_sh.patch').is_file()


def test_selector_keeps_generation_prompts_probability_math_and_last_candidate():
    original = upstream(release.SELECTOR)
    modified = release.selector_source(original)
    for function in ('custom_chat','load_image','dynamic_preprocess'):
        trees = [ast.parse(s) for s in (original,modified)]
        assert len({ast.dump(next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==function)) for t in trees}) == 1
    assert 'ratio = no_prob / yes_prob' in modified
    assert 'frame_file = all_frames[-1]' in modified
    assert 'max_new_tokens=1024, do_sample=False, output_scores=True' in modified
    assert 'cpu_layers=3, compact_cache=True' in modified
    with pytest.raises(ValueError,match='patch anchor'):
        release.selector_source(original.replace('    model.cuda()', '    model.cpu()'))


def test_cpu_t5_transfers_without_precision_change():
    import torch
    from types import SimpleNamespace
    values = {'y':torch.tensor([1.234567],dtype=torch.float32), 'mask':torch.tensor([1],dtype=torch.int64)}
    encoder = release.cpu_text_encoder(SimpleNamespace(encode=lambda _:values), 'cpu')
    result = encoder.encode(['text'])
    assert all(torch.equal(result[k],v) and result[k].dtype==v.dtype for k,v in values.items())


def test_whole_previous_reference_is_observed_without_crop(tmp_path):
    import torch
    seen=[]
    def original(vae, video, refs, strategies, loop, length, edit):
        seen.append(video)
        return [[torch.ones(1,5,1,1)]], strategies
    wrapper=release.trace_references(original,tmp_path/'trace.json')
    for count in (8,17,97):
        video=torch.ones(1,3,count,2,2)
        wrapper(None,video,[],[],1,5,0)
        assert seen[-1] is video and seen[-1].shape[2]==count
    assert [r['input_frames'] for r in release.read(tmp_path/'trace.json')]==[8,17,97]


def test_resume_reuses_completed_stage_but_restarts_failed_stage(tmp_path,monkeypatch,cfg):
    from semantic_transmission import lgvsc_qwen_baseline
    root=tmp_path/'out'
    plan=dict(signature='fixed',profile=cfg,models={},interpreters={'python':sys.executable,'channel_python':sys.executable})
    video=dict(id='v',input=str(tmp_path/'raw.mp4'))
    def init(dest,plan,video):
        write_json(dest/'run/request.json',{'input':'raw'})
        (dest/'run/release').mkdir();(dest/'run/release/decoder.py').write_text('pass')
    monkeypatch.setattr(baseline,'initialize',init)
    monkeypatch.setattr(baseline,'STAGES',('prepare','select'))
    monkeypatch.setattr(baseline,'stage_files',lambda stage:[f'run/{stage}.json'])
    calls=[]
    def launch(cmd,log,env,resource,**kwargs):
        stage=cmd[cmd.index('--worker')+1];calls.append(stage)
        write_json(root/f'videos/v/run/{stage}.json',{'attempt':len(calls)})
        if len(calls)==2: raise RuntimeError('original selector failed')
        write_json(resource,{'returncode':0})
    monkeypatch.setattr(lgvsc_qwen_baseline,'launch',launch)
    with pytest.raises(RuntimeError):baseline.run_one(root,plan,video,'run')
    before=(root/'videos/v/run/prepare.json').read_bytes()
    baseline.run_one(root,plan,video,'run')
    assert calls==['prepare','select','select']
    assert (root/'videos/v/run/prepare.json').read_bytes()==before
    assert len(list((root/'videos/v/failed_attempts').glob('select_*/run/select.json')))==1
    (root/'videos/v/run/prepare.json').write_text('corrupted')
    with pytest.raises(ValueError,match='artifacts changed'):baseline.run_one(root,plan,video,'run')


def test_snapshot_of_prepare_does_not_include_future_keyframes(tmp_path):
    from semantic_transmission.webvid_ablation import Stages
    def prepare(_):
        (tmp_path/'data').mkdir();(tmp_path/'data/frame.png').write_bytes(b'frame')
    stages=Stages(tmp_path,'id')
    stages.step('prepare',['data'],lambda:baseline.concrete_files(tmp_path,['data']),prepare)
    (tmp_path/'data/keys').mkdir();(tmp_path/'data/keys/0.png').write_bytes(b'key')
    Stages(tmp_path,'id').step('prepare',['data'],[],lambda _:pytest.fail('must reuse'))


def test_metadata_csv_preserves_comma_newline_unicode_and_fails_on_corruption(tmp_path,monkeypatch,cfg):
    from semantic_transmission import metadata_channel
    rows=[dict(path='clips/sample/00000.mp4',text='사람, walking\nwith a dog.',flow=1.234)]
    write_json(tmp_path/'metadata_tx.json',rows)
    def transmit(payload,snr,seed,**kwargs):
        assert (snr,seed,kwargs)==(10,1024,dict(k=6144,n=9216,bits_per_symbol=4))
        return payload,dict(complex_channel_uses=2304,bit_errors=0)
    monkeypatch.setattr(metadata_channel,'transmit',transmit)
    workers.channel(cfg,baseline.REPO,tmp_path)
    with (tmp_path/'received/received.csv').open(newline='') as stream:
        decoded=list(csv.DictReader(stream))
    assert decoded[0]['text']==rows[0]['text'] and float(decoded[0]['flow'])==rows[0]['flow']
    other=tmp_path/'bad';other.mkdir();write_json(other/'metadata_tx.json',rows)
    monkeypatch.setattr(metadata_channel,'transmit',lambda *a,**kw:(b'bad',dict(bit_errors=2)))
    with pytest.raises(ValueError,match='no fallback'):workers.channel(cfg,baseline.REPO,other)
    assert release.read(other/'metadata_channel.json')['status']=='FAILED'


def test_reference_rate_is_not_reported_as_complete_packet_cbr(tmp_path):
    plan=dict(signature='id',videos=[dict(id='v')],scope={},profile={'snr_db':10},
              cohort=dict(name='custom',expected_videos=1,complete=False))
    run=tmp_path/'videos/v/run'
    write_json(run/'ntscc.json',dict(status='COMPLETE_OFFICIAL_NTSCC',official_reference_cbr=.002,
                                   official_denominator=1000,normalized_source_denominator=2000,
                                   visual_complex_channel_uses=1,rate_index_capacity_estimate_channel_uses=1.0))
    write_json(run/'metadata_channel.json',dict(status='PASSED',complex_channel_uses=2304))
    result=baseline.report_rates(tmp_path,plan)
    assert result['mean_official_reference_cbr']==.002
    assert result['mean_measured_simulation_cbr']==2305/2000
    assert result['mean_estimated_semantic_total_cbr']==2306/2000
    assert result['actual_complete_transmission_cbr'] is None
    assert result['videos'][0]['caption_flow_ldpc_channel_uses']==2304
    with (tmp_path/'CBR_TABLE.csv').open() as stream:
        table=list(csv.DictReader(stream))
    assert table[0]['complete_cohort']=='False' and table[0]['snr_db']=='10'


def test_protects_sources_existing_experiments_and_repository(tmp_path):
    raw=tmp_path/'source/raw.mp4';raw.parent.mkdir();raw.write_bytes(b'video')
    for root in (raw,raw.parent,baseline.REPO,baseline.REPO/'src',baseline.REPO/'.local/x'):
        with pytest.raises(ValueError):baseline.protect(root,[raw])
    old=tmp_path/'old';write_json(old/'protocol.json',{})
    with pytest.raises(ValueError):baseline.protect(old/'new',[raw])
    baseline.protect(tmp_path/'new',[raw])


def test_modified_author_source_is_rejected(tmp_path):
    path=tmp_path/'.local/vendor/LGVSC-official/commit/script.py'
    path.parent.mkdir(parents=True);path.write_text('original')
    write_json(tmp_path/'configs/lgvsc_official_sources.json',dict(commit='commit',files={
        'script.py':dict(sha256=sha256(path))}))
    assert release.source_path('script.py',tmp_path)==path
    path.write_text('modified')
    with pytest.raises(ValueError,match='source changed'):
        release.source_path('script.py',tmp_path)


def test_modified_protocol_is_rejected_before_execution(tmp_path,cfg):
    plan=dict(profile=cfg,output=str(tmp_path))
    plan['signature']=baseline.signature(plan)
    plan['output']=str(tmp_path/'another')
    with pytest.raises(ValueError,match='signature/output'):
        baseline.validate_plan(plan,tmp_path)


def test_ntscc_executes_author_test_and_keeps_its_cbr_scope(tmp_path,monkeypatch,cfg):
    import math
    import torch
    from PIL import Image
    from types import SimpleNamespace
    from semantic_transmission import codec_transport
    upstream('03_jscc_transmission/ntscc/main_save.py')
    keys=tmp_path/'data/frames/sample'/cfg['method']
    keys.mkdir(parents=True)
    for i in (0,8): Image.new('RGB',(576,320),(i,20,30)).save(keys/f'{i}.png')
    write_json(tmp_path/'keyframes.json',dict(indices=[0,8]))
    checkpoint=tmp_path/'weights';checkpoint.write_bytes(b'fake weights for orchestration test')
    class Net:
        channel=SimpleNamespace(chan_param=10,forward=lambda x:(x,8))
        def eval(self): return self
        def __call__(self,x):
            self.channel.forward(x)
            one=torch.tensor(1.)
            return one,one,one,one,8/x.numel(),x,x
    monkeypatch.setattr(codec_transport,'load_codec',lambda repo:(Net(),checkpoint))
    monkeypatch.setattr(torch.Tensor,'cuda',lambda x,*a,**kw:x)
    workers.ntscc(dict(cfg,frames=10),baseline.REPO,tmp_path)
    result=release.read(tmp_path/'ntscc.json')
    assert result['visual_complex_channel_uses']==16
    assert result['official_reference_cbr']==pytest.approx((8/(3*576*320)+4/(16*16*3)/math.log2(11))*2/9)
    assert result['rate_index_information_bits']==5760
    assert result['rate_index_capacity_estimate_channel_uses']==pytest.approx(5760/math.log2(11))
    assert result['official_denominator']==3*576*320*9
    assert result['normalized_source_denominator']==3*576*320*10
    assert result['actual_complete_transmission_cbr'] is None
    assert len(list(keys.with_name(keys.name+'_10').glob('*.png')))==2


def test_selector_accepts_last_candidate_and_rejects_added_actual_eof(tmp_path,monkeypatch,cfg):
    frames=tmp_path/'data/frames/sample'/cfg['method']
    frames.mkdir(parents=True)
    monkeypatch.setattr(workers.subprocess,'run',lambda *a,**kw:None)
    for i in (0,20): (frames/f'{i}.png').write_bytes(b'key')
    current=dict(cfg,frames=24,candidate_indices=list(range(0,24,4)),models={'internvl':'local'})
    workers.select(current,baseline.REPO,tmp_path)
    assert release.read(tmp_path/'keyframes.json')['indices']==[0,20]
    (frames/'23.png').write_bytes(b'extra EOF')
    with pytest.raises(ValueError,match='candidate endpoints'):
        workers.select(current,baseline.REPO,tmp_path)


def test_official_extraction_retains_grid_without_eof(tmp_path):
    import cv2
    import numpy as np
    from semantic_transmission.workers import load_module
    upstream('01_data_prep/get_frame.py')
    movie=tmp_path/'source.mp4'
    writer=cv2.VideoWriter(str(movie),cv2.VideoWriter_fourcc(*'mp4v'),24,(32,32))
    assert writer.isOpened()
    for i in range(24): writer.write(np.full((32,32,3),i,dtype=np.uint8))
    writer.release()
    extractor=load_module(release.source_path('01_data_prep/get_frame.py'),'official_extract_test')
    extractor.extract_frames(str(movie),str(tmp_path/'frames'),.2)
    with (tmp_path/'frames/frames.csv').open() as stream:
        indices=[int(Path(r['frame_path']).stem) for r in csv.DictReader(stream)]
    assert indices==[0,4,8,12,16,20]
    assert not (tmp_path/'frames/23.png').exists()
