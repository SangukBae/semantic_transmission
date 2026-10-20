from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from semantic_transmission import bsu_budget_pilot as pilot, ntscc_plus_codec as codec
from semantic_transmission import ltx_conditioning as cond
from semantic_transmission.artifacts import write_json, sha256
from semantic_transmission.wire import pack, unpack
from semantic_transmission.packets import accounting


def test_sender_selection_uses_transient_change_and_preserves_semantic_keys():
    frames=np.zeros((17,80,144,3),dtype=np.uint8); frames[12]=255
    keys,audit=pilot.select_added_key(frames,[0,8,16],pilot.read(pilot.CONFIG))
    assert keys==[0,8,12,16] and audit['selected']['index']==12
    keys,_=pilot.select_added_key(np.zeros_like(frames),[0,8,16],pilot.read(pilot.CONFIG))
    assert keys==[0,4,8,16]  # deterministic tie
    with pytest.raises(ValueError,match='no interior candidate'):
        pilot.select_added_key(frames[:7],[0,3,6],pilot.read(pilot.CONFIG))


def test_caption_facts_negation_and_temporal_coverage_survive_remapping():
    rows=[dict(text='A person does not jump.',flow=1.),dict(text='A person turns left.',flow=3.)]
    divided=pilot.remap_segments([0,8,16],rows,[0,4,8,16])
    assert [r['text'] for r in divided]==[rows[0]['text'],rows[0]['text'],rows[1]['text']]
    merged=pilot.remap_segments([0,8,16],rows,[0,16])
    assert merged[0]['text']=='A person does not jump. A person turns left.'
    assert merged[0]['flow']==2.
    with pytest.raises(ValueError,match='not covered'):
        pilot.remap_segments([0,8,16],rows,[0,17])


def test_rate_search_uses_actual_total_cost_and_handles_infeasible_budget():
    def encode(eta):
        return dict(eta=eta,total_base_uses=2304+int(eta*1000)//4*4)
    best,trials=pilot.rate_search(encode,2505,.005,1.,14)
    assert best['total_base_uses']==2504 and len(trials)==16
    assert best==max((r for r in trials if r['total_base_uses']<=2505),key=lambda r:(r['total_base_uses'],r['eta']))
    with pytest.raises(ValueError,match='INFEASIBLE_REGISTERED_BUDGET'):
        pilot.rate_search(encode,100,.005,1.,14)
    def bad(_): raise RuntimeError('encoder failed')
    with pytest.raises(RuntimeError,match='encoder failed'):
        pilot.rate_search(bad,100,.005,1.,14)


def cpu_tensors(monkeypatch):
    original=torch.tensor
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self,*a,**k:self)
    def tensor(*a,**k):
        if k.get('device')=='cuda': k['device']='cpu'
        return original(*a,**k)
    monkeypatch.setattr(torch,'tensor',tensor)


def test_wire_cost_and_receive_need_only_received_files(tmp_path,monkeypatch):
    cpu_tensors(monkeypatch)
    source=tmp_path/'source'; source.mkdir()
    for i in (0,8,16): Image.new('RGB',(576,320),(i,0,0)).save(source/f'{i:05d}.png')
    def encode(model,x):
        assert model.eta==.3
        return torch.tensor([1+2j,3+4j]),torch.tensor([1,1]),.5
    monkeypatch.setattr(codec,'encode_symbols',encode)
    row=dict(video=dict(frames=17,fps=24,width=576,height=320),keys=[0,8,16],
        segments=[dict(text='A person moves.',flow=1.)]*2)
    result=pilot.encode_trial(SimpleNamespace(),row,[0,8,16],source,.3,tmp_path/'sent',pilot.read(pilot.CONFIG))
    packet=(tmp_path/'sent/metadata.bin').read_bytes()
    assert result['visual_uses']==6
    assert result['total_base_uses']==6+accounting(len(packet),bits_per_symbol=4)['complex_channel_uses']
    bare=tmp_path/'receiver_only'; (bare/'received').mkdir(parents=True)
    for name in ('metadata.bin','visual.c64'):
        (bare/'received'/name).write_bytes((tmp_path/'sent'/name).read_bytes())
    def decode(model,values,indices,power,height,width):
        assert values.tolist()==[1+2j,3+4j] and indices.tolist()==[1,1] and power==.5
        return torch.ones(1,3,height,width)*.5
    monkeypatch.setattr(codec,'decode_symbols',decode)
    pilot.receive_only(bare,SimpleNamespace())
    assert not (bare/'run_config.json').exists() and not (bare/'source').exists()
    audit=pilot.read(bare/'receiver_accounting.json')
    assert not audit['source_pixels_read'] and audit['keys']==3 and audit['eta']==.3
    assert np.asarray(Image.open(bare/'receiver/frames/sample/key_frames_received/8.png')).min()==128


@pytest.mark.parametrize('eta', [0.,2.,'0.2'])
def test_received_unregistered_rates_rejected_before_model_loading(tmp_path,eta):
    (tmp_path/'received').mkdir()
    (tmp_path/'received/metadata.bin').write_bytes(pack(dict(schema=pilot.SCHEMA,
        checkpoint_sha256=codec.CHECKPOINT_SHA256,visual_codec=dict(codec.SPEC,eta=eta))))
    with pytest.raises(ValueError,match='unknown received'):
        pilot.receive_only(tmp_path,SimpleNamespace())


def test_actual_channel_over_budget_stops_before_generation(tmp_path):
    dest=tmp_path/'transport'; run=dest/'run'; run.mkdir(parents=True)
    write_json(run/'channel_accounting.json',dict(total_complex_channel_uses=100))
    write_json(dest/'trajectory_received/channel.json',dict(complex_channel_uses=10))
    write_json(tmp_path/'transport_plan.json',dict(jobs=[dict(transport=str(dest),total_uses=110,budget=109)]))
    with pytest.raises(ValueError,match='budget violated'):
        pilot.prepare_requests(tmp_path,dict(seeds=[2025]))
    assert not (tmp_path/'jobs').exists()


def test_ltx_batch_resume_preserves_completed_jobs_and_intervals(tmp_path):
    keydir=tmp_path/'received_keys'; keydir.mkdir()
    for i in (0,4,8,16): Image.new('RGB',(576,320),(i,0,0)).save(keydir/f'{i}.png')
    jobs=[]
    for arm,keys in [('baseline',[0,8,16]),('more_keys',[0,4,8,16])]:
        packet=tmp_path/f'{arm}.bin'
        packet.write_bytes(pack(dict(video=dict(frames=17,fps=24,width=576,height=320),
            keyframes=[dict(index=i) for i in keys],segments=[dict(text='A person moves.',flow=1.)]*(len(keys)-1))))
        dest=tmp_path/'jobs'/arm
        request=cond.prepare(packet,keydir,dest/'conditions','context',seed=2025)
        write_json(dest/'request.json',request)
        jobs.append(dict(output=str(dest),id='case',arm=arm,seed=2025,request_sha256=sha256(dest/'request.json')))
    write_json(tmp_path/'generation_plan.json',dict(jobs=jobs))
    calls=[]
    class Generator:
        fail=True
        def __init__(self,assets): assert assets==dict(execution_offload='disk')
        def __call__(self,w,r):
            arm='more_keys' if len(r['receiver_improvements']['original_key_indices'])==4 else 'baseline'
            calls.append((arm,w['start']))
            if self.fail and w['start']==8: raise RuntimeError('GPU interrupted')
            yield from [np.zeros((320,576,3),dtype=np.uint8)]*w['generated_frames']
    p=dict(runtime={},policy=dict(execution_offload='disk'))
    with pytest.raises(RuntimeError,match='GPU interrupted'): pilot.generate_all(tmp_path,p,Generator)
    Generator.fail=False; pilot.generate_all(tmp_path,p,Generator)
    assert calls==[('baseline',0),('more_keys',0),('more_keys',8),('more_keys',8)]
    frames=tmp_path/'jobs/more_keys/raw/frames'
    assert len(list(frames.glob('*.png')))==17
    (frames/'00001.png').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='artifacts changed'): pilot.generate_all(tmp_path,p,Generator)


def test_progress_has_fixed_rows_failure_and_frozen_completed_elapsed(tmp_path):
    write_json(tmp_path/'status.json',dict(status='FAILED',stage='LTX generation',started_epoch=10,ended_epoch=72,
        jobs=6,evaluations=12,error='exit 1; see logs/generate.log'))
    lines=pilot.status_lines(tmp_path,telemetry=False)
    assert len(lines)==14 and 'FAILED' in lines[0]
    assert '0/6' in lines[5] and '0/unknown' in lines[6]
    assert '0:01:02' in lines[8] and 'ETA unknown' in lines[8]
    assert 'exit 1' in lines[-1] and 'hashes verified' in lines[10]


def test_frozen_input_mutation_stops_worker(tmp_path,monkeypatch):
    path=tmp_path/'input'; path.write_bytes(b'original')
    code=tmp_path/'code.py'; code.write_text('frozen code')
    monkeypatch.setattr(pilot,'code_identity',lambda:pytest.fail('worker imported optional environments'))
    p=dict(code={str(code):sha256(code)},inputs={str(path):sha256(path)})
    p['signature']=pilot.fingerprint(p); write_json(tmp_path/'protocol.json',p)
    pilot.verify_contract(tmp_path)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='input changed'): pilot.verify_contract(tmp_path)


def test_worker_rejects_changed_code_and_protocol(tmp_path):
    code=tmp_path/'code.py'; code.write_text('original')
    p=dict(code={str(code):sha256(code)},inputs={})
    p['signature']=pilot.fingerprint(p); write_json(tmp_path/'protocol.json',p)
    code.write_text('edited')
    with pytest.raises(ValueError,match='implementation changed'): pilot.verify_contract(tmp_path)
    p['code'][str(code)]=sha256(code); write_json(tmp_path/'protocol.json',p)
    with pytest.raises(ValueError,match='signature changed'): pilot.verify_contract(tmp_path)
