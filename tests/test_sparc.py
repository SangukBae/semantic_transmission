import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

from semantic_transmission import sparc as model
from semantic_transmission import sparc_semantics as semantics
from semantic_transmission import predictive_codec, dense_temporal_receiver, metadata_channel
from semantic_transmission import sparc_transport as transport
from semantic_transmission.artifacts import sha256
from semantic_transmission.packets import accounting
from semantic_transmission.wire import pack, unpack


def source(tmp_path, frames=9):
    tmp_path.mkdir(parents=True)
    for i in range(frames): Image.new('RGB', (16, 16), (i*2,)*3).save(tmp_path/f'{i}.png')
    return model.base.inspect_frames(tmp_path,24)


@pytest.fixture
def qwen_stub(monkeypatch):
    calls=[]; state={'interrupt':False}
    class Model:
        runtime=dict(model_id=semantics.MODEL_ID,revision=semantics.REVISION,quantization='nf4')
        def __init__(self,_):pass
        def generate(self,images,labels,prompt,tokens):
            ids=[int(label.split()[2].rstrip(',')) for label in labels]
            calls.append(ids)
            if state['interrupt'] and ids[-1]==8:
                state['interrupt']=False
                raise KeyboardInterrupt('simulated Qwen interruption')
            changed=ids[-1]==4
            value=dict(decision='changed' if changed else 'same',kind='pose' if changed else 'none',
                reference_state='Person standing',candidate_state='Person waving' if changed else 'Person standing',
                evidence_frames=[ids[0],ids[-1]])
            return dict(raw_text=json.dumps(value),model_runtime=self.runtime,truncated=False,seconds=.01)
    original=semantics.Session
    class Session(original):
        def __init__(self,*args,**kwargs):
            kwargs.setdefault('factory',Model);super().__init__(*args,**kwargs)
    monkeypatch.setattr(semantics,'Session',Session)
    return calls,state


def test_semantic_replay_uses_only_provenance_checked_calls(tmp_path,qwen_stub):
    frames=source(tmp_path/'frames');one=tmp_path/'one';one.mkdir()
    first=semantics.select(frames,model.profile(),one)
    assert first['keys']==[0,4,8] and first['new_model_calls']==2
    two=tmp_path/'two';two.mkdir()
    second=semantics.select(frames,model.profile(),two,one/'semantic.json')
    assert second['keys']==first['keys'] and second['new_model_calls']==0
    assert len(qwen_stub[0])==2
    file=next((one/'semantic_cache/calls').rglob('*.json'))
    file.write_text('{}')
    with pytest.raises(ValueError,match='artifact changed'):
        semantics.validate_manifest(one/'semantic.json',frames,model.profile())


def test_interrupted_qwen_resumes_completed_calls(tmp_path,qwen_stub):
    frames=source(tmp_path/'frames');out=tmp_path/'run';out.mkdir()
    qwen_stub[1]['interrupt']=True
    with pytest.raises(KeyboardInterrupt):semantics.select(frames,model.profile(),out)
    assert not (out/'semantic.json').exists()
    result=semantics.select(frames,model.profile(),out)
    assert result['new_model_calls']==1 and result['replayed_model_calls']==1
    assert qwen_stub[0].count([0,4])==1


def test_replay_rejects_different_pixels_or_stride(tmp_path,qwen_stub):
    frames=source(tmp_path/'frames');out=tmp_path/'run';out.mkdir()
    semantics.select(frames,model.profile(),out)
    Image.new('RGB',(16,16),(255,0,0)).save(tmp_path/'frames/2.png')
    altered=model.base.inspect_frames(tmp_path/'frames',24)
    with pytest.raises(ValueError,match='source/policy'):
        semantics.validate_manifest(out/'semantic.json',altered,model.profile())
    with pytest.raises(ValueError,match='source/policy'):
        semantics.validate_manifest(out/'semantic.json',frames,dict(model.profile(),candidate_stride=2))


def test_length_failure_repair_preserves_raw_calls_and_replays(tmp_path,monkeypatch):
    calls=[]
    class Model:
        runtime=dict(model_id=semantics.MODEL_ID,revision=semantics.REVISION,quantization='nf4')
        def __init__(self,_):pass
        def generate(self,images,labels,prompt,tokens):
            calls.append(prompt)
            concise='at most 16 words EACH' in prompt
            value=dict(decision='same',kind='none',reference_state='Person standing' if concise else 'visible '*30,
                candidate_state='Person standing' if concise else 'visible '*30,evidence_frames=[0,4])
            return dict(raw_text=json.dumps(value),model_runtime=self.runtime,truncated=False,seconds=.01)
    original=semantics.Session
    class Session(original):
        def __init__(self,*args,**kwargs):
            kwargs.setdefault('factory',Model);super().__init__(*args,**kwargs)
    monkeypatch.setattr(semantics,'Session',Session)
    frames=source(tmp_path/'frames',5);one=tmp_path/'one';one.mkdir()
    first=semantics.select(frames,model.profile(),one)
    assert first['keys']==[0,4] and len(calls)==7
    assert len(list((one/'semantic_cache/failed_attempts').rglob('*.json')))==6
    assert first['calls'][0]['role']=='semantic_select_concise'
    two=tmp_path/'two';two.mkdir()
    second=semantics.select(frames,model.profile(),two,one/'semantic.json')
    assert second['new_model_calls']==0 and len(calls)==7


@pytest.fixture
def codecs(monkeypatch):
    class Codec:
        def __init__(self,*_):pass
        def encode(self,frames,indices,qp):
            assert qp==32
            return pack(dict(transport.CODEC,indices=indices,height=frames.shape[1],width=frames.shape[2]),frames.tobytes())
        def decode(self,packet):
            h,p=unpack(packet)
            return h['indices'],np.frombuffer(p,np.uint8).reshape(-1,h['height'],h['width'],3)
    def interpolate(anchors,indices,frames,*_):
        out=np.empty((frames,*anchors.shape[1:]),np.uint8)
        for k,(a,b) in enumerate(zip(indices,indices[1:])):
            for i in range(a,b+1):
                out[i]=np.round(anchors[k].astype(float)*(b-i)/(b-a)+anchors[k+1].astype(float)*(i-a)/(b-a))
        return out
    monkeypatch.setattr(predictive_codec,'Codec',Codec)
    monkeypatch.setattr(dense_temporal_receiver,'interpolate',interpolate)
    monkeypatch.setattr(model,'configure_numerics',lambda:None)
    monkeypatch.setattr(metadata_channel,'transmit',lambda data,*a,**kw:
        (data,dict(**accounting(len(data)),bit_errors=0)))


def record(tmp_path,frames=None):
    return dict(output=str(tmp_path/'run'),signature='test-sparc',profile=model.profile(),
        source=frames or source(tmp_path/'source'),semantic_replay=None,
        runtimes={k:sys.executable for k in ('selector','codec','channel','receiver')})


def test_pipeline_restores_all_frames_and_verified_resume(tmp_path,qwen_stub,codecs,monkeypatch):
    request=record(tmp_path);calls=[]
    def launch(stage,job,*_):
        calls.append(stage);job=model.base.read(job)
        if stage in ('decode','restore'):
            assert 'source' not in job and 'profile' not in job
        model.worker(job,stage)
    monkeypatch.setattr(model,'launch',launch)
    model.execute(request);root=Path(request['output'])
    result=model.base.read(root/'RESULT.json')
    assert result['status']==model.COMPLETE and result['indices']==[0,4,8]
    assert result['channel_uses']==accounting((root/'sender/packet.bin').stat().st_size)['complex_channel_uses']
    assert model.base.read(root/'reconstruction/receiver.json')['video']['frames']==9
    before={str(p.relative_to(root)):sha256(p) for p in (root/'stages').glob('*.json')}
    model.execute(request)
    assert calls==list(model.STAGES[1:])
    assert before=={str(p.relative_to(root)):sha256(p) for p in (root/'stages').glob('*.json')}
    (root/'received/packet.bin').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='artifacts changed'):model.execute(request)
    assert not (root/'RESULT.json').exists()
    assert model.base.read(root/'status.json')['status']=='FAILED'


def test_failed_channel_never_calls_decoder(tmp_path,qwen_stub,codecs,monkeypatch):
    calls=[]
    def launch(stage,job,*_):calls.append(stage);model.worker(model.base.read(job),stage)
    monkeypatch.setattr(model,'launch',launch)
    monkeypatch.setattr(metadata_channel,'transmit',lambda data,*a,**kw:(data,dict(bit_errors=1)))
    request=record(tmp_path)
    with pytest.raises(ValueError,match='channel integrity'):model.execute(request)
    assert calls==['semantic','select','sender','channel']
    assert not (Path(request['output'])/'decoded').exists()


def test_pipeline_semantic_interrupt_keeps_call_cache(tmp_path,qwen_stub,codecs,monkeypatch):
    monkeypatch.setattr(model,'launch',lambda stage,job,*_:model.worker(model.base.read(job),stage))
    qwen_stub[1]['interrupt']=True;request=record(tmp_path)
    with pytest.raises(KeyboardInterrupt):model.execute(request)
    assert model.base.read(Path(request['output'])/'status.json')['status']=='INTERRUPTED'
    model.execute(request)
    assert qwen_stub[0].count([0,4])==1


def test_read_only_profile_and_status_without_assets(tmp_path,capsys,monkeypatch):
    monkeypatch.setattr(model,'preflight',lambda _:pytest.fail('unexpected preflight'))
    model.main(['profile']);assert json.loads(capsys.readouterr().out)['name']=='SPARC'
    model.main(['status','--output',str(tmp_path/'missing')])
    assert json.loads(capsys.readouterr().out)['status']=='NOT_STARTED'
    assert not (tmp_path/'missing').exists()
