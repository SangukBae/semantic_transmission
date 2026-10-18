import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from semantic_transmission import ltx_receiver as receiver
from semantic_transmission.artifacts import sha256
from semantic_transmission.wire import pack


@pytest.mark.parametrize('count,padded',[(2,9),(8,9),(9,9),(10,17),(240,241),(241,241)])
def test_causal_padding_never_shortens_the_source(count,padded):
    assert receiver.padded_frames(count)==padded


@pytest.fixture
def received_request(tmp_path):
    keys=tmp_path/'received_keys';keys.mkdir()
    for i in (0,3,8): Image.new('RGB',(576,320),(i,20,30)).save(keys/f'{i}.png')
    packet=tmp_path/'received.bin'
    packet.write_bytes(pack(dict(video=dict(width=576,height=320,fps=24,frames=9),
        keyframes=[dict(index=i) for i in (0,3,8)],
        segments=[dict(text='A person moves.',flow=1.),dict(text='The person stops.',flow=2.)])))
    return receiver.request(packet,keys)


class FakeGenerator:
    calls=[]
    fail=False
    def __init__(self,_): pass
    def __call__(self,window,value):
        self.calls.append(window['start'])
        assert value['source_pixels_available'] is False
        assert {k for k in value['inputs']} == {str(Path(p).resolve()) for p in value['inputs']}
        for i in range(window['generated_frames']):
            if self.fail and window['start']==3 and i==2:
                raise RuntimeError('GPU worker interrupted')
            yield np.full((320,576,3),10+window['start']*10+i,dtype=np.uint8)


def test_received_only_temporal_join_tail_trim_and_resume(tmp_path,received_request):
    FakeGenerator.calls=[];FakeGenerator.fail=False
    root=tmp_path/'run'
    receiver.reconstruct(received_request,{},root,FakeGenerator)
    assert FakeGenerator.calls==[0,3]
    frames=sorted((root/'frames').glob('*.png'))
    assert len(frames)==9
    # The shared boundary uses the next interval's first frame, not both.
    assert [np.asarray(Image.open(p))[0,0,0] for p in frames]==[10,11,12,40,41,42,43,44,45]
    assert json.loads((root/'generation.json').read_text())['source_frames_read'] is False
    before=sha256(root/'reconstruction.mp4')
    receiver.reconstruct(received_request,{},root,FakeGenerator)
    assert FakeGenerator.calls==[0,3] and sha256(root/'reconstruction.mp4')==before
    frames[1].write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifacts changed'):
        receiver.reconstruct(received_request,{},root,FakeGenerator)


def test_interruption_preserves_completed_interval(tmp_path,received_request):
    FakeGenerator.calls=[];FakeGenerator.fail=True
    root=tmp_path/'run'
    with pytest.raises(RuntimeError,match='interrupted'):
        receiver.reconstruct(received_request,{},root,FakeGenerator)
    FakeGenerator.fail=False
    receiver.reconstruct(received_request,{},root,FakeGenerator)
    assert FakeGenerator.calls==[0,3,3]
    assert list((root/'failed_attempts').iterdir())


def test_corrupted_received_condition_refuses_generation(tmp_path,received_request):
    Path(received_request['windows'][0]['images'][0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='received condition changed'):
        receiver.reconstruct(received_request,{},tmp_path/'run',FakeGenerator)
    assert not (tmp_path/'run').exists()


def test_request_cannot_be_relabelled_as_a_different_timeline(received_request):
    received_request['video']['frames']=8
    with pytest.raises(ValueError,match='request or profile changed'):
        receiver.verify_request(received_request)
