import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from semantic_transmission import sparc_motion as pilot
from semantic_transmission import sparc_motion_cache as cache
from semantic_transmission import sparc_motion_transport as transport
from semantic_transmission import sparc_motion_receiver as receiver
from semantic_transmission import sparse_trajectory as sparse
from semantic_transmission import predictive_codec, metadata_channel, official_quality, stutter_metrics
from semantic_transmission.artifacts import sha256, write_json
from semantic_transmission.packets import accounting
from semantic_transmission.wire import pack, unpack
from semantic_transmission.webvid_ablation import snapshot


def key_packet(frames=33, payload=b'x'*80000):
    video=dict(frames=frames,width=16,height=16,fps=24.)
    codec=pack(dict(transport.CODEC,indices=[0,frames-1],height=16,width=16),payload)
    return transport.make_keys(video,codec)


def window(frames=33):
    points=np.array([[40,40],[140,40],[240,40],[40,120],[140,120],[240,120]],np.float32)
    points=np.concatenate([points,points])
    times=np.array([0,frames//2,frames-1])
    xy=np.repeat(points[None],3,axis=0)
    valid=np.ones((3,12),bool)
    return dict(start=0,end=frames-1,trajectory=sparse.encode(xy,valid,np.repeat([0,1],6),times,frames))


def test_separate_long_gap_contract_and_packet_binding():
    packet=key_packet();row=window()
    motion=transport.make_motion(packet,[row])
    assert transport.parse_motion(motion,packet)[0,32]['frames']==33
    with pytest.raises(ValueError,match='64'):key_packet(66)
    with pytest.raises(ValueError,match='checksum'):transport.parse_motion(motion[:-1]+bytes([motion[-1]^1]),packet)
    with pytest.raises(ValueError,match='match'):transport.parse_motion(motion,key_packet(payload=b'different'))
    with pytest.raises(ValueError,match='duplicate'):transport.make_motion(packet,[row,row])
    with pytest.raises(ValueError,match='interval'):transport.make_motion(packet,[dict(row,end=31)])


def test_budget_counts_actual_ldpc_blocks_and_thinning_preserves_times():
    packet=key_packet();row=window()
    motion,audit=cache.choose_motion(packet,[row],pilot.profile())
    assert accounting(len(motion))['complex_channel_uses']<=audit['cap']
    assert not audit['quality_used']
    original=sparse.decode(row['trajectory'],33)
    reduced=sparse.decode(cache.thin([row],8,2)[0]['trajectory'],33)
    assert set(reduced['times'])<=set(original['times'])
    assert len(reduced['origins'])<=8
    with pytest.raises(ValueError,match='budget'):
        cache.choose_motion(packet,[row],dict(pilot.profile(),side_information_ratio_cap=0))
    assert cache.auxiliary_schedule([0,8,12],2)==[0,2,4,8,12]


def test_received_visibility_gates_correction_and_preserves_anchors():
    import cv2
    rng=np.random.default_rng(3)
    key=cv2.GaussianBlur(rng.integers(0,256,(160,288,3),dtype=np.uint8),(5,5),1)
    blurred=cv2.GaussianBlur(key,(15,15),4)
    frames=np.repeat(blurred[None],9,axis=0);frames[[0,-1]]=key
    track=sparse.decode(window(9)['trajectory'],9)
    track['valid'][:]=False
    anchors=np.stack([key,key])
    masked,_=receiver.correct(frames,anchors,[0,8],{(0,8):track})
    unmasked,audit=receiver.correct(frames,anchors,[0,8],{(0,8):track},visibility=False)
    assert np.array_equal(masked,frames)
    assert audit['corrected_frames']>0 and not audit['source_pixels_used']
    assert np.array_equal(unmasked[[0,-1]],anchors)


def test_legacy_reuse_accepts_different_png_encoding_but_rejects_changed_pixels(tmp_path):
    old=tmp_path/'legacy';oldframes=old/'source_frames';oldframes.mkdir(parents=True)
    new=tmp_path/'new';new.mkdir()
    for i in range(3):
        image=Image.new('RGB',(16,16),(i*40,20,10))
        image.save(oldframes/f'{i}.png',compress_level=0)
        image.save(new/f'{i}.png',compress_level=9)
    source=pilot.base.inspect_frames(new,24)
    prepared=old/'source_prepared.json'
    write_json(prepared,dict(source['video'],source_frames=str(oldframes),
        source_frame_hashes={str(i):sha256(oldframes/f'{i}.png') for i in range(3)}))
    case=tmp_path/'case';case.mkdir();write_json(case/'job.json',dict(source_frames=str(oldframes)))
    base=case/'base/reconstruction/event_pairs/run/transmitter/metadata.bin';base.parent.mkdir(parents=True)
    base.write_bytes(pack(dict(video=source['video'],keyframes=[dict(index=0),dict(index=2)])))
    sender=case/'sender';sender.mkdir()
    sender.joinpath('trajectory.bin').write_bytes(pack(dict(policy='bsu_multisegment_trajectory_v1',
        video=source['video'],base_metadata_sha256=sha256(base),windows=[window(3)])))
    write_json(case/'stages/prepare.json',dict(status='PASSED',required=['sender'],artifacts=snapshot(case,['sender'])))
    rows,audit=cache.audit_legacy(source,[0,2],case,prepared)
    assert len(rows)==1 and audit['rgb_identical_frames']==3 and audit['png_byte_identical_frames']==0
    assert audit['new_tracking_calls']==0
    Image.new('RGB',(16,16),(255,0,0)).save(new/'1.png')
    changed=pilot.base.inspect_frames(new,24)
    with pytest.raises(ValueError,match='pixels differ'):cache.audit_legacy(changed,[0,2],case,prepared)


def test_pipeline_receiver_inputs_accounting_and_verified_resume(tmp_path,monkeypatch):
    folder=tmp_path/'frames';folder.mkdir()
    for i in range(9):Image.new('RGB',(64,64),(i*20,40,60)).save(folder/f'{i}.png')
    source=pilot.base.inspect_frames(folder,24)
    root=tmp_path/'run';calls=[]
    def prepare(record):
        write_json(root/'source.json',source);write_json(root/'semantic.json',dict(keys=[0,8]))
        write_json(root/'semantic_cache/calls.json',dict(reused=True))
        write_json(root/'legacy_windows.json',[window(9)]);write_json(root/'reuse.json',dict(qwen_new_calls=0))
    class Codec:
        def __init__(self,*_):pass
        def encode(self,frames,indices,qp):
            return pack(dict(transport.CODEC,indices=indices,height=64,width=64),frames.tobytes())
        def decode(self,packet):
            h,p=unpack(packet);return h['indices'],np.frombuffer(p,np.uint8).reshape(-1,64,64,3)
    def interpolate(anchors,indices,video,*_):
        result=np.empty((video['frames'],64,64,3),np.uint8)
        for k,(a,b) in enumerate(zip(indices,indices[1:])):
            for i in range(a,b+1):result[i]=np.rint(anchors[k]*(float(b-i)/(b-a))+anchors[k+1]*(float(i-a)/(b-a)))
        return result
    class Metrics:
        def evaluate(self,a,b,progress):
            assert a.shape==b.shape
            return dict(psnr_db=30.),[dict(frame=i,psnr_db=30.) for i in range(len(a))]
    def launch(stage,path,*_):
        calls.append(stage);job=pilot.base.read(path)
        if stage in ('channel','decode','restore'):assert 'source' not in job
        pilot.worker(stage,job)
    monkeypatch.setattr(pilot,'prepare',prepare);monkeypatch.setattr(pilot,'launch',launch)
    monkeypatch.setattr(predictive_codec,'Codec',Codec);monkeypatch.setattr(receiver,'interpolate',interpolate)
    monkeypatch.setattr(pilot.sparc,'configure_numerics',lambda:None)
    monkeypatch.setattr(metadata_channel,'transmit',lambda data,*a,**kw:(data,dict(accounting(len(data)),bit_errors=0)))
    monkeypatch.setattr(official_quality,'OfficialMetrics',Metrics)
    monkeypatch.setattr(stutter_metrics,'evaluate',lambda a,b:dict(test_stub=True))
    record=dict(output=str(root),signature='test-motion',source=source,profile=pilot.profile(),historical=None,
        runtimes={k:sys.executable for k in ('codec','channel','receiver')})
    pilot.execute(record)
    result=pilot.base.read(root/'RESULT.json');comparison=pilot.base.read(root/'comparison.json')
    assert result['status']==pilot.COMPLETE
    assert comparison['same_key_bitstream_for_keys_and_motion']
    assert comparison['results']['motion']['channel_uses']>comparison['results']['keys']['channel_uses']
    assert comparison['results']['auxiliary']['channel_uses']<=comparison['results']['motion']['channel_uses']
    receipts={p.name:sha256(p) for p in (root/'stages').glob('*.json')}
    pilot.execute(record)
    assert calls==list(pilot.STAGES[1:])
    assert receipts=={p.name:sha256(p) for p in (root/'stages').glob('*.json')}
    (root/'received/motion.bin').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='artifacts changed'):pilot.execute(record)
    assert not (root/'RESULT.json').exists()
    assert pilot.base.read(root/'status.json')['status']=='FAILED'
