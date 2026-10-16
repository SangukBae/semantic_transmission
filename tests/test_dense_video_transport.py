import copy
from pathlib import Path
import subprocess
import numpy as np
import pytest
from PIL import Image

from semantic_transmission import dense_video_transport as d
from semantic_transmission.dense_video_receiver import restore, detail_projection
from semantic_transmission.wire import pack, unpack


def context(n=4):
    return dict(video=dict(frames=n,width=64,height=32,fps=24),indices=[0,n-1],
        segments=[dict(path='sample/0_3.mp4',text='A red object moves.',flow=1.0)],decoder={})


def test_paid_context_roundtrip_and_corruption():
    c=context(); packet=d.encode_packet(c,b'actual encoded bytes',32,16)
    assert d.decode_packet(packet)[:2]==(c,b'actual encoded bytes')
    with pytest.raises(ValueError):d.decode_packet(packet[:-1]+bytes([packet[-1]^1]))
    h,p=unpack(packet);h['guide_sha256']='0'*64
    with pytest.raises(ValueError):d.decode_packet(pack(h,p))


def test_invalid_timeline_and_geometry():
    for keys in ([0,0,3],[0,2],[1,3]):
        c=context();c['indices']=keys
        with pytest.raises(ValueError):d.encode_packet(c,b'x',32,16)
    with pytest.raises(ValueError):d.encode_packet(context(),b'x',31,16)


def test_receiver_projection_preserves_timing_and_bounds():
    rng=np.random.default_rng(42)
    a=rng.integers(0,256,(3,32,64,3),dtype=np.uint8)
    b=rng.integers(0,256,a.shape,dtype=np.uint8)
    same, _=detail_projection(a,a)
    assert np.array_equal(same,a)
    result,audit=detail_projection(a,b)
    assert np.max(np.abs(result.astype(float)-a))<=3
    assert not audit['temporal_interpolation']
    assert np.array_equal(detail_projection(a,b,strength=0)[0],a)
    assert restore(a,(64,32)).shape==a.shape


def test_real_encoder_receiver_and_frame_count(tmp_path):
    source=tmp_path/'source';source.mkdir()
    for i in range(4):
        a=np.full((32,64,3),40,np.uint8);a[8:20,4+i*8:16+i*8]=[220,40,30]
        Image.fromarray(a).save(source/f'{i}.png')
    record=d.send(source,context(),tmp_path/'sender',uses_per_frame=4096,width=32,height=16)
    assert record['channel_uses']<=record['capacity_channel_uses']
    c,s,h=d.decode_packet((tmp_path/'sender/packet.bin').read_bytes())
    frames=d.decode_video(s,tmp_path/'receiver',c,h)
    assert frames.shape==(4,16,32,3)
    assert np.mean(np.abs(frames[0].astype(float)-frames[-1]))>1
    wrong=copy.deepcopy(c);wrong['video']['frames']=5
    with pytest.raises(ValueError):d.decode_video(s,tmp_path/'wrong',wrong,h)


def test_dense_guidance_respects_fixed_latents_and_cfg():
    import torch
    from semantic_transmission.dense_generation_runtime import corrected_velocity
    z=torch.zeros(2,4,7,2,2);out=torch.zeros(2,8,7,2,2)
    prior=torch.ones_like(z[:1]);free=torch.tensor([[0,0,0,0,0,1,0]])
    corrected=corrected_velocity(z,out,torch.tensor([500.,500.]),prior,free)
    assert torch.count_nonzero(corrected[:,:,:5])==0
    assert torch.count_nonzero(corrected[:,:,6:])==0
    assert torch.count_nonzero(corrected[:,4:])==0
    assert torch.all(corrected[:,:4,5]>0)
    assert torch.equal(corrected[0],corrected[1])
    # Adding the same residual to both branches adds exactly that residual
    # after CFG, instead of multiplying its strength by guidance scale.
    cfg=corrected[1,:4]+7*(corrected[0,:4]-corrected[1,:4])
    assert torch.equal(cfg,corrected[0,:4])
    assert torch.equal(corrected_velocity(z,out,torch.tensor([500.,500.]),prior,free,strength=0),out)


def test_learned_packet_roundtrip_and_wrong_stream_rejected():
    from semantic_transmission.bsu_dense_neural import packet,unpack_packet
    data=packet(context(),b'paid neural entropy stream')
    assert unpack_packet(data)==(context(),b'paid neural entropy stream')
    h,p=unpack(data);h['video_sha256']='f'*64
    with pytest.raises(ValueError):unpack_packet(pack(h,p))


def test_rate_cap_fails_closed_and_comparator_reports_miss(tmp_path,monkeypatch):
    def encode(source,frames,fps,w,h,crf,target):
        Path(target).write_bytes(b'x'*5000)
        return ['test encoder']
    monkeypatch.setattr(d,'encode_stream',encode)
    with pytest.raises(ValueError,match='cannot fit'):
        d.send(tmp_path,context(),tmp_path/'closed',uses_per_frame=1024,width=32,height=16)
    result=d.send(tmp_path,context(),tmp_path/'comparator',uses_per_frame=1024,width=32,height=16,
                  allow_budget_miss=True)
    assert not result['budget_met']
    assert result['channel_uses']>result['capacity_channel_uses']


def test_guide_vae_does_not_draw_noise():
    import torch
    from types import SimpleNamespace as NS
    from semantic_transmission.dense_generation_runtime import guide_latents
    class Posterior:
        def __init__(self,x):self.x=x
        def mode(self):return self.x
        def sample(self):raise AssertionError('guide must not consume random noise')
    spatial=NS(micro_batch_size=2,scaling_factor=2.,module=NS(encode=lambda x:NS(latent_dist=Posterior(x))))
    vae=NS(spatial_vae=spatial,micro_frame_size=3,temporal_vae=NS(encode=lambda x:Posterior(x)),shift=1.,scale=2.)
    x=torch.ones(1,3,7,4,4);before=torch.random.get_rng_state()
    actual=guide_latents(vae,x)
    assert torch.equal(actual,torch.full_like(x,.5))
    assert torch.equal(before,torch.random.get_rng_state())


def test_router_uses_only_paid_transmitter_cost():
    from semantic_transmission.bsu_dense_neural import packet
    from semantic_transmission.bsu_dense_router import choose,receive
    c=context()
    metadata=pack(dict(video=c['video'],keyframes=[{'index':i} for i in c['indices']],
                       segments=c['segments'],decoder=c['decoder']))
    neural=packet(context(),b'neural'+bytes(range(256))*4)
    p,decision=choose(metadata,None,100000,neural)
    assert decision['mode']=='neural'
    assert receive(p)==('neural',{'neural':neural})
    p,decision=choose(metadata,None,100000,neural,minimum_keys=5)
    assert decision['mode']=='bsu' and not decision['neural_eligible']
    assert receive(p)==('bsu',{'metadata':metadata})
    p,decision=choose(metadata,None,1,neural)
    assert decision['mode']=='bsu'
    assert receive(p)==('bsu',{'metadata':metadata})
    h,payload=unpack(p);h['lengths']=[1]
    with pytest.raises(ValueError):receive(pack(h,payload))
    other=context();other['segments'][0]['text']='A different video.'
    with pytest.raises(ValueError,match='different semantic'):
        choose(metadata,None,100000,packet(other,b'stream'))


def test_short_gap_indices_keep_endpoints_and_cut_brackets():
    from semantic_transmission.dense_temporal_receiver import select_indices,validate_indices
    images=np.zeros((8,8,8,3),dtype=np.uint8)
    assert select_indices(images)==[0,2,4,6,7]
    assert select_indices(images,4,protected=[1,5])==[0,1,4,5,7]
    with pytest.raises(ValueError):select_indices(images,4,protected=[8])
    images[3:]=255
    assert select_indices(images)==[0,2,3,4,6,7]
    validate_indices(select_indices(images),len(images),2)
    for bad in ([0,4,7],[0,2,4,6],[0,2,2,7],[-1,1,3,5,7]):
        with pytest.raises(ValueError):validate_indices(bad,8,2)


def test_routed_comparison_rejects_partial_cohort_before_reporting_rate(tmp_path,monkeypatch):
    from semantic_transmission import bsu_dense_validation as validation
    from semantic_transmission.bsu_dense_router import validate
    neural=tmp_path/'neural'
    values={neural/'RESULT.json':{'status':'COMPLETE_NEURAL_DENSE_RATE_QUALITY_ACTION'},
        neural/'protocol.json':{'scale':1,'cases':[{'id':'one'}]},
        neural/'action/RESULT.json':{},
        validation.REPO/'outputs/bsu_kinetics_context/RESULT.json':{'videos':[{'id':'one'},{'id':'two'}]}}
    monkeypatch.setattr(validation,'read',lambda p:values[p])
    with pytest.raises(ValueError,match='complete matched'):
        validate(tmp_path/'invalid_comparison',neural)
    assert not (tmp_path/'invalid_comparison').exists()
