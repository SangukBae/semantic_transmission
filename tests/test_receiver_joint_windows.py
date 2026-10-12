import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from semantic_transmission.receiver_joint_windows import grid_plan,sample_windows


def test_grid_uses_the_same_vae_blocks_for_both_windows():
    plan=grid_plan(49,[0,24,48])
    assert plan['latent_frames']==14
    assert [a['latent'] for a in plan['anchors']]==[0,7,13]
    assert [(w['start'],w['stop'],w['first_frame'],w['frames']) for w in plan['windows']]==[(0,10,0,34),(5,14,17,32)]
    for anchor in plan['anchors']:
        block=next(b for b in plan['blocks'] if b['latent_start']<=anchor['latent']<b['latent_start']+b['latents'])
        lo=block['frame_start']+4*anchor['block_slot']-block['left_padding']
        assert lo<=anchor['frame']<=lo+3
    with pytest.raises(ValueError,match='collide'):
        grid_plan(17,[0,2,3,16])


def test_joint_averages_velocity_and_sequential_freezes_overlap():
    z=torch.zeros(1,1,3,1,1);fixed=torch.zeros(1,3,dtype=torch.bool)
    predict=lambda i,x,t,active:torch.full_like(x,1+2*i)
    windows=[(0,2),(1,3)];times=[torch.tensor([1000.])]
    joint=sample_windows(z,fixed,windows,times,predict,'joint')
    sequential=sample_windows(z,fixed,windows,times,predict,'sequential')
    assert joint.flatten().tolist()==[1,2,3]
    assert sequential.flatten().tolist()==[1,1,3]
    fixed[:,1]=True
    for mode in ['sequential','joint']:
        result=sample_windows(z,fixed,windows,times,predict,mode)
        assert result.flatten().tolist()==[1,0,3]


def test_joint_future_context_can_affect_earlier_window():
    z=torch.zeros(1,1,4,1,1);z[:,:,3]=4
    fixed=torch.zeros(1,4,dtype=torch.bool);fixed[:,3]=True
    times=[torch.tensor([1000.]),torch.tensor([500.])]
    predict=lambda i,x,t,active:torch.ones_like(x)*x.mean()
    sequential=sample_windows(z,fixed,[(0,3),(1,4)],times,predict,'sequential')
    joint=sample_windows(z,fixed,[(0,3),(1,4)],times,predict,'joint')
    assert sequential[0,0,0,0,0]==0
    assert joint[0,0,0,0,0]>0
    assert joint[0,0,3,0,0]==4


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_single_window_matches_pinned_rflow_binary_mask_math(dtype):
    path=Path(__file__).resolve().parents[1]/'.local/vendor/Open-Sora/opensora/schedulers/rf/__init__.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RFLOW')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='sample')
    ns={'torch':torch,'tqdm':lambda x:x}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),ns)
    def model(x,t,**kwargs):
        velocity=x.float()*.1+t[:,None,None,None,None]*.0001
        return torch.cat([velocity,torch.zeros_like(velocity)],1)
    encoder=SimpleNamespace(encode=lambda texts:{'y':torch.zeros(1,1)},null=lambda n:torch.zeros(n,1))
    scheduler=SimpleNamespace(cfg_scale=7.,num_sampling_steps=4,num_timesteps=1000,
        use_discrete_timesteps=False,use_timestep_transform=False,
        scheduler=SimpleNamespace(add_noise=lambda x,n,t:x+n))
    torch.manual_seed(8);z=torch.randn(1,2,7,2,3).to(dtype)
    fixed=torch.tensor([[True,False,False,True,False,False,True]])
    expected=ns['sample'](scheduler,model,encoder,z.clone(),['caption'],'cpu',mask=(~fixed).float(),progress=False)
    def predict(i,x,t,active):
        v=model(torch.cat([x,x]),t.repeat(2)).chunk(2,dim=1)[0]
        cond,uncond=v.chunk(2,dim=0)
        return uncond+7*(cond-uncond)
    times=[torch.tensor([float(t)]) for t in [1000,750,500,250]]
    rng=torch.get_rng_state().clone()
    for mode in ['sequential','joint']:
        actual=sample_windows(z,fixed,[(0,7)],times,predict,mode)
        assert torch.equal(actual,expected)
    assert torch.equal(rng,torch.get_rng_state())


def test_missing_coverage_invalid_times_and_nonfinite_prediction_fail():
    z=torch.zeros(1,1,3,1,1);fixed=torch.zeros(1,3,dtype=torch.bool)
    fn=lambda i,z,t,m:z
    with pytest.raises(ValueError,match='uncovered'):
        sample_windows(z,fixed,[(0,2)],[torch.tensor([1000.])],fn,'joint')
    with pytest.raises(ValueError,match='decrease'):
        sample_windows(z,fixed,[(0,3)],[torch.tensor([500.]),torch.tensor([600.])],fn,'joint')
    with pytest.raises(ValueError,match='velocity'):
        sample_windows(z,fixed,[(0,3)],[torch.tensor([1000.])],lambda *args:z+float('nan'),'joint')
