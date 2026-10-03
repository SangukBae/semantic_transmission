"""Numeric scheduling and actual RFLOW sampling regression checks (CPU)."""
import sys
from pathlib import Path

import pytest
import torch

from semantic_transmission import short_schedule as fix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".local/vendor/Open-Sora"))
from opensora.schedulers import rf


def arguments(n, dtype=torch.bfloat16):
    return {k:torch.tensor([v],dtype=dtype) for k,v in dict(height=320,width=576,num_frames=n).items()}


@pytest.mark.parametrize("n", range(1,42))
def test_all_short_lengths_are_finite_and_other_lengths_identical(n):
    kw=arguments(n)
    times=[torch.tensor([1000*(1-i/30)]) for i in range(30)]
    repaired=[fix.transform(t,kw,rf.timestep_transform,num_timesteps=1000) for t in times]
    fix.validate_times(repaired)
    original=[rf.timestep_transform(t,kw,num_timesteps=1000) for t in times]
    if 1<n<17:
        assert not torch.isfinite(original[0]).all()
        assert all((t==0).all() for t in original[1:])
    else:
        assert all(torch.equal(a,b) for a,b in zip(repaired,original))


@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16,torch.float32])
def test_short_schedule_dtype_and_exact_latent_ratio(dtype):
    kw=arguments(9,dtype)
    t=torch.tensor([500.0])
    actual=fix.transform(t,kw,rf.timestep_transform,num_timesteps=1000)
    values={k:v.float() if dtype==torch.float16 else v for k,v in kw.items()}
    ratio=((values['height']*values['width'])/(512*512)).sqrt()*torch.tensor([3.0],dtype=values['height'].dtype).sqrt()
    unit=t/1000
    expected=ratio*unit/(1+(ratio-1)*unit)*1000
    assert torch.equal(actual,expected)


@pytest.mark.parametrize("frames", [[0],[-1],[float('nan')],[9.5],[9,17],[]])
def test_reject_invalid_or_mixed_lengths(frames):
    kw=arguments(9)
    kw['num_frames']=torch.tensor(frames)
    with pytest.raises(ValueError,match='uniform positive'):
        fix.transform(torch.tensor([500]),kw,rf.timestep_transform,num_timesteps=1000)


@pytest.mark.parametrize("values", [[1000,float('nan')],[1000,0],[100,200],[100,100]])
def test_fail_before_inference_on_invalid_time_grid(values):
    with pytest.raises(ValueError,match='invalid sampling'):
        fix.validate_times([torch.tensor([v]) for v in values])


def test_real_rflow_preserves_keys_updates_middle_and_consumes_same_noise(tmp_path,monkeypatch):
    class Encoder:
        def encode(self,text):return dict(y=torch.zeros(1,1,1))
        def null(self,n):return torch.zeros(n,1,1)
    class Model(torch.nn.Module):
        def forward(self,z,t,**kw):return torch.cat([torch.ones_like(z),torch.zeros_like(z)],dim=1)
    model,encoder=Model(),Encoder()
    z=torch.tensor([2.,-1.,4.]).reshape(1,1,3,1,1)
    mask=torch.tensor([[0.,1.,0.]])
    original_transform,original_sample=rf.timestep_transform,rf.RFLOW.sample
    records=fix.install([0,8],tmp_path/'trace.json')
    try:
        torch.manual_seed(123)
        scheduler=rf.RFLOW(num_sampling_steps=30,use_timestep_transform=True)
        actual=scheduler.sample(model,encoder,z.clone(),['caption'],'cpu',additional_args=arguments(9),mask=mask,progress=False)
        repaired_rng=torch.random.get_rng_state()
        assert actual.flatten()[0]==2 and actual.flatten()[2]==4
        assert actual.flatten()[1]!=-1
        fix.validate_trace(records,[0,8],30)
    finally:
        rf.timestep_transform,rf.RFLOW.sample=original_transform,original_sample
    torch.manual_seed(123)
    baseline=scheduler.sample(model,encoder,z.clone(),['caption'],'cpu',additional_args=arguments(9),mask=mask,progress=False)
    assert torch.equal(baseline,z)
    assert torch.equal(torch.random.get_rng_state(),repaired_rng)


def test_trace_rejects_missing_steps_or_no_updates():
    with pytest.raises(ValueError,match='incomplete'):
        fix.validate_trace([],[0,8],30)


def test_check_does_not_create_outputs(tmp_path,monkeypatch,capsys):
    from semantic_transmission import schedule_repair as run
    dest=tmp_path/'untouched'
    monkeypatch.setattr(run,'preflight',lambda *_:({'schedule_plan':[]},'signature'))
    monkeypatch.setattr(run,'execute',lambda *_:pytest.fail('check ran model'))
    run.main(['--check','--output',str(dest)])
    assert not dest.exists()
    assert '"model_inference_started": false' in capsys.readouterr().out


def test_runner_rejects_source_overwrite(tmp_path):
    from semantic_transmission.schedule_repair import preflight
    source=tmp_path/'source'
    for output in (source,source/'child',tmp_path):
        with pytest.raises(ValueError,match='separate'):
            preflight(source,output)
