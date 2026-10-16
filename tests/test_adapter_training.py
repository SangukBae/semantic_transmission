import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from semantic_transmission import adapter_training_state as state
from semantic_transmission.adapter_training_data import fixed_latents, source_roles, DEFAULT_CONFIG
from semantic_transmission import condition_collision

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'.local/vendor/Open-Sora'))


def config():return state.validate_config(json.loads(DEFAULT_CONFIG.read_text()))


def test_budget_exhaustion_is_not_convergence():
    c=config();plateau=dict(best=1.,bad=0,evaluations=1)
    assert state.stop_reason(c['min_steps'],c,plateau,plateau) is None
    assert state.stop_reason(c['max_steps'],c,plateau,plateau)=='MAX_STEPS_WITHOUT_CONFIRMED_PLATEAU'
    plateau['bad']=20
    assert state.stop_reason(c['min_steps']-1,c,plateau,plateau) is None
    assert state.stop_reason(c['min_steps'],c,plateau,plateau)=='VALIDATION_PLATEAU'


def test_loss_plateau_alone_cannot_stop_before_quality_checks():
    c=config();p=dict(best=1.,bad=99,evaluations=100)
    assert state.stop_reason(c['min_steps'],c,p,None) is None
    assert state.stop_reason(c['min_steps'],c,p,dict(p,bad=0)) is None


def test_relative_plateau_and_nonfinite_rejection():
    p,good=state.update_plateau(None,2.,.01);assert good
    p,good=state.update_plateau(p,1.99,.01);assert not good and p['bad']==1
    p,good=state.update_plateau(p,1.95,.01);assert good and p['bad']==0
    with pytest.raises(ValueError):state.update_plateau(p,float('nan'),.01)


def test_atomic_checkpoint_survives_failed_write(tmp_path,monkeypatch):
    state.commit_checkpoint(tmp_path,dict(step=1,weight=torch.ones(2)),'s')
    head=(tmp_path/'latest.json').read_bytes()
    def fail(*a,**kw):raise OSError('simulated disk failure')
    monkeypatch.setattr(torch,'save',fail)
    with pytest.raises(OSError):state.commit_checkpoint(tmp_path,dict(step=2),'s')
    assert (tmp_path/'latest.json').read_bytes()==head
    assert state.load_checkpoint(tmp_path,'s')['step']==1


def test_checkpoint_hash_and_protocol_rejected(tmp_path):
    head=state.commit_checkpoint(tmp_path,dict(step=1,weight=torch.ones(2)),'s')
    with pytest.raises(ValueError):state.load_checkpoint(tmp_path,'other')
    Path(head['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError):state.load_checkpoint(tmp_path,'s')


def test_optimizer_and_rng_replay_is_exact(tmp_path):
    torch.manual_seed(12);rng=np.random.default_rng(32)
    model=torch.nn.Linear(3,2);opt=torch.optim.AdamW(model.parameters(),lr=.001)
    def update():
        opt.zero_grad();x=torch.randn(4,3)*rng.random();model(x).square().mean().backward();opt.step()
    update()
    state.commit_checkpoint(tmp_path,dict(step=1,weights=model.state_dict(),optimizer=opt.state_dict(),
        cpu_rng=torch.get_rng_state(),numpy_rng=json.dumps(rng.bit_generator.state)),'s')
    update();expected=copy.deepcopy(model.state_dict())
    saved=state.load_checkpoint(tmp_path,'s');model.load_state_dict(saved['weights']);opt.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['cpu_rng']);rng.bit_generator.state=json.loads(saved['numpy_rng']);update()
    for k,v in expected.items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)


def test_native_end_alignment_and_generated_history_are_used():
    f=condition_collision.upstream_functions(REPO);rows=condition_collision.preview([0,17,33],f)
    row=rows[1];e=dict(segment=1,latent_frames=row['latent_frames'],mask_plan=row)
    target=torch.zeros(1,4,10,2,2);keys=[torch.ones(1,4,1,2,2)*2,torch.ones(1,4,1,2,2)*3]
    history=torch.ones(1,4,5,2,2)*9
    fixed,mask=fixed_latents(e,keys,history,target)
    assert torch.all(fixed[:,:,:5]==9) and torch.all(fixed[:,:,5]==3)
    assert mask.tolist()==[[False]*6+[True]*4]
    assert torch.count_nonzero(fixed[:,:,-1])==0  # release alignment is not simply -1


def test_first_segment_collision_repair_preserves_distinct_keys():
    f=condition_collision.upstream_functions(REPO);row=condition_collision.preview([0,16],f)[0]
    e=dict(segment=0,latent_frames=5,mask_plan=row)
    keys=[torch.ones(1,4,1,2,2)*2,torch.ones(1,4,1,2,2)*3]
    fixed,mask=fixed_latents(e,keys,None,torch.zeros(1,4,5,2,2))
    assert fixed[0,0,:,0,0].tolist()==[2,0,0,0,3]
    assert mask.tolist()==[[False,True,True,True,False]]


def test_previous_evaluation_is_development_not_new_test(monkeypatch):
    from semantic_transmission import adapter_training_data as data
    monkeypatch.setattr(data.common,'read',lambda p:dict(examples=[dict(role='train',source_sha256='a'),dict(role='heldout_webvid',source_sha256='b')]))
    cases=[dict(id=f'webvid_00{i}_id',dataset='webvid',source_sha256=s) for i,s in enumerate('abc',1)]
    assert source_roles(cases)=={'a':'train','b':'validation','c':'test'}


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_independent_memory_chunks_preserve_input_gradients(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from opensora.models.layers.blocks import Attention,Mlp
    from semantic_transmission.adapter_training_memory import independent_chunks
    torch.manual_seed(7)
    model=torch.nn.Sequential(Attention(32,num_heads=4),Mlp(32,hidden_features=64)).to(device).eval().requires_grad_(False)
    x=torch.randn(4,19,32,device=device,requires_grad=True);y=model(x);y.square().sum().backward();g=x.grad.clone()
    x2=x.detach().clone().requires_grad_()
    with independent_chunks(1024,512) as record:
        y2=model(x2);y2.square().sum().backward()
    torch.testing.assert_close(y2,y,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(x2.grad,g,atol=2e-6,rtol=1e-4)
    assert record['attention_calls']>0 and record['mlp_calls']>0


def test_quality_cannot_reward_incorrect_added_motion():
    from semantic_transmission.adapter_training_evaluation import quality_objective
    a=dict(psnr_db=20,lpips_vgg=.3,flow_error_mean=1.,cadence_error_mean=1.,boundary_error=1.,dists=.2)
    b=dict(a,lpips_vgg=.25,flow_error_mean=2.)
    q=quality_objective(dict(means={'webvid':{'combined_context':dict(baseline=a,adapter=b)}}))
    assert not q['passes_quality_guards'] and not q['guards']['webvid']['motion']


def test_controller_resumes_pending_quality_before_more_training(tmp_path,monkeypatch):
    from semantic_transmission import bsu_adapter_training as training
    from semantic_transmission.artifacts import write_json
    c=dict(config(),min_steps=3,max_steps=5,save_every=1,validate_every=1,quality_every=2,
        loss_patience=99,quality_patience=99,automatic_final_test=False)
    p=dict(config=c,signature='s',remaining_distribution_gap='recorded',test_scope='reserved',convergence_scope='bounded')
    write_json(tmp_path/'data.json',dict(signature='s',receipts=[]))
    def saved(step):return dict(step=step,adapter={'weight':torch.tensor([float(step)])},training={'loss_plateau':dict(best=1,bad=0,evaluations=step)})
    state.commit_checkpoint(tmp_path,saved(2),'s')
    calls=[]
    def worker(root,config,stage,**kw):
        calls.append((stage,kw))
        if stage=='train':state.commit_checkpoint(root,saved(kw['stop_step']),'s')
        elif stage=='quality':
            step=kw['step'];checkpoint=kw['checkpoint']
            q=dict(step=step,score=1/step,passes_quality_guards=step==4,checkpoint=str(checkpoint))
            write_json(Path(root)/f'validation/step_{step:07d}/quality.json',q)
        else:raise AssertionError(stage)
    monkeypatch.setattr(training,'worker',worker)
    training.run(tmp_path,DEFAULT_CONFIG,p)
    assert calls[0][0]=='quality' and calls[0][1]['step']==2
    report=json.loads((tmp_path/'RESULT.json').read_text())
    assert report['stop_reason']=='MAX_STEPS_WITHOUT_CONFIRMED_PLATEAU'
    assert report['best_validation']['step']==4 and not report['sufficient_training_proven']
    assert not report['generalization_passed']


def test_internal_worker_cannot_bypass_output_lock(tmp_path,monkeypatch):
    from semantic_transmission import bsu_adapter_training as training
    monkeypatch.delenv('BSU_ADAPTER_TRAINING_PARENT',raising=False)
    monkeypatch.setattr(sys,'argv',['train','worker','--root',str(tmp_path),'--worker-stage','train'])
    with pytest.raises(ValueError,match='locked parent'):training.main()
    assert not (tmp_path/'protocol.json').exists()
