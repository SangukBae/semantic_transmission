"""Bounded CUDA verification on one real training clip; no quality claim or deployment."""
import argparse
import copy
import fcntl
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from semantic_transmission import sparc_selector_budget as budget
from semantic_transmission import sparc_selector_risk as risk
from semantic_transmission import sparc_selector_train as supervision
from semantic_transmission import sparc_selector_train_v4 as training
from semantic_transmission import sparc_selector_v4 as runner
from semantic_transmission import sparc_selector_v4_data as data
from semantic_transmission.sparc_selector_data import read, save, sha, digest
from semantic_transmission.sparc_selector_workers import REPO


def check(parent, output):
    p,rows=runner.initialize(output,parent,write=False)
    teacher=[r for r in rows if r['split']=='train' and (parent/'clips'/r['id']/'teacher.json').exists()]
    row=min(teacher,key=lambda r:(r['duration_seconds'],r['id']))
    p=copy.deepcopy(p);p['search']['teacher_clips']=0;p['calibration_ids']=[]
    p['verification_only']=True;p['signature']=digest({k:v for k,v in p.items() if k!='signature'})
    output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists() and read(output/'protocol.json')!=p:
        raise ValueError('smoke protocol changed; use new output')
    save(output/'protocol.json',p)
    supervision.runtimes=lambda:p['runtimes']
    data.prepare(output,parent,Path(p['source_root']),[row],p,
        lambda stage,done,total,**kw:print(stage,done,total,kw,flush=True))
    folder=output/'clips'/row['id'];data.verify_cache(folder,p)
    observations=read(folder/'observations.json');baseline=read(folder/'baseline.json');ready=read(folder/'prepared.json')
    pixels=np.prod([ready['source']['video'][k] for k in ('frames','height','width')])
    with np.load(folder/'features.npz') as z:features=[torch.from_numpy(z[k].copy()) for k in ('patches','motion')]
    with np.load(folder/'risk_labels.npz') as z:local=torch.from_numpy(z['local_log_mse'].copy())
    initial=torch.load(parent/'training/best.pt',map_location='cpu',weights_only=False)
    record=dict(observations=observations,baseline=baseline,pixels=int(pixels),
        absolute=torch.tensor([[*[o['quality'][k] for k in budget.METRICS],budget.cbr(o,pixels)] for o in observations]).clamp_min(1e-8).log(),
        delta=torch.tensor([[o['quality']['psnr_db']-baseline['quality']['psnr_db'],o['quality']['lpips_vgg']-baseline['quality']['lpips_vgg']] for o in observations]),
        unsafe=torch.tensor([not risk.quality_feasible(o,baseline,p['quality_goal']) for o in observations],dtype=torch.float32))
    norm=dict(initial['normalization'],delta_mean=torch.zeros(2),delta_std=torch.tensor([.5,.01]),
              local_mean=local.mean(),local_std=local.std().clamp_min(.05))
    corpus=SimpleNamespace(records={row['id']:record},prepared={row['id']:ready},normalization=norm,p=p,
                           tensors=lambda _: (features,local))
    torch.set_num_threads(2);torch.manual_seed(p['seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=risk.RiskSelector(**p['model']).cuda().train();model.load_state_dict(initial['model'],strict=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
    ids=torch.arange(min(4,len(observations)))
    def step(model,optimizer):
        optimizer.zero_grad(set_to_none=True)
        loss=training.batch_loss(model,corpus,row,ids,p['target_cbr'],'cuda')
        if not torch.isfinite(loss):raise ValueError('nonfinite CUDA loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        gradients={name:sum(float(x.grad.abs().sum()) for x in module.parameters() if x.grad is not None)
                   for name,module in [('delta',model.delta),('spatial',model.local_error),('violation',model.violation)]}
        if min(gradients.values())<=0:raise ValueError('new heads did not receive gradients')
        optimizer.step();return float(loss.detach()),gradients
    losses=[]
    for _ in range(2):losses.append(step(model,optimizer)[0])
    checkpoint=output/'cuda_resume.pt'
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all()),checkpoint)
    loss,gradients=step(model,optimizer);losses.append(loss)
    expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    del model,optimizer
    restored=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=risk.RiskSelector(**p['model']).cuda().train();model.load_state_dict(restored['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5);optimizer.load_state_dict(restored['optimizer'])
    torch.set_rng_state(restored['cpu_rng']);torch.cuda.set_rng_state_all(restored['cuda_rng'])
    resumed_loss,_=step(model,optimizer)
    same=all(torch.equal(expected[k],v.cpu()) for k,v in model.state_dict().items())
    if not same or resumed_loss!=loss:raise ValueError('real CUDA resume diverged')
    result=dict(status='REAL_TRAINING_CLIP_CUDA_CHECK_PASSED',clip=row['id'],frames=len(features[0]),
        candidates=len(observations),spatial_shape=list(local.shape),losses=losses,new_head_gradients=gradients,
        cuda_resume_bit_identical=same,parent_checkpoint_sha256=sha(parent/'training/best.pt'),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(0),full_retraining=False,quality_improvement_proven=False)
    save(output/'RESULT.json',result);print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=runner.PARENT)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with (REPO/'.local/etri_60s_check.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check(args.parent.resolve(),args.output.resolve())
