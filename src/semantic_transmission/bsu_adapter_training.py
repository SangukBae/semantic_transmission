"""Resumable native BSU adapter training with measured stopping criteria.

The command never promises that a fixed update count is sufficient. It records
validation plateaus, quality guard failures and budget exhaustion separately.
"""
import argparse
from collections import OrderedDict
from contextlib import nullcontext
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from . import bsu_motion_validation as common
from .artifacts import sha256, write_json
from . import adapter_training_data as data
from .adapter_training_state import (atomic_torch_save,commit_checkpoint,load_checkpoint,
                                     stop_reason,update_plateau)


def event(root,stage,done=None,total=None,**fields):
    text=stage if total is None else f'{stage} {done}/{total} ({100*done/total:.1f}%)'
    record=dict(stage=text,current='BSU 생성 내부 어댑터 학습',state='실행 중')
    record.update(fields)
    print('COMPARE_PROGRESS '+json.dumps(record,ensure_ascii=False),flush=True)
    write_json(Path(root)/'status.json',dict(record,updated_unix=time.time(),pid=os.getpid()))


def export_adapter(root,state,signature):
    from .trajectory_adapter import POLICY
    path=Path(root)/'exports'/f'step_{state["step"]:07d}.pt'
    receipt=path.with_suffix('.json')
    if receipt.exists():
        old=common.read(receipt)
        if old['signature']!=signature or old['sha256']!=sha256(path):raise ValueError('exported checkpoint changed')
    else:
        atomic_torch_save(dict(policy=POLICY,state_dict=state['adapter'],steps=state['step'],training_signature=signature),path)
        write_json(receipt,dict(signature=signature,sha256=sha256(path),step=state['step']))
    return path


class NativeTrainer:
    def __init__(self,root,p):
        import torch
        from .bsu_adapter_validation import imports,model_config
        from .trajectory_adapter import TrajectoryAdapter
        from opensora.acceleration.checkpoint import set_grad_checkpoint
        self.root,self.p,self.torch=Path(root),p,torch
        modes,build=imports();cfg=dict(model_config().model)
        torch.manual_seed(p['config']['seed'])
        self.model=build(cfg,modes,input_size=(5,40,72),in_channels=4,caption_channels=4096,model_max_length=300,
            enable_sequence_parallelism=False).cuda().to(torch.bfloat16).eval()
        self.model.requires_grad_(False);set_grad_checkpoint(self.model)
        self.adapter=TrajectoryAdapter(hidden=self.model.hidden_size).cuda()
        self.handle=self.adapter.attach(self.model)
        self.optimizer=torch.optim.AdamW(self.adapter.parameters(),lr=p['config']['learning_rate'],weight_decay=p['config']['weight_decay'])
        self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=.5,patience=2,
            threshold=p['config']['relative_min_delta'],threshold_mode='rel',min_lr=p['config']['min_learning_rate'])
        self.cache=OrderedDict();self.rng=np.random.default_rng(p['config']['seed'])
        self.state=dict(step=0,microbatches=0,epoch=0,order=[],offset=0,loss_history=[],validation=[],loss_plateau=None,
            best_loss=None,sample_counts={},training_seconds=0.)

    def load(self,e):
        import torch
        if e['name'] in self.cache:
            self.cache.move_to_end(e['name']);return self.cache[e['name']]
        path=self.root/'data'/f'{e["name"]}.pt';receipt=common.read(path.with_suffix('.json'))
        if receipt['signature']!=self.p['signature'] or receipt['sha256']!=sha256(path):raise ValueError('training cache hash mismatch')
        sample=torch.load(path,map_location='cpu',weights_only=True)
        if sample['signature']!=self.p['signature'] or sample['example']!=e:raise ValueError('training cache identity mismatch')
        self.cache[e['name']]=sample
        if len(self.cache)>4:self.cache.popitem(last=False)
        return sample

    def restore(self,state):
        import torch
        self.adapter.load_state_dict(state['adapter']);self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler']);self.state=state['training']
        if self.state['step']!=state['step']:raise ValueError('training counter mismatch')
        self.rng.bit_generator.state=json.loads(state['numpy_rng_json'])
        torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])

    def snapshot(self):
        import torch
        return dict(step=self.state['step'],adapter={k:v.detach().cpu() for k,v in self.adapter.state_dict().items()},
            optimizer=self.optimizer.state_dict(),scheduler=self.scheduler.state_dict(),training=self.state,
            numpy_rng_json=json.dumps(self.rng.bit_generator.state),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())

    def choose(self,examples):
        s=self.state
        if s['offset']>=len(s['order']):
            s['order']=self.rng.permutation(len(examples)).tolist();s['offset']=0;s['epoch']+=1
        e=examples[s['order'][s['offset']]];s['offset']+=1
        s['sample_counts'][e['name']]=s['sample_counts'].get(e['name'],0)+1
        return e

    def loss(self,e,seed,timestep=None,drop=False,enabled=True):
        import torch
        from .short_schedule import transform
        from opensora.schedulers.rf.rectified_flow import timestep_transform
        sample=self.load(e);r=sample['received'];target=sample['target'].cuda().float();mask=r['mask'].cuda()
        fixed=r['fixed'].cuda().float();generator=torch.Generator(device='cuda').manual_seed(seed)
        noise=torch.randn(target.shape,device='cuda',dtype=torch.float32,generator=generator)
        t=torch.rand(1,device='cuda',generator=generator)*999.998+.001 if timestep is None else torch.tensor([timestep],device='cuda')
        kwargs=dict(y=r['y'].cuda().to(torch.bfloat16),mask=r['text_mask'].cuda(),x_mask=mask,
            height=torch.tensor([320],device='cuda',dtype=torch.bfloat16),width=torch.tensor([576],device='cuda',dtype=torch.bfloat16),
            fps=torch.tensor([24],device='cuda',dtype=torch.bfloat16))
        frames=torch.tensor([e['frames']],device='cuda',dtype=torch.bfloat16)
        if frames.item()!=e['frames']:frames=torch.tensor([e['frames']],device='cuda',dtype=torch.float32)
        t=transform(t,dict(kwargs,num_frames=frames),timestep_transform,num_timesteps=1000)
        z=(1-t[:,None,None,None,None]/1000)*target+t[:,None,None,None,None]/1000*noise
        z=torch.where(mask[:,None,:,None,None],z,fixed)
        condition=r['condition'].cuda().float()
        self.adapter.condition=None if not enabled else (torch.zeros_like(condition) if drop else condition)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            prediction=self.model(z.to(torch.bfloat16),t,**kwargs).chunk(2,dim=1)[0].float()
            loss=(prediction-(target-noise)).square()[:,:,mask[0]].mean()
        if not torch.isfinite(loss):raise ValueError(f'non-finite loss for {e["name"]}')
        return loss

    def validate(self):
        import torch
        import statistics as st
        rows=[];examples=[e for e in self.p['examples'] if e['role']=='validation'];config=self.p['config']
        self.adapter.eval()
        with torch.no_grad():
            for i,e in enumerate(examples):
                values=[]
                for seed in config['validation_seeds']:
                    for t in config['validation_timesteps']:
                        actual=int(data.fingerprint([e['name'],seed,t])[:14],16)
                        values.append(float(self.loss(e,actual,timestep=t)))
                rows.append(dict(id=e['id'],name=e['name'],loss=st.mean(values)))
                event(self.root,'fixed-noise validation',i+1,len(examples))
        self.adapter.train()
        # Equal source weight, despite differing numbers/lengths of intervals.
        value=st.mean(st.mean(r['loss'] for r in rows if r['id']==name) for name in {r['id'] for r in rows})
        return dict(step=self.state['step'],loss=value,rows=rows,fixed_noise=True,source_disjoint=True)


def train_chunk(root,p,stop_step):
    import torch
    from .adapter_training_memory import independent_chunks
    root=Path(root);config=p['config'];trainer=NativeTrainer(root,p)
    old=load_checkpoint(root,p['signature'])
    if old:
        trainer.restore(old);print(f'RESUME: verified optimizer/RNG/data at step {old["step"]}',flush=True)
    interrupted=[]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda signum,frame:interrupted.append(signum))
    examples=[e for e in p['examples'] if e['role']=='train'];start=time.monotonic();initial_step=trainer.state['step']
    torch.cuda.reset_peak_memory_stats()
    def save():
        state=trainer.snapshot();head=commit_checkpoint(root,state,p['signature'])
        export_adapter(root,state,p['signature']);return head
    with independent_chunks() as memory:
        if not trainer.state['validation']:
            v=trainer.validate();trainer.state['validation'].append(v)
            trainer.state['loss_plateau'],_=update_plateau(None,v['loss'],config['relative_min_delta'])
            trainer.state['best_loss']=v['loss'];save()
        while trainer.state['step']<stop_step:
            step=trainer.state['step'];trainer.optimizer.zero_grad(set_to_none=True);loss_sum=0.
            if step<config['warmup_steps']:
                for group in trainer.optimizer.param_groups:group['lr']=config['learning_rate']*(step+1)/config['warmup_steps']
            for micro in range(config['gradient_accumulation']):
                e=trainer.choose(examples);seed=int(trainer.rng.integers(0,2**50));drop=trainer.rng.random()<config['condition_dropout']
                offload=torch.autograd.graph.save_on_cpu(pin_memory=False) if config['cpu_activation_offload'] else nullcontext()
                with offload:
                    loss=trainer.loss(e,seed,drop=drop)/config['gradient_accumulation']
                    loss.backward()
                loss_sum+=float(loss.detach());trainer.state['microbatches']+=1
            norm=torch.nn.utils.clip_grad_norm_(trainer.adapter.parameters(),1.,error_if_nonfinite=True)
            trainer.optimizer.step();trainer.state['step']+=1
            step=trainer.state['step'];trainer.state['loss_history'].append(dict(step=step,loss=loss_sum,grad_norm=float(norm),lr=trainer.optimizer.param_groups[0]['lr']))
            if step%config['validate_every']==0 or step==config['max_steps']:
                v=trainer.validate();trainer.state['validation'].append(v)
                trainer.state['loss_plateau'],_=update_plateau(trainer.state['loss_plateau'],v['loss'],config['relative_min_delta'])
                trainer.scheduler.step(v['loss'])
                if v['loss']<trainer.state['best_loss']:
                    trainer.state['best_loss']=v['loss']
                    write_json(root/'best_loss.json',dict(step=step,loss=v['loss']))
                write_json(root/'validation_loss.json',dict(rows=trainer.state['validation']))
            if step%config['save_every']==0 or step==stop_step or interrupted:
                now=time.monotonic();trainer.state['training_seconds']+=now-start;start=now
                save();completed=step-initial_step
                event(root,'optimizer updates',step,config['max_steps'],
                    overall=f'{step}/{config["max_steps"]} updates (상한 기준; 조기 종료 가능)',
                    state=f'체크포인트 저장 완료 · loss={loss_sum:.5f} · lr={trainer.optimizer.param_groups[0]["lr"]:.2g}')
            if interrupted:
                event(root,'interrupted after committed optimizer step',state='중단 · 체크포인트 저장 완료')
                return 130
        write_json(root/'memory.json',dict(peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30,
            frozen_backbone=not any(p.requires_grad or p.grad is not None for p in trainer.model.parameters()),
            cpu_activation_offload=config['cpu_activation_offload'],**memory))
    trainer.handle.remove();return 0


def smoke(root,p):
    """Two real native backward passes, including the longest train interval."""
    import torch
    import resource
    from .adapter_training_memory import independent_chunks
    original=Path(root);root=original/'smoke';root.mkdir(parents=True,exist_ok=True)
    train=[e for e in p['examples'] if e['role']=='train']
    chosen=list({e['name']:e for e in (max(train,key=lambda e:e['frames']),max((e for e in train if e['segment']>0),key=lambda e:e['frames']))}.values())
    data.prepare_text(root,chosen)
    data.prepare_latents(root,chosen,p['signature'],lambda *a,**k:event(original,*a,**k))
    # Release the VAE in a separate worker before allocating the full backbone.
    write_json(root/'selection.json',dict(names=[e['name'] for e in chosen]))


def smoke_backward(root,p):
    import torch
    import resource
    from .adapter_training_memory import independent_chunks
    original=Path(root);root=original/'smoke'
    names=common.read(root/'selection.json')['names'];chosen=[e for e in p['examples'] if e['name'] in names]
    trainer=NativeTrainer(root,p);rows=[]
    with independent_chunks() as memory:
        for e in chosen:
            torch.cuda.reset_peak_memory_stats();start=time.monotonic();trainer.optimizer.zero_grad(set_to_none=True)
            before={k:v.detach().cpu().clone() for k,v in trainer.adapter.state_dict().items()}
            offload=torch.autograd.graph.save_on_cpu(pin_memory=False) if p['config']['cpu_activation_offload'] else nullcontext()
            with offload:
                loss=trainer.loss(e,12345);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(trainer.adapter.parameters(),1.,error_if_nonfinite=True);trainer.optimizer.step()
            if not any(not torch.equal(before[k],v.cpu()) for k,v in trainer.adapter.state_dict().items()):raise ValueError('adapter did not update')
            if any(q.requires_grad or q.grad is not None for q in trainer.model.parameters()):raise ValueError('backbone changed')
            rows.append(dict(name=e['name'],frames=e['frames'],history_frames=len(e['history']),loss=float(loss),grad_norm=float(norm),
                seconds=time.monotonic()-start,peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30,
                max_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2))
            event(original,'real native training smoke',len(rows),len(chosen))
        saved=trainer.snapshot();receipt=commit_checkpoint(root,saved,p['signature'])
        restored=load_checkpoint(root,p['signature'])
        for key,value in restored['adapter'].items():torch.testing.assert_close(value,saved['adapter'][key],rtol=0,atol=0)
    result=dict(status='PASSED',rows=rows,memory=memory,checkpoint_reload_exact=True,
        production_training_started=False,smoke_weights_not_used_for_training=True,signature=p['signature'])
    write_json(original/'smoke_result.json',result);print(json.dumps(result),flush=True)


def worker(root,config,stage,**options):
    command=[sys.executable,'-m',__package__+'.bsu_adapter_training','worker','--root',str(root),'--config',str(config),'--worker-stage',stage]
    for k,v in options.items():command+=['--'+k.replace('_','-'),str(v)]
    env=dict(os.environ,BSU_ADAPTER_TRAINING_PARENT=str(os.getpid()))
    completed=subprocess.run(command,env=env)
    if completed.returncode:raise SystemExit(completed.returncode)


def run(root,config,p):
    root=Path(root)
    if (root/'RESULT.json').exists():
        report=common.read(root/'RESULT.json')
        if report['signature']!=p['signature'] or sha256(report['best_checkpoint'])!=report['best_checkpoint_sha256']:raise ValueError('completed training changed')
        print('RESUME: training already complete; '+str(root/'RESULT.json'),flush=True);return
    if not (root/'data.json').exists():
        worker(root,config,'text');worker(root,config,'data')
    else:
        prepared=common.read(root/'data.json')
        if prepared['signature']!=p['signature']:raise ValueError('prepared data signature changed')
        for r in prepared['receipts']:
            if r['sha256']!=sha256(r['path']):raise ValueError('prepared data changed')
        print('RESUME: verified prepared native latents',flush=True)
    control=common.read(root/'control.json') if (root/'control.json').exists() else dict(quality=[],quality_plateau=None,best=None)
    conf=p['config']
    while True:
        state=load_checkpoint(root,p['signature']);step=state['step'] if state else 0
        # A killed quality worker resumes before any further optimizer update.
        quality_due=step>0 and (step%conf['quality_every']==0 or step==conf['max_steps'])
        if quality_due and not any(r['step']==step for r in control['quality']):
            checkpoint=export_adapter(root,state,p['signature'])
            worker(root,config,'quality',step=step,checkpoint=checkpoint,split='validation')
            quality=common.read(root/f'validation/step_{step:07d}/quality.json')
            control['quality'].append(quality)
            control['quality_plateau'],_=update_plateau(control['quality_plateau'],quality['score'],conf['relative_min_delta'])
            # Prefer guard-passing checkpoints, then the fixed composite score.
            rank=lambda q:(not q['passes_quality_guards'],q['score'])
            if control['best'] is None or rank(quality)<rank(control['best']):
                control['best']=quality
                best=common.read(root/'best.json') if (root/'best.json').exists() else None
                from .adapter_training_state import atomic_torch_save
                import torch
                atomic_torch_save(torch.load(checkpoint,map_location='cpu',weights_only=True),root/'best.pt')
                write_json(root/'best.json',dict(quality,path=str(root/'best.pt'),sha256=sha256(root/'best.pt')))
            write_json(root/'control.json',control)
        loss=state['training']['loss_plateau'] if state else None
        reason=stop_reason(step,conf,loss,control['quality_plateau'])
        if reason:
            if not control['best']:raise ValueError('no full-video validation checkpoint')
            break
        stop=min(conf['max_steps'],((step//conf['validate_every'])+1)*conf['validate_every'])
        worker(root,config,'train',stop_step=stop)
    best=common.read(root/'best.json');final_test=None
    write_json(root/'training_stop.json',dict(reason=reason,step=step,best=best,convergence_proven=False))
    if conf['automatic_final_test']:
        worker(root,config,'quality',step=best['step'],checkpoint=root/'best.pt',split='test')
        final_test=common.read(root/'final_test/quality.json')
    report=dict(status='COMPLETE',stop_reason=reason,steps=step,signature=p['signature'],
        operational_plateau=reason=='VALIDATION_PLATEAU',sufficient_training_proven=False,
        best_checkpoint=str(root/'best.pt'),best_checkpoint_sha256=sha256(root/'best.pt'),
        best_validation=best,final_test=final_test,production_default_changed=False,
        generalization_passed=bool(final_test and final_test['passes_quality_guards'] and best['passes_quality_guards']),
        limitations=[p['remaining_distribution_gap'],p['test_scope'],p['convergence_scope']])
    write_json(root/'RESULT.json',report);event(root,reason,state='학습·검증 완료 (성능 판정은 RESULT.json)')
    print(json.dumps(report,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('check','run','status','smoke','prepare','worker'))
    parser.add_argument('--root',type=Path,default=data.DEFAULT_ROOT)
    parser.add_argument('--config',type=Path,default=data.DEFAULT_CONFIG)
    parser.add_argument('--worker-stage',choices=('text','data','train','quality','smoke_prepare','smoke_backward'))
    parser.add_argument('--stop-step',type=int);parser.add_argument('--step',type=int)
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--split',choices=('validation','test'))
    args=parser.parse_args();root=args.root.resolve();config=args.config.resolve()
    if args.command=='status':
        for name in ('status.json','latest.json','best.json','training_stop.json','RESULT.json'):
            if (root/name).exists():print(name+'\n'+json.dumps(common.read(root/name),ensure_ascii=False,indent=2))
        if not root.exists():print('NOT_STARTED: '+str(root))
        return
    if args.command=='worker':
        if os.environ.get('BSU_ADAPTER_TRAINING_PARENT')!=str(os.getppid()):raise ValueError('internal worker requires the locked parent')
        p=common.read(root/'protocol.json')
        emit=lambda *a,**k:event(root,*a,**k)
        if args.worker_stage=='text':return data.prepare_text(root,p['examples'])
        if args.worker_stage=='data':return data.prepare_latents(root,p['examples'],p['signature'],emit)
        if args.worker_stage=='smoke_prepare':return smoke(root,p)
        if args.worker_stage=='smoke_backward':return smoke_backward(root,p)
        if args.worker_stage=='train':raise SystemExit(train_chunk(root,p,args.stop_step))
        if args.worker_stage=='quality':
            from .adapter_training_evaluation import run as quality
            return quality(root,p,args.step,args.checkpoint,args.split,emit)
        raise ValueError('worker stage required')
    if args.command=='check':
        print('CHECK: '+json.dumps(data.summary(data.protocol(root,config)),ensure_ascii=False,indent=2));return
    root.mkdir(parents=True,exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('another training/preparation process owns this output directory')
        p=data.protocol(root,config,create=True)
        try:
            if args.command=='smoke':
                worker(root,config,'smoke_prepare');worker(root,config,'smoke_backward')
            elif args.command=='prepare':
                worker(root,config,'text');worker(root,config,'data')
            else:run(root,config,p)
        except (Exception,KeyboardInterrupt):
            event(root,'failed/interrupted; last committed checkpoint retained',state='실패 또는 중단 · 재실행으로 검증 후 재개')
            raise


if __name__=='__main__':main()
