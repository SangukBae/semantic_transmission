"""Exercise accumulated grounded backward without optimizer updates.

Keeps the frozen training implementation/checkpoints intact. A low FD limit can
be requested explicitly to reproduce a WSL GPU-handle allocation failure.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import resource
import time
import traceback


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--batches',type=int,default=3)
    p.add_argument('--nofile-soft',type=int)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    if not 1<=args.batches<=31:raise ValueError('Expected1..31 batches')
    if args.nofile_soft is not None:
        _,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE,(args.nofile_soft,hard))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    if args.report.exists():raise ValueError('Do not overwrite an existing runtime report')
    # Keep an open report descriptor so an EMFILE failure can still be recorded.
    with args.report.open('w') as output,ExitStack() as stack:
        from semantic_transmission.qwen35_grounded_training import (
            RUN,SETTINGS,config_and_parent,load_model,model_digest,epoch_order,
            features_for,caption_loss,fixed_prediction,eligible_facts,grounded_loss,validate)
        from semantic_transmission.qwen35_training_data import ROOT,read_json,sha256
        for name in ['/tmp/qwen35_multitask_gpu.lock',str(ROOT/'.local/etri_60s_check.lock')]:
            lock=stack.enter_context(open(name,'w'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import torch
        from semantic_transmission.qwen35_multitask import TemporalSelector
        report=dict(status='RUNNING',optimizer_steps=0,checkpoint_saved=False,
            nofile_limit=list(resource.getrlimit(resource.RLIMIT_NOFILE)),allocator=os.getenv('PYTORCH_ALLOC_CONF'),
            training_implementation_sha256=sha256(ROOT/'src/semantic_transmission/qwen35_grounded_training.py'),
            events=[],planned_batches=args.batches,phase='startup')
        def record(phase,**extra):
            torch.cuda.synchronize()
            event=dict(phase=phase,open_fds=len(os.listdir('/proc/self/fd')),
                allocated_gib=torch.cuda.memory_allocated()/2**30,reserved_gib=torch.cuda.memory_reserved()/2**30,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,**extra)
            report['events'].append(event);report['phase']=phase
            print(json.dumps(event),flush=True)
        try:
            config,parent=config_and_parent();model,processor,runtime=load_model(config,parent/'adapter')
            report['runtime']=runtime;report['before_adapter_digest']=model_digest(model)
            head=TemporalSelector(hidden=config['head_hidden']).cuda().eval()
            head.load_state_dict(torch.load(parent/'head.pt',weights_only=True,map_location='cuda'));head.requires_grad_(False)
            params=[x for x in model.parameters() if x.requires_grad]
            rows=read_json(RUN/'train.json')['records'];order=epoch_order(rows,0)
            report['phase']='initial_validation'
            validate(model,processor,head,read_json(RUN/'validation.json')['records'],config)
            torch.cuda.reset_peak_memory_stats();record('initial_validation_complete')
            moments=[];start=time.monotonic()
            for offset in range(0,args.batches*SETTINGS['gradient_accumulation'],SETTINGS['gradient_accumulation']):
                batch=order[offset:offset+SETTINGS['gradient_accumulation']];model.zero_grad(set_to_none=True)
                for row in batch:
                    report['phase']='caption_backward';report['row_id']=row['id']
                    feat=features_for(row);ce=caption_loss(model,processor,head,fixed_prediction(row['keys']),feat,row,config)
                    (ce/len(batch)).backward();del ce
                    facts=eligible_facts(row)
                    if facts:
                        fact=facts[rows.index(row)%len(facts)];report['phase']='grounded_backward'
                        aux,logits,_=grounded_loss(model,processor,row,feat,config,fact)
                        (SETTINGS['grounding_weight']*aux/len(batch)).backward();del aux,logits
                    record('row_complete',row_id=row['id'],offset=offset)
                norm=torch.nn.utils.clip_grad_norm_(params,1.)
                if not torch.isfinite(norm) or norm.item()<=0:raise ValueError('Invalid accumulated gradient')
                if not moments:
                    report['phase']='simulate_adam_moment_allocation'
                    moments=[torch.zeros_like(x) for x in params for _ in range(2)]
                record('batch_complete_without_step',offset=offset,gradient_norm=float(norm))
            report['after_adapter_digest']=model_digest(model)
            if report['after_adapter_digest']!=report['before_adapter_digest']:raise ValueError('Weights changed during no-update probe')
            report.update(status='PASS_NO_WEIGHT_UPDATES',seconds=time.monotonic()-start,
                simulated_adam_state=True,completed_batches=args.batches,
                peak_open_fds=max(x['open_fds'] for x in report['events']))
        except Exception as exc:
            report.update(status='FAILED_NO_OPTIMIZER_STEPS',error=str(exc),exception=type(exc).__name__,traceback=traceback.format_exc())
            raise
        finally:
            json.dump(report,output,indent=2);output.write('\n');output.flush()


if __name__=='__main__':main()
