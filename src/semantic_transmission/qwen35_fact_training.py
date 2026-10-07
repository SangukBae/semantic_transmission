"""Evidence-aware fact supervision and a matched CE joint-training control.

v1 remains immutable. Both arms start from the same staged checkpoint, use the
same re-reviewed data/order/learning rates and validation policy. The fact arm
requires every temporal evidence group, masks unsupported caption targets and
balances the CE of retained fact spans. This is supervised learning, not an
independent semantic reward or a guarantee of factual free generation.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import math
from pathlib import Path
import random
import time

from .qwen35_training import (CONFIG, RUN, engine, images_for, labels_for, rows_for,
    checkpoint_path, save_checkpoint, load_features, load_lora, seed_all, release_gpu, caption_loss, ensure_cache)
from .qwen35_training_data import ROOT, DEFAULT_DATA, read_json, write_json, sha256, verify, validate_record
from .qwen35_multitask import TemporalSelector, coverage, supervised_head_loss, targets_for

ANNOTATIONS=ROOT/'configs/qwen35_fact_annotations_v2.json'
SETTINGS=ROOT/'configs/qwen35_fact_joint_v2.json'


def covered(fact, keys):
    selected=set(map(int,keys))
    return all(selected.intersection(group) for group in fact['evidence_groups'])


def validate_fact(fact,n=8):
    groups=fact.get('evidence_groups')
    if not fact.get('text') or not groups or any(not g for g in groups):
        raise ValueError('Fact lacks text/evidence groups')
    if any(type(i) is not int or not 0<=i<n for g in groups for i in g):
        raise ValueError('Invalid fact evidence index')
    if fact.get('kind') not in ('state','transition','continuous_motion'):
        raise ValueError('Invalid fact kind')


def revised_rows(data,annotations,split):
    if split not in ('train','validation'):
        raise ValueError('Training sidecar cannot supply test targets')
    meta=read_json(annotations)
    if meta['base_freeze_sha256']!=sha256(data/'freeze.json'):
        raise ValueError('Annotations refer to changed base data')
    base=rows_for(data,split)
    valid={r['id'] for s in ('train','validation') for r in rows_for(data,s)}
    if set(meta['records'])!=valid:
        raise ValueError('Sidecar contains missing/extra/test rows')
    result=[]
    for row in base:
        patch=meta['records'][row['id']]
        if (patch['split']!=split or patch['source_sha256']!=row['source_sha256'] or
            patch['source_frame_hashes']!=[f['sha256'] for f in row['frames']]):
            raise ValueError('Annotation source/split mismatch')
        for fact in patch['facts']:validate_fact(fact)
        updated=dict(row, facts=patch['facts'],events=patch['events'],caption=patch['caption'])
        validate_record(updated)
        result.append(updated)
    return result


def balanced_order(rows,seed):
    """Keep every original window; repeat event windows to balance strata."""
    rng=random.Random(seed)
    event=[r for r in rows if r['events']]
    other=[r for r in rows if not r['events']]
    result=list(rows)
    if event and other and len(event)<len(other):
        pool=list(event);rng.shuffle(pool)
        result.extend(pool[i%len(pool)] for i in range(len(other)-len(event)))
    rng.shuffle(result)
    return result


def evidence_penalty(head,pred,row):
    """Expected missing-fact cost over valid hard subsets; AND across groups."""
    import torch
    from torch.nn import functional as F
    weights=[2. if f['kind']!='state' else 1. for f in row['facts']]
    penalties=[]
    for subset in head.subsets.tolist():
        missing=sum(w*(not covered(f,subset)) for w,f in zip(weights,row['facts']))/sum(weights)
        cov=coverage(row,subset)
        missed_event=1-cov['event_hits']/cov['event_count'] if cov['event_count'] else 0.
        penalties.append(2*missed_event+2*missing)
    bce=F.binary_cross_entropy_with_logits(pred['candidate_logits'],targets_for(row,pred['candidate_logits'].device),
                                         pos_weight=pred['candidate_logits'].new_tensor(3.))
    missing=(pred['probabilities']*pred['probabilities'].new_tensor(penalties)).sum()
    return bce+missing


def supported_target(row,keys):
    facts=[f for f in row['facts'] if covered(f,keys)]
    if not facts:raise ValueError('No grounded caption fact remains')
    return ' '.join(f['text'] for f in facts),facts


def balanced_token_loss(losses, offsets, facts):
    """Average tokens within each fact before averaging fact costs."""
    import torch
    start=0
    terms=[];weights=[]
    for f in facts:
        end=start+len(f['text'])
        indices=[i for i,(a,b) in enumerate(offsets) if b>a and a<end and b>start]
        if not indices:raise ValueError('Fact has no aligned target tokens')
        terms.append(losses[indices].mean())
        weights.append(2. if f['kind']!='state' else 1.)
        start=end+1
    weight=losses.new_tensor(weights)
    return (torch.stack(terms)*weight).sum()/weight.sum()+0.05*losses[-1]


def fact_loss(model,processor,head,pred,features,row,config):
    import torch
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    keys=pred['indices'].tolist()
    target,facts=supported_target(row,keys)
    images=images_for(row,keys)
    messages=engine().base.messages_for(images,labels_for(row,keys),config['caption_prompt'])
    prefix=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    kwargs={'images':images,'return_tensors':'pt','images_kwargs':{'size':{'shortest_edge':4096,'longest_edge':config['max_pixels']}}}
    inputs=processor(text=[prefix+target+'<|im_end|>'],**kwargs)
    prefix_ids=processor(text=[prefix],**kwargs).input_ids
    length=prefix_ids.shape[1]
    assert torch.equal(inputs.input_ids[:,:length],prefix_ids)
    assert torch.equal(inputs.image_grid_thw,features['grid'][keys])
    encoded=processor.tokenizer(target+'<|im_end|>',add_special_tokens=False,return_offsets_mapping=True)
    if inputs.input_ids[0,length:].tolist()!=encoded['input_ids']:
        raise ValueError('Fact-token boundary differs from multimodal target')
    if inputs.input_ids.shape[1]>config['max_input_tokens']:raise ValueError('Token budget exceeded')
    inputs.pop('pixel_values');inputs=inputs.to('cuda')
    selected=head.select_tokens(features['tokens'].to('cuda'),pred)
    pos=torch.arange(length-1,inputs.input_ids.shape[1]-1,device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out=model(**inputs,use_cache=False,logits_to_keep=pos,
            mm_encoder_outputs={'image':BaseModelOutputWithPooling(pooler_output=tuple(selected.unbind(0)))})
        ce=torch.nn.functional.cross_entropy(out.logits.float().flatten(0,1),inputs.input_ids[:,length:].flatten(),reduction='none')
        loss=balanced_token_loss(ce,encoded['offset_mapping'],facts)
    return loss, {'supervised_facts':len(facts),'masked_facts':len(row['facts'])-len(facts),'target_words':len(target.split())}


def validation(model,processor,head,args,config,rows):
    """Same fixed-key language targets for both arms, plus real hard coverage."""
    import torch
    model.eval();head.eval()
    losses=[];events=event_count=facts=fact_count=0
    with torch.no_grad():
        for row in rows:
            feat=load_features(args,row)
            pred=head(feat['pooled'].cuda(),config['temperature'])
            keys=pred['indices'].tolist()
            cov=coverage(row,keys)
            events+=cov['event_hits'];event_count+=cov['event_count']
            facts+=sum(covered(f,keys) for f in row['facts']);fact_count+=len(row['facts'])
            # Fixed oracle-best reference subset is validation-only teacher loss;
            # it is never used to generate evaluation captions or select keys.
            subsets=head.subsets.tolist()
            fixed=max(subsets,key=lambda s:(sum(covered(f,s) for f in row['facts']),coverage(row,s)['event_hits']))
            target,_=supported_target(row,fixed)
            matrix=torch.nn.functional.one_hot(torch.tensor(fixed,device='cuda'),8).float()
            fixed_pred={'indices':torch.tensor(fixed,device='cuda'),'matrix':matrix}
            valrow=dict(row,caption=target)
            losses.append(float(caption_loss(model,processor,head,fixed_pred,feat,valrow,config)))
    if not all(math.isfinite(x) for x in losses):raise RuntimeError('Nonfinite validation loss')
    return {'fixed_evidence_caption_ce':sum(losses)/len(losses),'event_hits':events,'event_count':event_count,
            'fact_evidence_hits':facts,'fact_count':fact_count}


def eligible(metrics,initial):
    return (metrics['event_hits']>=initial['event_hits'] and
            metrics['fact_evidence_hits']>=initial['fact_evidence_hits'])


def train(args):
    import torch
    config=read_json(CONFIG);setting=read_json(SETTINGS)
    verify(args.data)
    ensure_cache(args,config)
    trainrows=revised_rows(args.data,args.annotations,'train')
    valrows=revised_rows(args.data,args.annotations,'validation')
    parent=checkpoint_path(RUN/'caption')
    folder=args.run/args.mode
    dependencies=[Path(__file__),Path(__file__).with_name('qwen35_training.py'),Path(__file__).with_name('qwen35_multitask.py'),Path(__file__).with_name('qwen35_training_data.py')]
    identity={'mode':args.mode,'settings':setting,'base_config':config,
        'data_sha256':sha256(args.data/'freeze.json'),'annotation_sha256':sha256(args.annotations),
        'parent':str(parent),'parent_sha256':sha256(parent/'complete.json'),
        'code':{str(p.relative_to(ROOT)):sha256(p) for p in dependencies},'smoke_updates':args.smoke_updates}
    if (folder/'protocol.json').exists():
        if read_json(folder/'protocol.json')!=identity:raise ValueError('Run conditions changed')
        if not args.resume:raise ValueError('Existing run; pass --resume or choose new --run')
    write_json(folder/'protocol.json',identity)
    if (folder/'RESULT.json').exists() and read_json(folder/'RESULT.json')['status']=='COMPLETE':
        checkpoint_path(folder);print('Already complete',folder);return
    current=checkpoint_path(folder,'latest') if (folder/'latest.json').exists() else parent
    seed_all(config['seed'])
    model,processor,runtime=load_lora(config,current/'adapter')
    head=TemporalSelector(hidden=config['head_hidden'],candidates=8,keys=4).cuda()
    head.load_state_dict(torch.load(current/'head.pt',weights_only=True,map_location='cuda'))
    seed_all(config['seed'])
    opt=torch.optim.AdamW([{'params':list(head.parameters()),'lr':setting['head_lr']},
        {'params':[p for p in model.parameters() if p.requires_grad],'lr':setting['lora_lr']}])
    params=[p for group in opt.param_groups for p in group['params']]
    state={'stage':'joint','mode':args.mode,'updates':0,'next_epoch':0,'next_offset':0,
           'history':[],'protocol_sha256':sha256(folder/'protocol.json'),'caption_head_gradient':None,
           'initial_validation':None,'best_validation_loss':None,'bad_epochs':0,'weights_changed':False}
    if current!=parent:
        state=read_json(current/'state.json')
        assert state['protocol_sha256']==sha256(folder/'protocol.json')
        opt.load_state_dict(torch.load(current/'optimizer.pt',weights_only=True,map_location='cuda'))
        rng=torch.load(current/'rng.pt',weights_only=True,map_location='cpu')
        torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state_all(rng['cuda']);random.setstate(rng['python'])
    else:
        val=validation(model,processor,head,args,config,valrows)
        state['initial_validation']=val;state['best_validation_loss']=val['fixed_evidence_caption_ce']
        state['history'].append(dict(epoch=0,validation=val,selected=True))
        save_checkpoint(folder,head,model,opt,state,best=True)
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    start_updates=state['updates']
    stopped=False
    for epoch in range(state['next_epoch'],setting['max_epochs']):
        rows=balanced_order(trainrows,config['seed']+epoch)
        offset=state['next_offset'] if epoch==state['next_epoch'] else 0
        model.train();model.base_model.model.model.visual.eval();head.train()
        while offset<len(rows):
            batch=rows[offset:offset+config['gradient_accumulation']]
            opt.zero_grad(set_to_none=True);losses=[];details=[]
            for row in batch:
                features=load_features(args,row)
                pred=head(features['pooled'].cuda(),config['temperature'])
                if args.mode=='facts':
                    ce,detail=fact_loss(model,processor,head,pred,features,row,config)
                    aux=evidence_penalty(head,pred,row)
                else:
                    ce=caption_loss(model,processor,head,pred,features,row,config)
                    aux,_=supervised_head_loss(head,pred,row)
                    detail={'supervised_facts':len(row['facts']),'masked_facts':0,'target_words':len(row['caption'].split())}
                if state['caption_head_gradient'] is None:
                    grads=torch.autograd.grad(ce,[head.selection.weight,head.candidate.weight],retain_graph=True)
                    norms=[float(g.norm()) for g in grads]
                    if not all(math.isfinite(x) and x>0 for x in norms):raise RuntimeError('Disconnected caption-to-head gradient')
                    state['caption_head_gradient']={'selection':norms[0],'candidate':norms[1]}
                loss=ce+setting['head_loss_weight']*aux
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                (loss/len(batch)).backward()
                losses.append(float(loss.detach()));details.append(detail)
                del loss,ce,aux,pred,features
            norm=torch.nn.utils.clip_grad_norm_(params,1.)
            if not torch.isfinite(norm) or norm<=0:raise RuntimeError('Invalid gradient norm')
            tracked=next(p for p in params if p.grad is not None and p.grad.abs().max()>0)
            before=tracked.detach().clone();opt.step()
            state['weights_changed']|=not torch.equal(before,tracked)
            del before
            offset+=len(batch)
            state.update(updates=state['updates']+1,next_epoch=epoch,next_offset=offset)
            entry={'epoch':epoch+1,'update':state['updates'],'loss':sum(losses)/len(losses),'gradient_norm':float(norm),
                   'supervised_facts':sum(x['supervised_facts'] for x in details),'masked_facts':sum(x['masked_facts'] for x in details)}
            state['history'].append(entry)
            write_json(folder/'status.json',dict(status='RUNNING',**entry))
            print(args.mode,entry,flush=True)
            if args.stop_after_updates and state['updates']-start_updates>=args.stop_after_updates:
                save_checkpoint(folder,head,model,opt,state)
                write_json(folder/'status.json',{'status':'CHECKPOINTED','updates':state['updates']})
                del opt,params,model,processor,head,tracked;release_gpu();return
            if args.smoke_updates and state['updates']>=args.smoke_updates:
                save_checkpoint(folder,head,model,opt,state)
                write_json(folder/'RESULT.json',{'status':'SMOKE_CHECKPOINTED','updates':state['updates'],
                    'caption_head_gradient':state['caption_head_gradient'],'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
                del opt,params,model,processor,head,tracked;release_gpu();return
            if offset<len(rows) and state['updates']%setting['checkpoint_every_updates']==0:
                save_checkpoint(folder,head,model,opt,state)
        val=validation(model,processor,head,args,config,valrows)
        ok=eligible(val,state['initial_validation'])
        best=ok and val['fixed_evidence_caption_ce']<state['best_validation_loss']-setting['min_delta']
        state['bad_epochs']=0 if best else state['bad_epochs']+1
        if best:state['best_validation_loss']=val['fixed_evidence_caption_ce']
        state.update(next_epoch=epoch+1,next_offset=0)
        state['history'].append(dict(epoch=epoch+1,validation=val,coverage_gate=ok,selected=best))
        save_checkpoint(folder,head,model,opt,state,best=best)
        print('validation',args.mode,val,'best',best,flush=True)
        if state['bad_epochs']>=setting['patience']:
            stopped=True;break
    write_json(folder/'RESULT.json',{'status':'COMPLETE','mode':args.mode,'updates':state['updates'],'epochs':state['next_epoch'],
        'early_stopped':stopped,'weights_changed':state['weights_changed'],'caption_head_gradient':state['caption_head_gradient'],
        'best_checkpoint':str(checkpoint_path(folder)),'best_validation_loss':state['best_validation_loss'],
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'session_seconds':time.monotonic()-start,
        'runtime':runtime,'test_used_for_training_or_selection':False,
        'training_sources':sorted({r['source_sha256'] for r in trainrows}),
        'validation_sources':sorted({r['source_sha256'] for r in valrows}),
        'quality_improvement_claimed':False})
    write_json(folder/'status.json',{'status':'COMPLETE'})
    del opt,params,model,processor,head,tracked;release_gpu()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['control','facts'])
    p.add_argument('--data',type=Path,default=DEFAULT_DATA)
    p.add_argument('--annotations',type=Path,default=ANNOTATIONS)
    p.add_argument('--run',type=Path,default=ROOT/'outputs/qwen35_fact_joint_v2')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke-updates',type=int,default=0)
    p.add_argument('--stop-after-updates',type=int,default=0)
    args=p.parse_args()
    with open('/tmp/qwen35_multitask_gpu.lock','w') as lock, (ROOT/'.local/etri_60s_check.lock').open('a+') as shared:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
        train(args)


if __name__=='__main__':main()
