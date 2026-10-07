"""User-started matched caption CE vs frame-evidence/order supervision.

No new temporal adapter or selector training. The auxiliary objective predicts
visible supporting frame positions through Qwen's shared language LoRA. It does
not receive the answer/evidence indices in its input query. Preflight performs
backward checks but never optimizer.step or saves adapted weights.
"""
import argparse
from contextlib import ExitStack
import fcntl
from itertools import product
from pathlib import Path
import random
import time
from .qwen35_grounded_data import RUN,prepare,freeze,verify_data
from .qwen35_astra_transfer import RUN as PARENT,load_model
from .qwen35_training_data import ROOT,read_json,write_json,sha256
from .qwen35_training import checkpoint_path,caption_loss,save_checkpoint,seed_all,release_gpu,engine,images_for,labels_for,tensor_save
from .qwen35_ablation import encode
from .qwen35_caption_repair import fixed_prediction

SETTINGS=dict(seed=20261006,epochs=2,lr=2e-5,gradient_accumulation=4,grounding_weight=.3,
    checkpoint_policy='Predeclared final epoch2; validation CE and evidence accuracy reported, never use test for selection.',
    arms=['caption','grounded'])


def dependencies():
    names=['qwen35_grounded_data.py','qwen35_astra_transfer.py','qwen35_training.py',
           'qwen35_ablation.py','qwen35_caption_repair.py','qwen35_multitask.py']
    return {name:sha256(Path(__file__).with_name(name)) for name in names}


def support_sequences(fact,keys):
    """All ordered witnesses, admitting alternative frames in every group."""
    allowed=set(keys);groups=[sorted(set(g)&allowed) for g in fact['evidence_groups']]
    if any(not g for g in groups):raise ValueError('Unsupported fact')
    paths=[tuple(keys.index(i) for i in seq) for seq in product(*groups) if all(a<b for a,b in zip(seq,seq[1:]))]
    if not paths:raise ValueError('No chronological support')
    return sorted(set(paths))


def eligible_facts(row):
    facts=[]
    for fact in row['facts']:
        paths=support_sequences(fact,row['keys'])
        # A state visible in every input carries no frame-discrimination signal.
        if len(paths[0])>1 or len(paths)<len(row['keys']):facts.append(fact)
    return facts


def grounding_prompt(fact,keys,phase):
    mapping=', '.join(f'{letter}=i{key}' for letter,key in zip('ABCD',keys))
    role={'single':'one visible supporting observation','before':'the earliest required supporting phase','after':'the latest required supporting phase'}[phase]
    return ('Use only the four source images to locate evidence for the stated fact. '
        'The fact is a training query, not a claim to invent evidence for. '
        f'Observation labels: {mapping}. Fact: {fact["text"]}\n'
        f'Select {role}. Return exactly one letter A, B, C, or D.')


def evidence_objective(logits,paths):
    import torch
    if len(logits)==1:
        scores=logits[0].float();allowed=sorted({p[0] for p in paths})
        return torch.logsumexp(scores,0)-torch.logsumexp(scores[allowed],0)
    scores=logits[0].float()[:,None]+logits[1].float()[None,:]
    pairs=sorted({(p[0],p[-1]) for p in paths})
    targets=torch.stack([scores[a,b] for a,b in pairs])
    # Denominator includes reversed and equal positions; both must be rejected.
    return torch.logsumexp(scores.flatten(),0)-torch.logsumexp(targets,0)


def choice_logits(model,processor,row,features,config,fact,phase):
    import torch
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    prompt=grounding_prompt(fact,row['keys'],phase)
    inputs=engine().base.preprocess(processor,images_for(row,row['keys']),labels_for(row,row['keys']),prompt,config['max_pixels'])
    if inputs.input_ids.shape[1]>config['max_input_tokens']:raise ValueError('Grounding query over budget')
    if not torch.equal(inputs.image_grid_thw,features['grid'][row['keys']]):raise ValueError('Vision grid mismatch')
    inputs.pop('pixel_values');inputs=inputs.to('cuda')
    ids=[processor.tokenizer.encode(x,add_special_tokens=False) for x in 'ABCD']
    if any(len(x)!=1 for x in ids):raise ValueError('Frame choice must be one token')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out=model(**inputs,use_cache=False,logits_to_keep=1,
            mm_encoder_outputs={'image':BaseModelOutputWithPooling(pooler_output=tuple(features['tokens'][row['keys']].to('cuda').unbind(0)))})
    return out.logits[0,-1,[x[0] for x in ids]]


def grounded_loss(model,processor,row,features,config,fact):
    paths=support_sequences(fact,row['keys']);phases=['single'] if len(paths[0])==1 else ['before','after']
    logits=[choice_logits(model,processor,row,features,config,fact,phase) for phase in phases]
    return evidence_objective(logits,paths),logits,paths


def cache_path(row):return RUN/'features'/f'{row["id"]}.pt'


def features_for(row):
    import torch
    path=cache_path(row);meta=read_json(path.with_suffix('.json'))
    if sha256(path)!=meta['sha256']:raise ValueError('Changed feature cache')
    features=torch.load(path,weights_only=True,map_location='cpu')
    if features['frame_hashes']!=[f['sha256'] for f in row['frames']]:raise ValueError('Wrong cached source')
    return features


def config_and_parent():
    p=verify_data(RUN,training_only=True)
    audit=read_json(RUN/'data_audit.json')
    if audit['status']!='PASS' or audit['data_freeze_sha256']!=sha256(RUN/'data_freeze.json'):raise ValueError('Source/split audit must pass')
    return dict(p['config'],lora_rank=32,lora_alpha=64),checkpoint_path(PARENT/'teacher_r32')


def cache():
    import torch
    config,parent=config_and_parent();rows=sum([read_json(RUN/f'{s}.json')['records'] for s in ['train','validation']],[])
    pending=[]
    for row in rows:
        if cache_path(row).exists():features_for(row)
        else:pending.append(row)
    if pending:
        model,processor,runtime=load_model(config,parent/'adapter');model.requires_grad_(False);model.gradient_checkpointing_disable();model.eval()
        for row in pending:
            feat=encode(model,processor,row,config)
            feat={k:v.detach().cpu().clone() for k,v in feat.items()};feat['frame_hashes']=[f['sha256'] for f in row['frames']]
            path=cache_path(row);tensor_save(path,feat)
            write_json(path.with_suffix('.json'),dict(sha256=sha256(path),split=row['split'],data_freeze_sha256=sha256(RUN/'data_freeze.json')))
            print('CACHE',row['split'],row['id'],flush=True)
        del model,processor;release_gpu()
    return len(rows)


def model_digest(model):
    import hashlib
    digest=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad:
            digest.update(name.encode());digest.update(p.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def preflight():
    import torch
    from .qwen35_multitask import TemporalSelector
    config,parent=config_and_parent();rows=read_json(RUN/'train.json')['records']
    candidates=[r for r in rows if eligible_facts(r)]
    # Two real rows, including one new source and temporal evidence.
    new=[r for r in candidates if r['origin']=='grounded_expansion']
    temporal=[r for r in candidates if any(len(f['evidence_groups'])>1 for f in eligible_facts(r))]
    selected=[max(temporal,key=lambda r:len(r['caption'])),max(new,key=lambda r:len(r['caption']))]
    model,processor,runtime=load_model(config,parent/'adapter');head=TemporalSelector(hidden=config['head_hidden']).cuda().eval()
    head.load_state_dict(torch.load(parent/'head.pt',weights_only=True,map_location='cuda'));head.requires_grad_(False)
    before=model_digest(model);params=[p for p in model.parameters() if p.requires_grad];records=[]
    # Reserve two Adam moment tensors without creating/updating an optimizer.
    simulated_adam=[torch.zeros_like(p) for p in params for _ in range(2)]
    torch.cuda.reset_peak_memory_stats()
    for row in selected:
        feat=features_for(row);model.zero_grad(set_to_none=True);start=time.monotonic()
        ce=caption_loss(model,processor,head,fixed_prediction(row['keys']),feat,row,config);ce.backward();del ce
        model.zero_grad(set_to_none=True)
        fact=next((f for f in eligible_facts(row) if len(f['evidence_groups'])>1),eligible_facts(row)[0])
        loss,logits,paths=grounded_loss(model,processor,row,feat,config,fact);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm) or norm.item()<=0:raise ValueError('No finite auxiliary LoRA gradient')
        records.append(dict(id=row['id'],auxiliary_loss=float(loss.detach()),auxiliary_gradient_norm=float(norm),
            phases=len(logits),valid_paths=paths,seconds=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
        del loss,logits
    after=model_digest(model)
    if before!=after:raise ValueError('Preflight changed adapter weights')
    write_json(RUN/'preflight.json',dict(status='PASS_NO_WEIGHT_UPDATES',optimizer_steps=0,checkpoint_saved=False,
        before_adapter_digest=before,after_adapter_digest=after,simulated_adam_state=True,runtime=runtime,records=records,
        code_sha256=sha256(Path(__file__)),dependency_sha256=dependencies(),data_freeze_sha256=sha256(RUN/'data_freeze.json')))
    del model,processor,head,params,simulated_adam;release_gpu()
    print('PREFLIGHT_PASS_NO_TRAINING',records,flush=True)


def epoch_order(rows,epoch):
    result=list(rows);random.Random(SETTINGS['seed']+epoch).shuffle(result);return result


def validate(model,processor,head,rows,config):
    import torch
    model.eval();ce=[];evidence=[];hits=0
    with torch.no_grad():
        for row in rows:
            feat=features_for(row)
            ce.append(float(caption_loss(model,processor,head,fixed_prediction(row['keys']),feat,row,config)))
            facts=eligible_facts(row)
            if facts:
                loss,logits,paths=grounded_loss(model,processor,row,feat,config,facts[0]);evidence.append(float(loss))
                pred=tuple(int(x.argmax()) for x in logits)
                targets={(p[0],) if len(logits)==1 else (p[0],p[-1]) for p in paths};hits+=int(pred in targets)
    model.train();model.base_model.model.model.visual.eval()
    return dict(caption_ce=sum(ce)/len(ce),evidence_loss=sum(evidence)/len(evidence) if evidence else None,
        evidence_hits=hits,evidence_queries=len(evidence),scope='Teacher-forced/query diagnostics, not free-caption semantic accuracy')


def train(arm):
    import torch
    from .qwen35_multitask import TemporalSelector
    if arm not in SETTINGS['arms']:raise ValueError('Unknown training arm')
    config,parent=config_and_parent();cache()
    ready=read_json(RUN/'preflight.json')
    if ready['status']!='PASS_NO_WEIGHT_UPDATES' or ready['code_sha256']!=sha256(Path(__file__)) or ready['dependency_sha256']!=dependencies() or ready['data_freeze_sha256']!=sha256(RUN/'data_freeze.json'):
        raise ValueError('Matching no-update GPU preflight required; run preflight command first')
    stage=RUN/arm;rows=read_json(RUN/'train.json')['records'];val=read_json(RUN/'validation.json')['records']
    identity=dict(arm=arm,settings=SETTINGS,config=config,parent=str(parent),parent_sha256=sha256(parent/'complete.json'),
        data_freeze_sha256=sha256(RUN/'data_freeze.json'),implementation_sha256=sha256(Path(__file__)),
        data_audit_sha256=sha256(RUN/'data_audit.json'),
        data_implementation_sha256=sha256(Path(__file__).with_name('qwen35_grounded_data.py')),
        dependency_sha256=dependencies(),preflight_sha256=sha256(RUN/'preflight.json'),
        objective='caption mean-token CE'+(' +0.3 alternative-frame/ordered-endpoint support loss' if arm=='grounded' else ''),
        test_access=False,selector_frozen=True)
    if (stage/'protocol.json').exists() and read_json(stage/'protocol.json')!=identity:raise ValueError('Changed training protocol')
    if (stage/'RESULT.json').exists():checkpoint_path(stage);print('ALREADY_COMPLETE',arm);return
    write_json(stage/'protocol.json',identity)
    previous=checkpoint_path(stage,'latest') if (stage/'latest.json').exists() else None
    model,processor,runtime=load_model(config,(previous or parent)/'adapter');seed_all(SETTINGS['seed'])
    head=TemporalSelector(hidden=config['head_hidden']).cuda().eval()
    original=torch.load(parent/'head.pt',weights_only=True,map_location='cuda');head.load_state_dict(original);head.requires_grad_(False)
    params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=SETTINGS['lr'])
    state=dict(updates=0,next_epoch=0,next_offset=0,history=[],runtime=runtime,training_seconds=0.,initial_adapter_digest=model_digest(model),protocol_sha256=sha256(stage/'protocol.json'))
    if previous:
        state=read_json(previous/'state.json');opt.load_state_dict(torch.load(previous/'optimizer.pt',weights_only=True,map_location='cuda'))
        rng=torch.load(previous/'rng.pt',weights_only=True,map_location='cpu');torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state_all(rng['cuda']);random.setstate(rng['python'])
    else:
        state['initial_validation']=validate(model,processor,head,val,config);save_checkpoint(stage,head,model,opt,state)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(state['next_epoch'],SETTINGS['epochs']):
        order=epoch_order(rows,epoch);ce_values=[];aux_values=[];start=time.monotonic()
        for offset in range(0,len(order),SETTINGS['gradient_accumulation']):
            batch=order[offset:offset+SETTINGS['gradient_accumulation']];opt.zero_grad(set_to_none=True)
            for row in batch:
                feat=features_for(row);ce=caption_loss(model,processor,head,fixed_prediction(row['keys']),feat,row,config)
                ce_values.append(float(ce.detach()));(ce/len(batch)).backward();del ce
                facts=eligible_facts(row)
                if arm=='grounded' and facts:
                    fact=facts[(epoch+rows.index(row))%len(facts)]
                    aux,logits,_=grounded_loss(model,processor,row,feat,config,fact)
                    aux_values.append(float(aux.detach()));(SETTINGS['grounding_weight']*aux/len(batch)).backward();del aux,logits
            norm=torch.nn.utils.clip_grad_norm_(params,1.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite gradients')
            opt.step();state['updates']+=1
            write_json(stage/'status.json',dict(status='RUNNING',epoch=epoch+1,updates=state['updates'],examples=offset+len(batch)))
            print('TRAIN',arm,epoch+1,state['updates'],flush=True)
        state['training_seconds']+=time.monotonic()-start
        if any(not torch.equal(original[k],v) for k,v in head.state_dict().items()):raise ValueError('Frozen selector changed')
        metrics=validate(model,processor,head,val,config)
        state['history'].append(dict(epoch=epoch+1,caption_ce=sum(ce_values)/len(ce_values),auxiliary_loss=sum(aux_values)/len(aux_values) if aux_values else None,validation=metrics))
        state['next_epoch']=epoch+1;state['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
        save_checkpoint(stage,head,model,opt,state,best=epoch+1==SETTINGS['epochs'])
    write_json(stage/'RESULT.json',dict(status='COMPLETE_TRAINING_SEMANTIC_EVALUATION_PENDING',state=state,checkpoint=str(checkpoint_path(stage)),default_pipeline_changed=False))
    write_json(stage/'status.json',dict(status='COMPLETE',updates=state['updates']))
    del model,processor,head,params,opt;release_gpu()


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','freeze','verify','cache','preflight','caption','grounded','both']);args=p.parse_args()
    if args.command in ['prepare','freeze','verify']:
        fn={'prepare':prepare,'freeze':freeze,'verify':verify_data}[args.command];print(fn());return
    with ExitStack() as stack:
        for name in ['/tmp/qwen35_multitask_gpu.lock',str(ROOT/'.local/etri_60s_check.lock')]:
            lock=stack.enter_context(open(name,'w'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='cache':print('CACHED',cache())
        elif args.command=='preflight':preflight()
        elif args.command=='both':
            for arm in SETTINGS['arms']:train(arm)
        else:train(args.command)


if __name__=='__main__':main()
