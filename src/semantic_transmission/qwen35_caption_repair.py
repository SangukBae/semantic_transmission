"""Fixed-key caption repair: matched fresh LoRA CE and temporal-QA arms.

No selection gradients, test targets, or diagnostic captions enter training.
Every artifact is tied to source hashes and a pre-training protocol. Legacy
compound facts remain replay data; new references use smaller atomic items.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from itertools import combinations
from pathlib import Path
import random
import time

from .qwen35_training_data import ROOT, INVENTORY, DEFAULT_DATA, read_json, write_json, sha256, exclusion_keys
from .qwen35_oracle_diagnostic import witnessed

RUN = ROOT/'outputs/qwen35_caption_repair_v1'
SETTINGS = ROOT/'configs/qwen35_caption_repair_v1.json'
DIAG = ROOT/'outputs/qwen35_oracle_diagnostic_v1'
OLD = ROOT/'outputs/qwen35_fact_joint_v3/control'
ANNOTATIONS = ROOT/'configs/qwen35_fact_annotations_v3.json'


def fixed_prediction(keys, device='cuda'):
    import torch
    if len(keys) != 4 or keys != sorted(set(keys)) or any(type(k) is not int or not 0 <= k < 8 for k in keys):
        raise ValueError('Four distinct chronological keys required')
    indices = torch.tensor(keys, device=device)
    return {'indices': indices, 'matrix': torch.nn.functional.one_hot(indices, 8).float()}


def supported_facts(row, keys):
    return [f for f in row['facts'] if witnessed(f['evidence_groups'], keys)]


def legacy_target(row):
    """Ordered evidence, maximum action coverage, then total coverage and spread."""
    choices = [(0,*mid,7) for mid in combinations(range(1,7),2)]
    def score(k):
        facts = supported_facts(row,k)
        gaps = [b-a for a,b in zip(k,k[1:])]
        return (-sum(f['kind'] != 'state' for f in facts),-len(facts),max(gaps)-min(gaps),k)
    keys = list(min(choices,key=score))
    facts = supported_facts(row,keys)
    if not facts: raise ValueError('No grounded target')
    return dict(row, keys=keys, caption=' '.join(f['text'] for f in facts),
                qa_answer=' '.join(f['text'] for f in facts if f['kind'] != 'state'),
                target_fact_ids=[f['id'] for f in facts], origin='legacy_replay')


def assert_disjoint(rows):
    seen = {}
    ids = set()
    for row in rows:
        if row['id'] in ids: raise ValueError('Duplicate window')
        ids.add(row['id'])
        for field in ('source_id','source_sha256'):
            key=(field,row[field]); split=row['split']
            if key in seen and seen[key] != split: raise ValueError('Source crosses splits')
            seen[key]=split


def order_for(rows, settings, epoch):
    if any(r['split'] != 'train' for r in rows): raise ValueError('Non-training row in training order')
    result=[]
    for row in rows:
        result.extend([row]* (settings['new_train_repeats'] if row['origin']=='new_action' else 1))
    random.Random(settings['seed']+epoch).shuffle(result)
    return result


def prepare(run):
    from .qwen35_training import CONFIG, checkpoint_path
    from .qwen35_training_data import verify as verify_old
    from .qwen35_oracle_diagnostic import verify as verify_diag, reject_overlap
    from .qwen35_fact_training import revised_rows
    if (run/'freeze.json').exists(): return verify(run)
    verify_old(DEFAULT_DATA)
    dp=verify_diag(DIAG)
    settings=read_json(SETTINGS)
    old_split=read_json(DEFAULT_DATA/'source_split.json')['sources']
    sources={r['id']:r for r in old_split}
    protected_ids,protected_hashes,exclusion_files=exclusion_keys()
    diagnostic_ids={r['source_id'] for r in dp['rows']}
    diagnostic_hashes={r['source_sha256'] for r in dp['rows']}
    rows=[]; checked={}
    for split in ('train','validation'):
        rows.extend(legacy_target(r) for r in revised_rows(DEFAULT_DATA,ANNOTATIONS,split))
    for ref in settings['records']:
        source=sources[ref['video_id']]
        reject_overlap(source,old_split,protected_ids|diagnostic_ids,protected_hashes|diagnostic_hashes)
        folder=INVENTORY/source['dataset'].lower()/source['id']
        meta=read_json(folder/'source_prepared.json')
        if source['id'] not in checked:
            if sha256(source['source_path'])!=source['source_sha256'] or sha256(folder/'normalized.mp4')!=meta['normalized_sha256']:
                raise ValueError('Source video changed')
            checked[source['id']]=dict(source,normalized_path=str(folder/'normalized.mp4'),normalized_sha256=meta['normalized_sha256'])
        if len(ref['indices'])!=8 or ref['indices']!=sorted(set(ref['indices'])): raise ValueError('Bad candidate indices')
        frames=[]
        for idx in ref['indices']:
            path=folder/'source_frames'/f'{idx}.png'; digest=sha256(path)
            if digest!=meta['source_frame_hashes'][str(idx)]: raise ValueError('Source image changed')
            frames.append(dict(index=idx,path=str(path),sha256=digest))
        row=dict(ref,frames=frames,fps=24,source_id=source['source_id'],source_sha256=source['source_sha256'],origin='new_action')
        fixed_prediction(row['keys'],'cpu')
        if len(supported_facts(row,row['keys']))!=len(row['facts']): raise ValueError(f'Unsupported fixed-key reference: {row["id"]}')
        if row['caption']!=' '.join(f['text'] for f in row['facts']): raise ValueError('Target differs from source facts')
        if row['qa_answer']!=' '.join(f['text'] for f in row['facts'] if f['kind']=='event'): raise ValueError('Unsupported QA')
        rows.append(row)
    regression=[dict(r,split='regression',keys=r['oracle']['keys'],origin='known_diagnostic') for r in dp['rows']]
    assert_disjoint(rows+regression)
    for split in ('train','validation','test','regression'):
        write_json(run/f'{split}.json',{'records':[r for r in rows+regression if r['split']==split]})
    checkpoint=checkpoint_path(OLD)
    deps=[SETTINGS,ANNOTATIONS,Path(__file__),ROOT/'scripts/run_qwen35_caption_repair.sh',
          ROOT/'src/semantic_transmission/qwen35_training.py',ROOT/'src/semantic_transmission/qwen35_training_data.py',
          ROOT/'src/semantic_transmission/qwen35_ablation.py',ROOT/'src/semantic_transmission/qwen35_multitask.py',
          ROOT/'src/semantic_transmission/qwen35_fact_training.py',ROOT/'src/semantic_transmission/qwen35_oracle_diagnostic.py',
          ROOT/'scripts/qwen35_advanced_caption.py',ROOT/'src/semantic_transmission/qwen35_backend.py']
    protocol=dict(settings=settings,config=read_json(CONFIG),checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint/'complete.json'),
        dependencies={str(p.relative_to(ROOT)):sha256(p) for p in deps},sources=list(checked.values()),
        base_freeze_sha256=sha256(DEFAULT_DATA/'freeze.json'),diagnostic_freeze_sha256=sha256(DIAG/'reference_freeze.json'),
        exclusion_files=exclusion_files,review='Same-assistant provisional source review; not independent GPT6-Astra or human ground truth',
        regression_role='Previously inspected development diagnostic, never training or checkpoint selection',
        key_policy='Manually evidence-grounded four-frame inputs fixed across captioners. New oracle keys need not retain endpoints. No learned selection is evaluated.',
        training_splits=['train'],checkpoint_selection_splits=['validation'],test_used_for_selection=False,
        training_examples_per_epoch=len(order_for([r for r in rows if r['split']=='train'],settings,0)))
    write_json(run/'protocol.json',protocol)
    write_json(run/'freeze.json',{'protocol_sha256':sha256(run/'protocol.json'),
        'files':{f'{s}.json':sha256(run/f'{s}.json') for s in ('train','validation','test','regression')},'before_training_and_generation':True})
    verify(run)
    print('FROZEN',len(rows),len(regression),'regression',flush=True)
    return protocol


def verify(run):
    freeze=read_json(run/'freeze.json')
    if freeze['protocol_sha256']!=sha256(run/'protocol.json'): raise ValueError('Protocol changed')
    p=read_json(run/'protocol.json')
    for name,digest in freeze['files'].items():
        if sha256(run/name)!=digest: raise ValueError(f'Frozen data changed: {name}')
    for name,digest in p['dependencies'].items():
        if sha256(ROOT/name)!=digest: raise ValueError(f'Frozen dependency changed: {name}')
    ck=Path(p['checkpoint'])
    if sha256(ck/'complete.json')!=p['checkpoint_sha256']: raise ValueError('Original checkpoint changed')
    for name,digest in read_json(ck/'complete.json')['files'].items():
        if sha256(ck/name)!=digest: raise ValueError('Original checkpoint file changed')
    rows=[r for s in ('train','validation','test','regression') for r in read_json(run/f'{s}.json')['records']]
    assert_disjoint(rows)
    for row in rows:
        for f in row['frames']:
            if sha256(f['path'])!=f['sha256']: raise ValueError('Frozen source frame changed')
    return p


def cache_path(run,row):
    return (DEFAULT_DATA if row['origin']=='legacy_replay' else run)/'features'/f'{row["id"]}.pt'


def features_for(run,row):
    import torch
    path=cache_path(run,row); meta=read_json(path.with_suffix('.json'))
    if meta['sha256']!=sha256(path): raise ValueError('Feature cache hash mismatch')
    feat=torch.load(path,map_location='cpu',weights_only=True)
    if feat['frame_hashes']!=[f['sha256'] for f in row['frames']]: raise ValueError('Feature source mismatch')
    return feat


def cache(run,p):
    import torch
    from .qwen35_training import engine,release_gpu,tensor_save
    from .qwen35_ablation import encode
    rows=[r for s in ('train','validation') for r in read_json(run/f'{s}.json')['records']]
    pending=[]
    for row in rows:
        if cache_path(run,row).exists(): features_for(run,row)
        elif row['origin']=='legacy_replay': raise ValueError('Missing frozen legacy feature cache')
        else: pending.append(row)
    if not pending:return
    model,processor,runtime=engine().load('nf4')
    for row in pending:
        feat=encode(model,processor,row,p['config'])
        feat={k:v.detach().cpu().clone() for k,v in feat.items()}
        feat['frame_hashes']=[f['sha256'] for f in row['frames']]
        path=cache_path(run,row);tensor_save(path,feat)
        write_json(path.with_suffix('.json'),dict(sha256=sha256(path),freeze_sha256=sha256(run/'freeze.json'),runtime=runtime))
        print('CACHE',row['id'],flush=True)
    del model,processor
    release_gpu()


def train(run,arm):
    import torch
    from .qwen35_training import load_lora,seed_all,caption_loss,save_checkpoint,checkpoint_path,release_gpu
    from .qwen35_multitask import TemporalSelector
    p=verify(run);settings=p['settings'];config=p['config'];stage=run/arm
    if arm not in settings['arms']:raise ValueError('Unknown arm')
    if (stage/'RESULT.json').exists():
        checkpoint_path(stage);print('TRAIN_COMPLETE',arm,flush=True);return
    cache(run,p)
    train_rows=read_json(run/'train.json')['records']; val=read_json(run/'validation.json')['records']
    identity=dict(freeze_sha256=sha256(run/'freeze.json'),arm=arm,objective='caption_mean_CE + weight * action_QA_mean_CE',
                  auxiliary_weight=settings['auxiliary_weight'] if arm=='caption_qa' else 0.)
    if (stage/'protocol.json').exists() and read_json(stage/'protocol.json')!=identity: raise ValueError('Changed arm')
    write_json(stage/'protocol.json',identity)
    previous=checkpoint_path(stage,'latest') if (stage/'latest.json').exists() else None
    model,processor,runtime=load_lora(config,previous/'adapter' if previous else None)
    seed_all(settings['seed'])
    head=TemporalSelector(hidden=config['head_hidden']).cuda().eval()
    original=torch.load(Path(p['checkpoint'])/'head.pt',weights_only=True,map_location='cuda')
    head.load_state_dict(original);head.requires_grad_(False)
    params=[v for v in model.parameters() if v.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=settings['lr'])
    state=dict(updates=0,next_epoch=0,next_offset=0,history=[],best_validation_loss=None,
               protocol_sha256=sha256(stage/'protocol.json'),runtime=runtime,training_seconds=0.,validation_seconds=0.,
               head_unchanged=True,initialization='fresh_base_LoRA',auxiliary_examples=0)
    if previous:
        state=read_json(previous/'state.json')
        if state['protocol_sha256']!=sha256(stage/'protocol.json'):raise ValueError('Resume mismatch')
        optimizer.load_state_dict(torch.load(previous/'optimizer.pt',weights_only=True,map_location='cuda'))
        rng=torch.load(previous/'rng.pt',weights_only=True,map_location='cpu')
        torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state_all(rng['cuda']);random.setstate(rng['python'])
    features={r['id']:features_for(run,r) for r in train_rows+val}
    def loss(row,qa=False):
        target=dict(row,caption=row['qa_answer']) if qa else row
        cfg=dict(config,caption_prompt=settings['auxiliary_prompt']) if qa else config
        return caption_loss(model,processor,head,fixed_prediction(row['keys']),features[row['id']],target,cfg)
    def validate(epoch):
        start=time.monotonic(); model.eval()
        with torch.no_grad(): scores=[float(loss(row)) for row in val]
        mean=sum(scores)/len(scores)
        history=dict(epoch=epoch,caption_ce=mean,legacy_ce=sum(scores[:16])/16,new_ce=sum(scores[16:])/4)
        state['history'].append(history);state['validation_seconds']+=time.monotonic()-start
        best=state['best_validation_loss'] is None or mean<state['best_validation_loss']
        if best:state['best_validation_loss']=mean
        model.train();model.base_model.model.model.visual.eval()
        print('VALIDATION',arm,history,'best',best,flush=True)
        return best
    torch.cuda.reset_peak_memory_stats()
    if previous is None:
        best=validate(0);save_checkpoint(stage,head,model,optimizer,state,best=best)
    for epoch in range(state['next_epoch'],settings['epochs']):
        order=order_for(train_rows,settings,epoch);acc=settings['gradient_accumulation']
        start=time.monotonic();ce_values=[];qa_values=[]
        for offset in range(0,len(order),acc):
            batch=order[offset:offset+acc];optimizer.zero_grad(set_to_none=True)
            for row in batch:
                ce=loss(row);ce_values.append(float(ce.detach()));(ce/len(batch)).backward();del ce
                if identity['auxiliary_weight'] and row['qa_answer']:
                    qa=loss(row,True);qa_values.append(float(qa.detach()))
                    (qa*identity['auxiliary_weight']/len(batch)).backward();del qa
                    state['auxiliary_examples']+=1
            norm=torch.nn.utils.clip_grad_norm_(params,1.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite gradients')
            if any(v.grad is not None for v in head.parameters()):raise ValueError('Frozen selector received gradients')
            optimizer.step();state['updates']+=1
            write_json(stage/'status.json',dict(status='RUNNING',epoch=epoch+1,updates=state['updates'],offset=offset+len(batch),examples=len(order)))
            print('TRAIN',arm,epoch+1,state['updates'],round(sum(ce_values[-len(batch):])/len(batch),4),flush=True)
        state['training_seconds']+=time.monotonic()-start
        if any(not torch.equal(original[k],v) for k,v in head.state_dict().items()):raise ValueError('Frozen selector changed')
        state['next_epoch']=epoch+1
        state['last_train_ce']=sum(ce_values)/len(ce_values)
        state['last_qa_ce']=sum(qa_values)/len(qa_values) if qa_values else None
        state['peak_gpu_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
        best=validate(epoch+1)
        save_checkpoint(stage,head,model,optimizer,state,best=best)
    best_path=checkpoint_path(stage)
    write_json(stage/'RESULT.json',dict(status='COMPLETE',arm=arm,best_checkpoint=str(best_path),
        best_state=read_json(best_path/'state.json'),final_state=state,head_frozen=True))
    write_json(stage/'status.json',dict(status='COMPLETE',updates=state['updates']))
    del model,processor,head,optimizer,params
    release_gpu()


def generate(run):
    import torch
    from .qwen35_training import checkpoint_path,release_gpu
    from .qwen35_ablation import load_inference,encode,generate as caption
    p=verify(run);checkpoints={arm:checkpoint_path(run/arm) for arm in p['settings']['arms']}
    identity=dict(freeze_sha256=sha256(run/'freeze.json'),checkpoints={arm:dict(path=str(ck),sha256=sha256(ck/'complete.json')) for arm,ck in checkpoints.items()},
        cells=['base','previous_joint','caption_ce','caption_qa'],config=p['config'],selection='Fixed identical source-reference keys; no target text enters inference')
    dest=run/'evaluation';protocol=dest/'protocol.json'
    if protocol.exists() and read_json(protocol)!=identity:raise ValueError('Generation checkpoint changed')
    write_json(protocol,identity)
    rows=[r for s in ('validation','test','regression') for r in read_json(run/f'{s}.json')['records'] if r['origin']!='legacy_replay']
    pending=[r for r in rows if not (dest/'records'/f'{r["id"]}.json').exists()]
    if pending:
        start=time.monotonic();model,processor,head,runtime=load_inference(p['config'],Path(p['checkpoint']))
        for arm,ck in checkpoints.items():model.load_adapter(ck/'adapter',adapter_name=arm,is_trainable=False)
        model.eval();load_seconds=time.monotonic()-start;torch.cuda.reset_peak_memory_stats()
        for row in pending:
            begin=time.monotonic();features=encode(model,processor,row,p['config'])
            torch.cuda.synchronize();encode_seconds=time.monotonic()-begin
            cells={};order=identity['cells'] if row['window']%2==0 else list(reversed(identity['cells']))
            for cell in order:
                model.set_adapter(cell if cell in checkpoints else 'default')
                output=caption(model,processor,row,row['keys'],features,p['config'],adapter=cell!='base')
                cells[cell]=dict(output,keys=row['keys'],source_hashes=[row['frames'][i]['sha256'] for i in row['keys']])
                print('GENERATE',row['id'],cell,output['status'],flush=True)
            if len({c['input_tokens'] for c in cells.values()})!=1:raise ValueError('Input mismatch')
            write_json(dest/'records'/f'{row["id"]}.json',dict(id=row['id'],split=row['split'],cells=cells,encode_seconds=encode_seconds,
                model_load_seconds=load_seconds,runtime=runtime,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,protocol_sha256=sha256(protocol)))
            del features
        del model,processor,head
        release_gpu()
    files={f'records/{r["id"]}.json':sha256(dest/'records'/f'{r["id"]}.json') for r in rows}
    for name in files:
        if read_json(dest/name)['protocol_sha256']!=sha256(protocol):raise ValueError('Stale generation')
    write_json(dest/'freeze.json',dict(protocol_sha256=sha256(protocol),files=files))
    write_json(dest/'RESULT.json',dict(status='GENERATED_REVIEW_PENDING',windows=len(rows),condition_windows=len(rows)*4))


def main():
    import fcntl
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','verify','caption_ce','caption_qa','generate','all'])
    parser.add_argument('--run',type=Path,default=RUN)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.run);return
    if args.command=='verify':verify(args.run);print('VERIFIED');return
    with ExitStack() as stack:
        for name in ('/tmp/qwen35_multitask_gpu.lock',ROOT/'.local/etri_60s_check.lock'):
            lock=stack.enter_context(open(name,'w'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='all':
            prepare(args.run)
            for arm in read_json(SETTINGS)['arms']:train(args.run,arm)
            generate(args.run)
        elif args.command=='generate':generate(args.run)
        else:train(args.run,args.command)

if __name__=='__main__':main()
