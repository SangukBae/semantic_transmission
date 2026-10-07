"""Matched free-caption evaluation after user-completed grounded training."""
import argparse
from contextlib import ExitStack
import fcntl
from pathlib import Path
import time
from .qwen35_grounded_data import RUN as DATA,PARENT,verify_data
from .qwen35_training_data import ROOT,read_json,write_json,sha256
from .qwen35_training import checkpoint_path,release_gpu
from .qwen35_astra_transfer import load_model
from .qwen35_ablation import encode,generate

RUN=DATA/'evaluation'
ARMS=['initial','caption','grounded']


def cell_order(ordinal):
    offset=ordinal%len(ARMS)
    return ARMS[offset:]+ARMS[:offset]


def prepare():
    p=verify_data();rows=sum([read_json(DATA/f'{s}_inputs.json')['records'] for s in ['validation','test']],[])
    checkpoints={'initial':checkpoint_path(PARENT/'teacher_r32')}
    for arm in ['caption','grounded']:
        result=read_json(DATA/arm/'RESULT.json')
        if result['status']!='COMPLETE_TRAINING_SEMANTIC_EVALUATION_PENDING' or result['state']['updates']!=62 or result['state']['next_epoch']!=2:raise ValueError('Training not complete')
        checkpoints[arm]=checkpoint_path(DATA/arm)
    config=dict(p['config'],lora_rank=32,lora_alpha=64)
    protocol=dict(arms=ARMS,config=config,rows=rows,
        checkpoints={a:dict(path=str(ck),complete_sha256=sha256(ck/'complete.json')) for a,ck in checkpoints.items()},
        data_freeze_sha256=sha256(DATA/'data_freeze.json'),implementation_sha256=sha256(Path(__file__)),
        dependency_sha256={name:sha256(Path(__file__).with_name(name)) for name in ['qwen35_ablation.py','qwen35_astra_transfer.py','qwen35_grounded_data.py']},
        initial_definition='Astra teacher rank32 parent before this124-window additional training; not raw unadapted Qwen.',
        input_policy='Identical fixed4 images, timestamps, resolution and caption prompt. No fact, caption, motion or evidence target in inference. One frozen vision encoding per window.',
        checkpoint_policy='Both predeclared final epoch2 checkpoints; no selection from new test outputs.',
        compute_policy='One Qwen backbone with three frozen LoRA adapters, rotated condition order per window. No optimizer or backward.',
        reference_policy='Frozen same4-image Astra facts made before student outputs. Opposite author checks source and grades anonymous captions. Original facts remain primary; source-only disputed facts reported separately.',
        gates='For each validation/test split, grounded fact/action hits >= initial AND caption; error/unsupported windows and failures <= both, with a strict gain against each. Report full+partial and source counts separately.',
        timing_policy='Per-call generation includes preprocessing/decode, excludes shared vision encoding, model/adapter loading, judging and transmission/reconstruction.',
        limitations='7 held-out WebVid originals,28 windows; test4 originals all stable-camera. Same-family AI labels and judging, not human truth. Fixed-key caption comparison, not selector or end-to-end improvement.')
    path=RUN/'protocol.json'
    if path.exists() and read_json(path)!=protocol:raise ValueError('Changed evaluation protocol')
    write_json(path,protocol)
    labels={r['id']:r for s in ['validation','test'] for r in read_json(DATA/f'{s}.json')['records']}
    for author in ['A','B']:
        source=read_json(DATA/'teacher'/f'packet_{author}.json')
        opposite='B' if author=='A' else 'A'
        items=[dict(r,facts=labels[r['id']]['facts'],source_author=author,sheet_sha256=sha256(r['sheet'])) for r in source['records'] if r['split']!='train']
        obj=dict(author=opposite,records=items,data_freeze_sha256=sha256(DATA/'data_freeze.json'),
            rules='Source-only audit before viewing candidate captions. Each frozen fact supported, partial or unsupported by these4 images. Record concerns, never invent replacements or read model outputs.')
        rp=RUN/f'reference_packet_{opposite}.json'
        if rp.exists() and read_json(rp)!=obj:raise ValueError('Changed reference packet')
        write_json(rp,obj)
    return protocol


def verify():
    p=read_json(RUN/'protocol.json')
    if sha256(DATA/'data_freeze.json')!=p['data_freeze_sha256']:raise ValueError('Changed data')
    if sha256(Path(__file__))!=p['implementation_sha256']:raise ValueError('Changed generator')
    for name,h in p['dependency_sha256'].items():
        if sha256(Path(__file__).with_name(name))!=h:raise ValueError('Changed dependency')
    for arm,ck in p['checkpoints'].items():
        path=Path(ck['path'])
        if sha256(path/'complete.json')!=ck['complete_sha256']:raise ValueError('Changed checkpoint manifest')
        for rel,h in read_json(path/'complete.json')['files'].items():
            if sha256(path/rel)!=h:raise ValueError('Changed checkpoint file')
    for row in p['rows']:
        for k in row['keys']:
            f=row['frames'][k]
            if sha256(f['path'])!=f['sha256']:raise ValueError('Changed image')
    return p


def validate_record(row,record,protocol_hash):
    if record['id']!=row['id'] or record['protocol_sha256']!=protocol_hash or set(record['cells'])!=set(ARMS):raise ValueError('Wrong generation record')
    first=record['cells']['initial']
    for arm,c in record['cells'].items():
        if c['keys']!=row['keys'] or c['source_hashes']!=[row['frames'][k]['sha256'] for k in row['keys']]:raise ValueError('Changed source keys')
        if c['input_tokens']!=first['input_tokens'] or c['grid']!=first['grid']:raise ValueError('Unmatched input budget')


def run():
    import torch
    p=prepare();verify();ph=sha256(RUN/'protocol.json');pending=[];files={}
    for row in p['rows']:
        path=RUN/'records'/f'{row["id"]}.json'
        if path.exists():
            if sha256(path)!=read_json(path.with_suffix('.receipt.json'))['sha256']:raise ValueError('Changed output')
            validate_record(row,read_json(path),ph)
        else:pending.append(row)
    if pending:
        model,processor,runtime=load_model(p['config'],Path(p['checkpoints']['initial']['path'])/'adapter')
        start=time.monotonic()
        for arm in ['caption','grounded']:
            model.load_adapter(str(Path(p['checkpoints'][arm]['path'])/'adapter'),adapter_name=arm,is_trainable=False)
        runtime['additional_adapter_load_seconds']=time.monotonic()-start
        model.requires_grad_(False);model.gradient_checkpointing_disable();model.eval()
        adapters={'initial':'default','caption':'caption','grounded':'grounded'}
        torch.cuda.reset_peak_memory_stats()
        for row in pending:
            start=time.monotonic();features=encode(model,processor,row,p['config']);torch.cuda.synchronize();encoding=time.monotonic()-start
            cells={};order=cell_order(p['rows'].index(row))
            for arm in order:
                model.set_adapter(adapters[arm]);model.requires_grad_(False);model.eval()
                c=generate(model,processor,row,row['keys'],features,p['config'])
                if c['input_tokens']>p['config']['max_input_tokens']:raise ValueError('Exceeded input budget')
                cells[arm]=dict(c,keys=row['keys'],source_hashes=[row['frames'][k]['sha256'] for k in row['keys']],adapter_name=adapters[arm])
                print('EVAL',row['id'],arm,c['status'],flush=True)
            record=dict(id=row['id'],split=row['split'],cells=cells,order=order,encode_seconds=encoding,runtime=runtime,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,protocol_sha256=ph)
            validate_record(row,record,ph);path=RUN/'records'/f'{row["id"]}.json';write_json(path,record)
            write_json(path.with_suffix('.receipt.json'),dict(sha256=sha256(path),protocol_sha256=ph))
            del features
        del model,processor;release_gpu()
    verify()
    for row in p['rows']:
        name=f'records/{row["id"]}.json';validate_record(row,read_json(RUN/name),ph);files[name]=sha256(RUN/name)
    write_json(RUN/'generation_freeze.json',dict(protocol_sha256=ph,files=files))
    write_json(RUN/'RESULT.json',dict(status='GENERATED_SEMANTIC_REVIEW_PENDING',windows=len(p['rows']),condition_windows=len(p['rows'])*3,optimizer_steps=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','generate','verify']);a=p.parse_args()
    if a.command=='prepare':prepare();print('PREPARED')
    elif a.command=='verify':verify();print('VERIFIED')
    else:
        with ExitStack() as stack:
            for name in ['/tmp/qwen35_multitask_gpu.lock',str(ROOT/'.local/etri_60s_check.lock')]:
                lock=stack.enter_context(open(name,'w'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run()
