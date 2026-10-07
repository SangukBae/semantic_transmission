"""Frozen-weight same-image observation->caption diagnostic, no training."""
import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import time
from .qwen35_training_data import ROOT,read_json,write_json,sha256
from .qwen35_astra_transfer import RUN as PARENT,load_model
from .qwen35_training import checkpoint_path,release_gpu
from .qwen35_ablation import encode,generate

RUN=ROOT/'outputs/qwen35_reasoning_probe_v1'
OBSERVATION_PROMPT='''Inspect only these four supplied source images, labeled by original i indices and time. First identify visible objects and their positions. Compare those same objects across images: report small movements, direction, occlusion and camera movement only if established. Distinguish moving objects from changed framing. Do not infer intent, sounds, species or hidden events. A stationary scene is allowed. Return concise JSON only, at most 5 observations and 3 changes:
{"observations":[{"id":"o1","fact":"one visible fact","evidence":[0],"uncertain":false}],"changes":[{"fact":"one visible change","before":[0],"after":[7],"cause":"object_motion|camera_motion|occlusion|cut|uncertain"}],"uncertainties":["unresolved detail"]}.
Use only indices actually supplied. An evidence list can contain alternatives. Every change needs ordered before/after evidence. Do not copy these example indices unless supported. Do not invent motion to fill the change list.'''


def parse_observations(text,keys):
    try:
        clean=text.strip()
        if clean.startswith('```'):clean='\n'.join(clean.splitlines()[1:-1])
        obj=json.loads(clean);allowed=set(keys)
        if set(obj)!={'observations','changes','uncertainties'}:raise ValueError('schema keys')
        if not isinstance(obj['uncertainties'],list) or not 1<=len(obj['observations'])<=5 or len(obj['changes'])>3:raise ValueError('counts')
        for o in obj['observations']:
            if not o['fact'] or type(o['uncertain']) is not bool or not o['evidence'] or not set(o['evidence'])<=allowed:raise ValueError('observation')
        for c in obj['changes']:
            if not c['fact'] or not c['before'] or not c['after'] or not set(c['before']+c['after'])<=allowed or max(c['before'])>=min(c['after']):raise ValueError('change evidence')
            if c['cause'] not in ['object_motion','camera_motion','occlusion','cut','uncertain']:raise ValueError('cause')
        return obj,None
    except (ValueError,KeyError,TypeError) as exc:return None,str(exc)


def run():
    import torch
    p=read_json(PARENT/'protocol.json');ck=checkpoint_path(PARENT/'teacher_r32')
    rows=sum([read_json(PARENT/f'{s}_inputs.json')['records'] for s in ['validation','test']],[])
    config=dict(p['config'],lora_rank=32,lora_alpha=64)
    protocol=dict(parent_checkpoint=str(ck),parent_checkpoint_sha256=sha256(ck/'complete.json'),config=config,
        rows=rows,observation_prompt=OBSERVATION_PROMPT,observation_tokens=512,caption_tokens=160,
        implementation_sha256=sha256(Path(__file__)),
        scope='Previously evaluated24 windows are development diagnostics only. No new19-source split, teacher facts or captions enter generation. No optimizer, backward or weight updates.',
        comparison='Identical four images/model. Structured condition adds one observation call and conditions caption on unverified generated notes; compute/token budget is larger and measured.',
        dependencies={str(q.relative_to(ROOT)):sha256(q) for q in [Path(__file__).with_name('qwen35_astra_transfer.py'),Path(__file__).with_name('qwen35_ablation.py')]})
    path=RUN/'protocol.json'
    if path.exists() and read_json(path)!=protocol:raise ValueError('Changed probe protocol')
    write_json(path,protocol)
    for row in rows:
        for k in row['keys']:
            f=row['frames'][k]
            if sha256(f['path'])!=f['sha256']:raise ValueError('Source changed')
    pending=[r for r in rows if not (RUN/'records'/f'{r["id"]}.json').exists()]
    if pending:
        model,processor,runtime=load_model(config,ck/'adapter');model.requires_grad_(False);model.gradient_checkpointing_disable();model.eval()
        torch.cuda.reset_peak_memory_stats()
        for row in pending:
            start=time.monotonic();features=encode(model,processor,row,config);torch.cuda.synchronize();encoding=time.monotonic()-start
            outputs={}
            # Alternate complete condition order, while observation precedes its caption.
            for condition in (['direct','structured'] if row['window']%2==0 else ['structured','direct']):
                if condition=='direct':outputs['direct']=generate(model,processor,row,row['keys'],features,config)
                else:
                    obs=generate(model,processor,row,row['keys'],features,dict(config,caption_prompt=OBSERVATION_PROMPT,max_caption_tokens=512))
                    parsed,error=parse_observations(obs['caption'],row['keys']);outputs['analysis']=dict(obs,parsed=parsed,parse_error=error)
                    prompt=config['caption_prompt']+'\nBelow is an unverified observation draft from the same model. Use it only as a checklist. Recheck against the supplied images, discard unsupported statements, and preserve supported specific objects, positions and changes. Do not describe the draft or output JSON.\n<draft>\n'+obs['caption']+'\n</draft>'
                    outputs['structured']=generate(model,processor,row,row['keys'],features,dict(config,caption_prompt=prompt))
            for out in outputs.values():out.update(keys=row['keys'],source_hashes=[row['frames'][i]['sha256'] for i in row['keys']])
            write_json(RUN/'records'/f'{row["id"]}.json',dict(id=row['id'],original_split=row['split'],cells=outputs,encode_seconds=encoding,
                structured_total_seconds=outputs['analysis']['seconds']+outputs['structured']['seconds'],runtime=runtime,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,protocol_sha256=sha256(path)))
            print('PROBE',row['id'],outputs['analysis']['status'],error or 'VALID_JSON',flush=True)
        del model,processor,features;release_gpu()
    if sha256(ck/'complete.json')!=protocol['parent_checkpoint_sha256']:raise ValueError('Checkpoint changed')
    write_json(RUN/'freeze.json',dict(protocol_sha256=sha256(path),files={f'records/{r["id"]}.json':sha256(RUN/'records'/f'{r["id"]}.json') for r in rows}))
    write_json(RUN/'RESULT.json',dict(status='GENERATED_REVIEW_PENDING',windows=len(rows),training_started=False))


if __name__=='__main__':
    with ExitStack() as stack:
        for p in ['/tmp/qwen35_multitask_gpu.lock',str(ROOT/'.local/etri_60s_check.lock')]:
            f=stack.enter_context(open(p,'w'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
