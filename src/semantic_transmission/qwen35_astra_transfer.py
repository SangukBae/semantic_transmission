"""Matched-input teacher labels and crossed data/rank caption pilot.

Existing checkpoints and frozen datasets are never modified. Teacher packets
contain only the exact source inputs, before any candidate generation.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import time

from .qwen35_training_data import ROOT, INVENTORY, DEFAULT_DATA, read_json, write_json, sha256, exclusion_keys
from .qwen35_caption_repair import assert_disjoint, fixed_prediction, features_for, order_for

RUN = ROOT / 'outputs/qwen35_astra_transfer_v1'
PARENT = ROOT / 'outputs/qwen35_caption_repair_v1'
ARMS = {'control_r8': ('control', 8), 'teacher_r8': ('teacher', 8),
        'control_r32': ('control', 32), 'teacher_r32': ('teacher', 32)}
NEW_SOURCES = {
    'validation': ['webvid_033_b069308f', 'webvid_037_143ffe14', 'tvsum_009_a66202fe', 'tvsum_011_0e11bb5b'],
    'test': ['webvid_041_681960eb', 'webvid_052_f5481395', 'tvsum_020_6960e885', 'tvsum_022_2411ac74'],
}


def display_sheet(row, keys, path):
    from PIL import Image, ImageDraw
    images = [Image.open(row['frames'][k]['path']).convert('RGB') for k in keys]
    width = max(im.width for im in images)
    height = max(im.height for im in images)
    sheet = Image.new('RGB', (2 * width, ((len(images) + 1) // 2) * (height + 26)), 'white')
    draw = ImageDraw.Draw(sheet)
    for n, (key, im) in enumerate(zip(keys, images)):
        x, y = (n % 2) * width, (n // 2) * (height + 26)
        draw.text((x + 4, y + 4), f"i{key} | frame {row['frames'][key]['index']} | {row['frames'][key]['index']/row['fps']:.3f}s", fill='black')
        sheet.paste(im, (x, y + 26))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return str(path)


def prepare(run=RUN):
    if (run / 'input_freeze.json').exists():
        return verify_inputs(run)
    from .qwen35_caption_repair import verify as verify_parent
    parent = verify_parent(PARENT)
    train = read_json(PARENT / 'train.json')['records']
    inventory = {v['id']: v for v in read_json(DEFAULT_DATA / 'source_split.json')['sources']}
    reserve = {s['id'] for s in read_json(ROOT / 'outputs/qwen35_astra_transfer_plan_20261005/readiness.json')['inventory']['reserve_sources']}
    ids, hashes, exclusion_files = exclusion_keys()
    rows = []
    source_audit = []
    for split, sources in NEW_SOURCES.items():
        for video_id in sources:
            source = inventory[video_id]
            if video_id not in reserve or source['source_id'] in ids or source['source_sha256'] in hashes:
                raise ValueError('Protected or previously used source')
            if sha256(source['source_path']) != source['source_sha256']:
                raise ValueError('Raw source changed')
            folder = INVENTORY / source['dataset'].lower() / video_id
            meta = read_json(folder / 'source_prepared.json')
            if sha256(folder / 'normalized.mp4') != meta['normalized_sha256']:
                raise ValueError('Normalized source changed')
            source_audit.append(source)
            count = len(meta['source_frame_hashes'])
            span = min(95, count // 4 - 1)
            for w, fraction in enumerate([.2, .5, .8]):
                start = max(0, min(count - span - 1, round(fraction * count) - span // 2))
                indices = [start + round(i * span / 7) for i in range(8)]
                frames = []
                for index in indices:
                    path = folder / 'source_frames' / f'{index}.png'
                    digest = sha256(path)
                    if digest != meta['source_frame_hashes'][str(index)]:
                        raise ValueError('Source frame changed')
                    frames.append(dict(index=index, path=str(path), sha256=digest))
                rows.append(dict(id=f'n_{video_id}_w{w}', video_id=video_id,
                    source_id=source['source_id'], source_sha256=source['source_sha256'],
                    source_video=source['source_path'], frames=frames, fps=24,
                    split=split, start=start, end_exclusive=start+span+1, window=w,
                    keys=[0, 2, 5, 7], origin='new_matched_teacher'))
    assert_disjoint(train + rows)
    config = copy.deepcopy(parent['config'])
    protocol = dict(version=1, parent_freeze_sha256=sha256(PARENT/'freeze.json'),
        seed=20261005, epochs=2, lr=2e-5, gradient_accumulation=4, new_train_repeats=2,
        arms=ARMS, config=config, checkpoint=parent['checkpoint'],
        checkpoint_policy='Predeclared final epoch 2 in all arms; no CE/test based checkpoint selection.',
        objective='Caption mean-token CE in all four arms; teacher versus original target and rank crossed.',
        alpha_policy='alpha=2*rank; same target layers and optimizer schedule',
        teacher_model='gpt-6-astra', existing_agents=['/root/astra_reference','/root/astra_judge_late'],
        teacher_input_policy='4-key packets contain exactly the same source PNGs and chronological labels as Qwen; no hidden frames or old captions. Full native pixels in sheets without resizing.',
        teacher_eight_frame_policy='Only release 8-frame packets after all four-frame labels have been written and frozen; diagnostic expansion, not training.',
        fixed_keys=True, selector_frozen=True, transmission_or_reconstruction=False,
        source_selection='Eight previously unused pilot sources, purposive dataset balance; deterministic 20/50/80 percent four-second windows; no output-dependent selection.',
        human_truth=False, independent_judge_policy='Swap existing Astra annotators for evaluation; same model family, no human adjudication.',
        gate='On validation AND test, teacher arm fact/action recall >= base and matched-rank control, clear-error windows <= both, and at least one strict gain per split. Conditional temporal/selector extension only after this pilot gate.',
        source_audit=source_audit, exclusion_files=exclusion_files,
        limited_scope='76 existing train windows and 24 new validation/test windows; not the planned 1000-window expansion or broad generalization proof.')
    write_json(run/'protocol.json', protocol)
    write_json(run/'train_control.json', {'records':train})
    for split in NEW_SOURCES:
        write_json(run/f'{split}_inputs.json', {'records':[r for r in rows if r['split']==split]})
    for name, part in [('A', train[::2]+[r for r in rows if r['split']=='validation']),
                       ('B', train[1::2]+[r for r in rows if r['split']=='test'])]:
        packet=[]
        for row in part:
            keys=row['keys']
            packet.append(dict(id=row['id'], split=row['split'], fps=row['fps'], keys=keys,
                frames=[dict(i=k, **row['frames'][k]) for k in keys],
                sheet=display_sheet(row,keys,run/'teacher'/'sheets4'/f"{row['id']}.png")))
        write_json(run/'teacher'/f'packet4_{name}.json', dict(author=name,
            caption_prompt=config['caption_prompt'], records=packet,
            rules='Read only these images/labels. Do not read other source frames, original captions, model outputs, or other annotator files. Caption <=80 words; atomic facts and temporal evidence must be visible in these exact frames.'))
    paths=['protocol.json','train_control.json','validation_inputs.json','test_inputs.json',
           'teacher/packet4_A.json','teacher/packet4_B.json']
    write_json(run/'input_freeze.json', dict(files={p:sha256(run/p) for p in paths},
        teacher_precedes_student_generation=True, source_frames={f['path']:f['sha256'] for r in train+rows for f in r['frames']}))
    write_json(run/'RESULT.json', dict(status='TEACHER_ANNOTATION_PENDING', train_windows=len(train), eval_windows=len(rows), default_pipeline_changed=False))
    return verify_inputs(run)


def verify_inputs(run=RUN):
    freeze=read_json(run/'input_freeze.json')
    for p,h in freeze['files'].items():
        if sha256(run/p)!=h:raise ValueError(f'Changed input {p}')
    for p,h in freeze['source_frames'].items():
        if sha256(p)!=h:raise ValueError(f'Changed source {p}')
    return read_json(run/'protocol.json')


def validate_annotation(record, packet_row):
    allowed=set(packet_row['keys'])
    if record['id']!=packet_row['id'] or not record.get('caption') or len(record['caption'].split())>80:
        raise ValueError('Wrong ID/empty/long caption')
    if not record.get('facts') or not isinstance(record.get('uncertainties'),list):
        raise ValueError('Missing facts/uncertainties')
    fact_ids=set()
    for f in record['facts']:
        if f['id'] in fact_ids or f['kind'] not in ['state','event','continuous_motion','camera_motion'] or not f['text']:
            raise ValueError('Bad atomic fact')
        fact_ids.add(f['id'])
        groups=f['evidence_groups']
        if not groups or any(not g or not set(g)<=allowed for g in groups):
            raise ValueError('Unseen fact evidence')
        witness=-1
        for group in groups:
            later=sorted(i for i in group if i>witness)
            if not later:raise ValueError('Reversed evidence order')
            witness=later[0]
        if any(a==b for a,b in zip(groups,groups[1:])):raise ValueError('Repeated intermediate evidence group')
    if record.get('event_annotation_status') not in ['reviewed_observed_frames','uncertain']:
        raise ValueError('Missing event annotation status')
    for e in record['events']:
        if not e['before'] or not e['after'] or not set(e['before']+e['after'])<=allowed or max(e['before'])>=min(e['after']):
            raise ValueError('Unseen/unordered event evidence')
        if e['cause'] not in ['object_motion','camera_motion','occlusion','cut','uncertain']:
            raise ValueError('Missing event cause')
    for entity in record['entities']:
        if entity['visibility'] not in ['visible','partially_visible','occluded','uncertain']:
            raise ValueError('Bad visibility state')
        if not set(entity['evidence'])<=allowed:raise ValueError('Unseen entity evidence')
    return record


def candidate_supervision(annotation, keys, n=8):
    """Explicit positives and reviewed negatives; unannotated slots stay masked."""
    targets=[[0.,0.] for _ in range(n)];mask=[[0.,0.] for _ in range(n)]
    allowed=set(keys)
    for index in annotation.get('confirmed_no_transition_indices',[]):
        if index not in allowed:raise ValueError('Unobserved negative')
        mask[index]=[1.,1.]
    for event in annotation['events']:
        for column,field in enumerate(['before','after']):
            for index in event[field]:
                if index not in allowed:raise ValueError('Unobserved positive')
                targets[index][column]=1.;mask[index][column]=1.
    return dict(targets=targets,mask=mask,observed_indices=keys,
        policy='Unannotated is unknown; empty events does not mean every frame is negative.')


def freeze_teachers(run=RUN):
    verify_inputs(run)
    if (run/'teacher_freeze.json').exists():
        verify_teachers(run);return
    combined={}
    files={}
    for author in ['A','B']:
        packet=read_json(run/'teacher'/f'packet4_{author}.json')
        path=run/'teacher'/f'labels4_{author}.json';labels=read_json(path)
        if labels['packet_sha256']!=sha256(run/'teacher'/f'packet4_{author}.json') or labels['annotator_model']!='gpt-6-astra' or not labels['source_only']:
            raise ValueError('Teacher provenance mismatch')
        by_id={r['id']:r for r in labels['records']}
        if len(by_id)!=len(labels['records']) or set(by_id)!={r['id'] for r in packet['records']}:
            raise ValueError('Incomplete teacher rows')
        if not {r['sheet'] for r in packet['records']}<=set(labels['viewed_sheet_paths']):
            raise ValueError('Not every teacher input inspected')
        for row in packet['records']:
            combined[row['id']]=validate_annotation(by_id[row['id']],row)
        files[str(path.relative_to(run))]=sha256(path)
    old=read_json(run/'train_control.json')['records']
    teacher=[]
    for row in old:
        label=combined[row['id']]
        teacher.append(dict(row,caption=label['caption'],facts=label['facts'],events=label['events'],
            entities=label['entities'],uncertainties=label['uncertainties'],
            event_annotation_status=label['event_annotation_status'],teacher_record=label,
            candidate_supervision=candidate_supervision(label,row['keys'])))
    write_json(run/'train_teacher.json',{'records':teacher})
    files['train_teacher.json']=sha256(run/'train_teacher.json')
    for split in NEW_SOURCES:
        inputs=read_json(run/f'{split}_inputs.json')['records']
        write_json(run/f'{split}_reference4.json',{'records':[dict(r,teacher_record=combined[r['id']]) for r in inputs]})
        files[f'{split}_reference4.json']=sha256(run/f'{split}_reference4.json')
    write_json(run/'teacher_freeze.json',dict(input_freeze_sha256=sha256(run/'input_freeze.json'),files=files,
        annotation_records=len(combined),model='gpt-6-astra',method='Existing explicitly model-configured annotator agents, source-only 4-frame annotation'))
    for author,split in [('A','validation'),('B','test')]:
        packet=[]
        for row in read_json(run/f'{split}_inputs.json')['records']:
            packet.append(dict(id=row['id'],split=split,fps=row['fps'],keys=list(range(8)),
                frames=[dict(i=k,**f) for k,f in enumerate(row['frames'])],
                sheet=display_sheet(row,list(range(8)),run/'teacher'/'sheets8'/f"{row['id']}.png")))
        write_json(run/'teacher'/f'packet8_{author}.json',dict(author=author,records=packet,
            caption_prompt=read_json(run/'protocol.json')['config']['caption_prompt'],
            teacher4_freeze_sha256=sha256(run/'teacher_freeze.json'),
            rules='Eight-frame source-only diagnostic after four-frame reference freeze. Do not modify four-frame annotations.'))
    verify_teachers(run)


def verify_teachers(run=RUN):
    verify_inputs(run)
    freeze=read_json(run/'teacher_freeze.json')
    if freeze['input_freeze_sha256']!=sha256(run/'input_freeze.json'):raise ValueError('Teacher parent changed')
    for p,h in freeze['files'].items():
        if sha256(run/p)!=h:raise ValueError('Teacher changed after freeze')
    return freeze


def load_model(config,adapter=None):
    """Same pinned NF4 path with actual free-memory budgeting and 1 GiB reserve."""
    import torch
    from transformers import BitsAndBytesConfig,Qwen3_5ForConditionalGeneration
    from peft import LoraConfig,PeftModel,get_peft_model
    from .qwen35_training import engine
    eng=engine();stats=eng.base.gpu_status()['gpus'][0]
    # Historical rank-8 4-image training peak is 8.54 GiB. Rank32 parameter,
    # gradient and Adam-state increment is about .97 GiB. Require extra margin.
    minimum=11264 if config['lora_rank']==32 else 10500
    if stats['free_mib']<minimum:raise RuntimeError(f'Need {minimum} MiB free; actual {stats["free_mib"]} MiB')
    torch.cuda.set_per_process_memory_fraction((stats['free_mib']-1024)/stats['total_mib'],0)
    torch.set_num_threads(2);torch.manual_seed(2025)
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,llm_int8_skip_modules=['visual','lm_head'])
    start=time.monotonic()
    model=Qwen3_5ForConditionalGeneration.from_pretrained(eng.base.SNAPSHOT,local_files_only=True,
        trust_remote_code=False,dtype=torch.bfloat16,device_map={'':'cuda:0'},
        quantization_config=quant,attn_implementation='sdpa')
    for name,p in model.named_parameters():
        p.requires_grad_(False)
        if 'norm' in name and 'visual' not in name and p.dtype in (torch.float16,torch.bfloat16):p.data=p.data.float()
    if adapter is None:
        targets=[n for n,m in model.named_modules() if 'language_model.layers.' in n and m.__class__.__name__=='Linear4bit']
        model=get_peft_model(model,LoraConfig(r=config['lora_rank'],lora_alpha=config['lora_alpha'],
            lora_dropout=0.,target_modules=targets,bias='none',task_type='CAUSAL_LM'))
    else:model=PeftModel.from_pretrained(model,str(adapter),is_trainable=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.config.use_cache=False
    model.train();model.base_model.model.model.visual.eval()
    return model,eng.base.load_processor(),dict(model_id=eng.base.MODEL_ID,revision=eng.base.REVISION,
        quantization='nf4',gpu_before=stats,reserve_mib=1024,allocator_budget_mib=stats['free_mib']-1024,
        load_seconds=time.monotonic()-start,rank=config['lora_rank'],alpha=config['lora_alpha'],
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        attention='sdpa',versions=eng.base.packages(),enable_thinking=False)


def arm_config(protocol,arm):
    target,rank=ARMS[arm]
    return dict(protocol['config'],lora_rank=rank,lora_alpha=2*rank),target


def train(run,arm,smoke=False):
    import torch
    from .qwen35_training import caption_loss,save_checkpoint,checkpoint_path,seed_all,release_gpu
    from .qwen35_multitask import TemporalSelector
    p=verify_inputs(run)
    if not smoke:verify_teachers(run)
    config,target=arm_config(p,arm);stage=run/('smoke_'+arm if smoke else arm)
    if (stage/'RESULT.json').exists():
        if not smoke:checkpoint_path(stage)
        print('ALREADY_COMPLETE',stage.name,flush=True);return
    rows=read_json(run/f'train_{"control" if smoke else target}.json')['records']
    identity=dict(input_freeze_sha256=sha256(run/'input_freeze.json'),
        teacher_freeze_sha256=sha256(run/'teacher_freeze.json') if not smoke else None,
        config=config,arm=arm,target=target,epochs=2,lr=p['lr'],smoke=smoke,
        checkpoint_policy=p['checkpoint_policy'],implementation_sha256=sha256(Path(__file__)),
        dependency_sha256={str(path.relative_to(ROOT)):sha256(path) for path in [
            Path(__file__).with_name('qwen35_training.py'),Path(__file__).with_name('qwen35_multitask.py'),
            Path(__file__).with_name('qwen35_caption_repair.py'),ROOT/'scripts/qwen35_advanced_caption.py']})
    if (stage/'protocol.json').exists() and read_json(stage/'protocol.json')!=identity:raise ValueError('Changed training protocol')
    write_json(stage/'protocol.json',identity)
    if not (stage/'implementation.py').exists():
        (stage/'implementation.py').write_bytes(Path(__file__).read_bytes())
    previous=checkpoint_path(stage,'latest') if (stage/'latest.json').exists() else None
    model,processor,runtime=load_model(config,previous/'adapter' if previous else None)
    seed_all(p['seed']);head=TemporalSelector(hidden=config['head_hidden']).cuda().eval()
    original=torch.load(Path(p['checkpoint'])/'head.pt',weights_only=True,map_location='cuda')
    head.load_state_dict(original);head.requires_grad_(False)
    params=[v for v in model.parameters() if v.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=p['lr'])
    state=dict(updates=0,next_epoch=0,next_offset=0,training_seconds=0.,runtime=runtime,
        epoch_losses=[],protocol_sha256=sha256(stage/'protocol.json'))
    if previous:
        state=read_json(previous/'state.json')
        optimizer.load_state_dict(torch.load(previous/'optimizer.pt',weights_only=True,map_location='cuda'))
        rng=torch.load(previous/'rng.pt',weights_only=True,map_location='cpu')
        torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state_all(rng['cuda']);random.setstate(rng['python'])
    features={r['id']:features_for(PARENT,r) for r in (rows[:1] if smoke else rows)}
    def loss(row):return caption_loss(model,processor,head,fixed_prediction(row['keys']),features[row['id']],row,config)
    torch.cuda.reset_peak_memory_stats()
    if previous is None and not smoke:save_checkpoint(stage,head,model,optimizer,state,best=False)
    if smoke:
        step_rows=[rows[0],rows[0]]
        records=[]
        for row in step_rows:
            start=time.monotonic();optimizer.zero_grad(set_to_none=True)
            ce=loss(row);ce.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
            optimizer.step();records.append(dict(loss=float(ce.detach()),gradient_norm=float(norm),
                seconds=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
        write_json(stage/'RESULT.json',dict(status='PASS_CAPACITY_ONLY',records=records,runtime=runtime,
            quality_evaluated=False,checkpoint_retained=False))
    else:
        for epoch in range(state['next_epoch'],p['epochs']):
            order=order_for(rows,p,epoch);acc=p['gradient_accumulation'];values=[];start=time.monotonic()
            for offset in range(0,len(order),acc):
                batch=order[offset:offset+acc];optimizer.zero_grad(set_to_none=True)
                for row in batch:
                    ce=loss(row);values.append(float(ce.detach()));(ce/len(batch)).backward();del ce
                norm=torch.nn.utils.clip_grad_norm_(params,1.)
                if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
                optimizer.step();state['updates']+=1
                write_json(stage/'status.json',dict(status='RUNNING',epoch=epoch+1,updates=state['updates'],examples=offset+len(batch)))
                print('TRAIN',arm,epoch+1,state['updates'],round(sum(values[-len(batch):])/len(batch),4),flush=True)
            if any(not torch.equal(original[k],v) for k,v in head.state_dict().items()):raise ValueError('Frozen head changed')
            state['training_seconds']+=time.monotonic()-start;state['next_epoch']=epoch+1
            state['epoch_losses'].append(sum(values)/len(values));state['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
            save_checkpoint(stage,head,model,optimizer,state,best=epoch+1==p['epochs'])
        write_json(stage/'RESULT.json',dict(status='COMPLETE',final_checkpoint=str(checkpoint_path(stage)),
            state=state,selector_unchanged=True,test_used_for_selection=False))
    del model,processor,head,optimizer,params
    release_gpu()


def generate(run):
    import torch
    from .qwen35_training import checkpoint_path,release_gpu
    from .qwen35_ablation import encode,generate as caption
    verify_teachers(run);p=read_json(run/'protocol.json');dest=run/'evaluation'
    checkpoints={arm:checkpoint_path(run/arm) for arm in ARMS}
    cells=['base',*ARMS,'base_8frames']
    identity=dict(input_freeze_sha256=sha256(run/'input_freeze.json'),teacher_freeze_sha256=sha256(run/'teacher_freeze.json'),
        checkpoints={a:dict(path=str(c),sha256=sha256(c/'complete.json')) for a,c in checkpoints.items()},cells=cells,
        config=p['config'],implementation_sha256=sha256(Path(__file__)),teacher_labels_input=False)
    if (dest/'protocol.json').exists() and read_json(dest/'protocol.json')!=identity:raise ValueError('Generation protocol changed')
    write_json(dest/'protocol.json',identity)
    rows=[r for s in NEW_SOURCES for r in read_json(run/f'{s}_inputs.json')['records']]
    pending=[r for r in rows if not (dest/'records'/f"{r['id']}.json").exists()]
    if pending:
        config,_=arm_config(p,'control_r8')
        model,processor,runtime=load_model(config,checkpoints['control_r8']/'adapter')
        for arm,c in checkpoints.items():
            if arm!='control_r8':model.load_adapter(c/'adapter',adapter_name=arm,is_trainable=False)
        model.requires_grad_(False);model.gradient_checkpointing_disable();model.eval()
        torch.cuda.reset_peak_memory_stats()
        for row in pending:
            features=encode(model,processor,row,p['config']);outputs={}
            for cell in cells:
                model.set_adapter('default' if cell in ['control_r8','base','base_8frames'] else cell)
                keys=list(range(8)) if cell=='base_8frames' else row['keys']
                output=caption(model,processor,row,keys,features,p['config'],adapter=cell not in ['base','base_8frames'])
                outputs[cell]=dict(output,keys=keys,source_hashes=[row['frames'][i]['sha256'] for i in keys])
                print('GENERATE',row['id'],cell,output['status'],flush=True)
            if len({outputs[c]['input_tokens'] for c in cells if c!='base_8frames'})!=1:raise ValueError('Input token mismatch')
            write_json(dest/'records'/f"{row['id']}.json",dict(id=row['id'],split=row['split'],cells=outputs,
                runtime=runtime,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,protocol_sha256=sha256(dest/'protocol.json')))
        del model,processor
        release_gpu()
    write_json(dest/'freeze.json',dict(protocol_sha256=sha256(dest/'protocol.json'),
        files={f"records/{r['id']}.json":sha256(dest/'records'/f"{r['id']}.json") for r in rows}))
    write_json(dest/'RESULT.json',dict(status='GENERATED_REVIEW_PENDING',windows=len(rows),cells=cells))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','verify','freeze-teachers','smoke-r8','smoke-r32','generate','all',*ARMS])
    args=parser.parse_args()
    if args.command in ['prepare','verify']:
        result=prepare() if args.command=='prepare' else verify_inputs()
        print('INPUTS_VERIFIED',result['limited_scope']);return
    if args.command=='freeze-teachers':freeze_teachers();return
    import fcntl
    with open('/tmp/qwen35_multitask_gpu.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command.startswith('smoke-'):train(RUN,'control_'+args.command.split('-')[1],smoke=True)
        elif args.command=='generate':generate(RUN)
        elif args.command=='all':
            for arm in ARMS:train(RUN,arm)
            generate(RUN)
        else:train(RUN,args.command)


if __name__=='__main__':main()
