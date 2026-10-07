"""Expanded source-disjoint teacher data; final test is never a training input."""
from pathlib import Path
import random
from .qwen35_training_data import ROOT,INVENTORY,DEFAULT_DATA,read_json,write_json,sha256,exclusion_keys
from .qwen35_astra_transfer import RUN as PARENT,display_sheet,validate_annotation,verify_teachers
from .qwen35_caption_repair import assert_disjoint
from .qwen35_oracle_diagnostic import witnessed

RUN=ROOT/'outputs/qwen35_grounded_v1'


def split_reserve(reserve):
    rows=sorted(reserve,key=lambda r:r['id']);rng=random.Random(20261006)
    tv=[r for r in rows if r['id'].startswith('tvsum')]
    web=[r for r in rows if r not in tv];rng.shuffle(web)
    # Only one remaining TVSum original; retain it in training for diversity.
    parts={'train':tv+web[:12-len(tv)],'validation':web[12-len(tv):15-len(tv)],'test':web[15-len(tv):]}
    if [len(parts[s]) for s in parts]!=[12,3,4]:raise ValueError('Expected current19-source reserve')
    return parts


def prepare(run=RUN):
    if (run/'input_freeze.json').exists():return verify_inputs(run)
    verify_teachers(PARENT)
    usage=read_json(PARENT/'source_usage.json');parts=split_reserve(usage['remaining_reserve_sources'])
    inventory={r['id']:r for r in read_json(DEFAULT_DATA/'source_split.json')['sources']}
    excluded_ids,excluded_hashes,_=exclusion_keys()
    old=[]
    for s in ['train','validation','test','regression']:
        old+=read_json(ROOT/'outputs/qwen35_caption_repair_v1'/f'{s}.json')['records']
    old+=sum([read_json(PARENT/f'{s}_inputs.json')['records'] for s in ['validation','test']],[])
    old_ids={r['source_id'] for r in old};old_hashes={r['source_sha256'] for r in old}
    rows=[];sources=[]
    for split,items in parts.items():
        for item in items:
            src=inventory[item['id']]
            if src['source_id'] in excluded_ids|old_ids or src['source_sha256'] in excluded_hashes|old_hashes:raise ValueError('Reused/protected source')
            if sha256(src['source_path'])!=src['source_sha256']:raise ValueError('Raw source changed')
            folder=INVENTORY/src['dataset'].lower()/src['id'];meta=read_json(folder/'source_prepared.json')
            if sha256(folder/'normalized.mp4')!=meta['normalized_sha256']:raise ValueError('Normalized video changed')
            sources.append(dict(src,split=split))
            count=len(meta['source_frame_hashes']);span=min(95,count//4-1)
            for w,fraction in enumerate([.125,.375,.625,.875]):
                start=max(0,min(count-span-1,round(fraction*count)-span//2))
                indices=[start+round(i*span/7) for i in range(8)];frames=[]
                for i in indices:
                    path=folder/'source_frames'/f'{i}.png';h=sha256(path)
                    if h!=meta['source_frame_hashes'][str(i)]:raise ValueError('Changed source frame')
                    frames.append(dict(index=i,path=str(path),sha256=h))
                rows.append(dict(id=f'g_{src["id"]}_w{w}',video_id=src['id'],source_id=src['source_id'],source_sha256=src['source_sha256'],
                    frames=frames,fps=24,split=split,window=w,keys=[0,2,5,7],origin='grounded_expansion'))
    replay=read_json(PARENT/'train_teacher.json')['records'];assert_disjoint(replay+rows)
    protocol=dict(seed=20261006,source_selection='19 remaining originals split before source annotation/model outputs; one TVSum allocated to train,18 WebVid shuffled deterministically.',
        source_counts={s:len(x) for s,x in parts.items()},replay_windows=len(replay),new_windows=len(rows),sources=sources,
        parent_teacher_freeze_sha256=sha256(PARENT/'teacher_freeze.json'),parent_usage_sha256=sha256(PARENT/'source_usage.json'),
        final_test_policy='No test labels/images in training, feature cache, preflight, checkpoint selection or step1 development diagnostic.',
        annotation_policy='Source-only same4 images.2-5 atomic facts, chronological evidence, stable/moving/uncertain motion review. Do not infer unobserved negatives.',
        config=read_json(PARENT/'protocol.json')['config'])
    write_json(run/'protocol.json',protocol);write_json(run/'replay.json',{'records':replay})
    for s in parts:write_json(run/f'{s}_inputs.json',{'records':[r for r in rows if r['split']==s]})
    for author,part in [('A',rows[::2]),('B',rows[1::2])]:
        packet=[]
        for row in part:
            packet.append(dict(id=row['id'],split=row['split'],fps=24,keys=row['keys'],
                frames=[dict(i=k,**row['frames'][k]) for k in row['keys']],
                sheet=display_sheet(row,row['keys'],run/'teacher/sheets'/f'{row["id"]}.png')))
        write_json(run/'teacher'/f'packet_{author}.json',dict(author=author,records=packet,caption_prompt=protocol['config']['caption_prompt'],
            rules=protocol['annotation_policy']))
    files=['protocol.json','replay.json',*[f'{s}_inputs.json' for s in parts],'teacher/packet_A.json','teacher/packet_B.json']
    write_json(run/'input_freeze.json',dict(files={f:sha256(run/f) for f in files},source_frames={f['path']:f['sha256'] for r in replay+rows for f in r['frames']},
        sheets={r['sheet']:sha256(r['sheet']) for a in ['A','B'] for r in read_json(run/'teacher'/f'packet_{a}.json')['records']}))
    return verify_inputs(run)


def verify_inputs(run=RUN,training_only=False):
    f=read_json(run/'input_freeze.json')
    for p,h in f['files'].items():
        # Training never opens test label/image files, even for verification.
        if training_only and (p=='test_inputs.json' or p.startswith('teacher/packet')):continue
        if sha256(run/p)!=h:raise ValueError(f'Changed frozen input {p}')
    if not training_only:
        for p,h in {**f['source_frames'],**f['sheets']}.items():
            if sha256(p)!=h:raise ValueError(f'Changed source {p}')
    return read_json(run/'protocol.json')


def validate_grounded(record,row):
    validate_annotation(record,row)
    for f in record['facts']:
        if f['kind']!='state' and len(f['evidence_groups'])<2:raise ValueError('Motion needs ordered evidence')
        if not witnessed(f['evidence_groups'],row['keys']):raise ValueError('Unobserved evidence')
    motion=record['motion_review']
    if set(motion)!={'camera','objects','reason','evidence'}:raise ValueError('Missing motion review')
    if motion['camera'] not in ['moving','stable','uncertain'] or motion['objects'] not in ['moving','stable','uncertain']:raise ValueError('Invalid motion state')
    if not motion['reason'] or not motion['evidence'] or not set(motion['evidence'])<=set(row['keys']):raise ValueError('Unobserved motion review')
    for i in record.get('confirmed_no_transition_indices',[]):
        if i not in row['keys']:raise ValueError('Unobserved negative')
    return record


def freeze(run=RUN):
    verify_inputs(run)
    if (run/'data_freeze.json').exists():return verify_data(run)
    labels={};files={}
    for author in ['A','B']:
        path=run/'teacher'/f'labels_{author}.json';doc=read_json(path);packet=read_json(run/'teacher'/f'packet_{author}.json')
        if doc['packet_sha256']!=sha256(run/'teacher'/f'packet_{author}.json') or doc['annotator_model']!='gpt-6-astra' or not doc['source_only']:raise ValueError('Bad author provenance')
        by_id={r['id']:r for r in doc['records']}
        if len(by_id)!=len(doc['records']) or set(by_id)!={r['id'] for r in packet['records']}:raise ValueError('Missing/extra annotations')
        if not {r['sheet'] for r in packet['records']}<=set(doc['viewed_sheet_paths']):raise ValueError('Missing image inspection')
        for row in packet['records']:labels[row['id']]=validate_grounded(by_id[row['id']],row)
        files[str(path.relative_to(run))]=sha256(path)
    for split in ['train','validation','test']:
        rows=read_json(run/f'{split}_inputs.json')['records']
        enriched=[dict(r,**{k:v for k,v in labels[r['id']].items() if k!='id'}) for r in rows]
        if split=='train':enriched=read_json(run/'replay.json')['records']+enriched
        write_json(run/f'{split}.json',{'records':enriched});files[f'{split}.json']=sha256(run/f'{split}.json')
    write_json(run/'data_freeze.json',dict(input_freeze_sha256=sha256(run/'input_freeze.json'),files=files,
        source_only=True,human_truth=False,training_started=False))
    return verify_data(run)


def verify_data(run=RUN,training_only=False):
    protocol=verify_inputs(run,training_only);freeze=read_json(run/'data_freeze.json')
    if freeze['input_freeze_sha256']!=sha256(run/'input_freeze.json'):raise ValueError('Changed input manifest')
    for p,h in freeze['files'].items():
        if training_only and p not in ['train.json','validation.json']:continue
        if sha256(run/p)!=h:raise ValueError('Changed teacher data')
    splits=['train','validation'] if training_only else ['train','validation','test']
    rows=sum([read_json(run/f'{s}.json')['records'] for s in splits],[]);assert_disjoint(rows)
    for row in rows:
        for k in row['keys']:
            f=row['frames'][k]
            if sha256(f['path'])!=f['sha256']:raise ValueError('Changed visible source')
        for fact in row['facts']:
            if not witnessed(fact['evidence_groups'],row['keys']):raise ValueError('Unsupported target')
    return protocol
